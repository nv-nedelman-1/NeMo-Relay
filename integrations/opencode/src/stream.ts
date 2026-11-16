// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Streamed provider responses.
 *
 * OpenCode reads the response an `http.response` hook leaves in place, so the integration swaps in
 * a pass-through body that collects the server-sent events, then assembles the final provider
 * response Relay records for the LLM call.
 */
import type { Json } from './modules.js';
import { isRecord, parseJson } from './util.js';

export type StreamOutcome = 'success' | 'cancelled' | 'error';

/**
 * A copy of `response` whose body yields the same bytes while `onData`, if given, receives each SSE
 * `data` payload. `onEnd` runs once: at end of stream, on cancellation, or on a read error.
 */
export function tapEventStream(
  response: Response,
  onData: ((data: string) => void) | undefined,
  onEnd: (outcome: StreamOutcome) => void,
): Response {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let data: string[] = [];
  let ended = false;
  const end = (outcome: StreamOutcome) => {
    if (ended) return;
    ended = true;
    onEnd(outcome);
  };
  const dispatch = () => {
    if (data.length > 0) onData?.(data.join('\n'));
    data = [];
  };
  const parse = (text: string) => {
    buffer += text;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (line === '') dispatch();
      else if (line.startsWith('data:')) data.push(line.slice(line.startsWith('data: ') ? 6 : 5));
    }
  };
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (onData) parse(done ? decoder.decode() + '\n' : decoder.decode(value, { stream: true }));
        if (done) {
          dispatch();
          controller.close();
          end('success');
          return;
        }
        controller.enqueue(value);
      } catch (error) {
        controller.error(error);
        end('error');
      }
    },
    cancel(reason) {
      end('cancelled');
      return reader.cancel(reason);
    },
  });
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}

type ChatToolCall = { index?: number; id?: string; type?: string; function?: { name?: string; arguments?: string } };
type ChatChoice = {
  index?: number;
  delta?: { role?: string; content?: string | null; tool_calls?: ChatToolCall[] };
  finish_reason?: string | null;
};
type ChatChunk = { id?: string; created?: number; model?: string; choices?: ChatChoice[]; usage?: unknown };

const RESPONSES_TERMINAL = new Set(['response.completed', 'response.incomplete', 'response.failed']);

/**
 * The provider response a stream's events add up to, and whether the stream reported a failure: the
 * response a non-streamed call would have returned, which Relay decodes, for OpenAI Chat Completions
 * and Responses, Anthropic Messages, and Gemini streams; otherwise, and for a failed stream, the events
 * as sent.
 */
export function assembleResponse(events: readonly Json[]): { response: Json; failed: boolean } {
  const failed = events.some(
    (event) => isRecord(event) && (event.type === 'response.failed' || event.type === 'error'),
  );
  for (const event of events)
    if (isRecord(event) && RESPONSES_TERMINAL.has(String(event.type)) && isRecord(event.response))
      return { response: event.response, failed };
  const has = (test: (event: Record<string, Json>) => boolean) =>
    events.some((event) => isRecord(event) && test(event));
  if (failed) return { response: [...events], failed };
  if (has((event) => event.type === 'message_start')) return { response: anthropicMessage(events), failed };
  if (has((event) => Array.isArray(event.candidates))) return { response: geminiResponse(events), failed };
  if (!has((event) => Array.isArray(event.choices))) return { response: [...events], failed };
  const choices = new Map<
    number,
    { role: string; content: string; tools: Map<number, Required<ChatToolCall>>; finish: string | null }
  >();
  const head: { id?: string | undefined; created?: number | undefined; model?: string | undefined } = {};
  let usage: unknown;
  for (const event of events as readonly ChatChunk[]) {
    head.id ??= event.id;
    head.created ??= event.created;
    head.model ??= event.model;
    usage = event.usage ?? usage;
    for (const choice of event.choices ?? []) {
      const index = choice.index ?? 0;
      const state = choices.get(index) ?? { role: 'assistant', content: '', tools: new Map(), finish: null };
      choices.set(index, state);
      state.role = choice.delta?.role ?? state.role;
      state.content += choice.delta?.content ?? '';
      state.finish = choice.finish_reason ?? state.finish;
      for (const call of choice.delta?.tool_calls ?? []) {
        const slot = call.index ?? 0;
        const tool = state.tools.get(slot) ?? {
          index: slot,
          id: '',
          type: 'function',
          function: { name: '', arguments: '' },
        };
        state.tools.set(slot, tool);
        tool.id = call.id ?? tool.id;
        tool.type = call.type ?? tool.type;
        tool.function = {
          name: tool.function.name + (call.function?.name ?? ''),
          arguments: tool.function.arguments + (call.function?.arguments ?? ''),
        };
      }
    }
  }
  const response = {
    ...head,
    object: 'chat.completion',
    choices: [...choices].map(([index, state]) => ({
      index,
      message: {
        role: state.role,
        content: state.content === '' && state.tools.size > 0 ? null : state.content,
        ...(state.tools.size > 0 ? { tool_calls: [...state.tools.values()].map(({ index: _, ...call }) => call) } : {}),
      },
      finish_reason: state.finish,
    })),
    ...(usage === undefined ? {} : { usage }),
  };
  return { response: response as Json, failed };
}

/** An Anthropic Messages stream as the `message` it builds, block by block. */
function anthropicMessage(events: readonly Json[]): Json {
  let message: Record<string, Json> = {};
  const blocks: Record<string, Json>[] = [];
  const inputs = new Map<number, string>();
  for (const event of events) {
    if (!isRecord(event)) continue;
    const index = Number(event.index);
    if (event.type === 'message_start' && isRecord(event.message)) message = { ...event.message };
    if (event.type === 'content_block_start' && isRecord(event.content_block))
      blocks[index] = { ...event.content_block };
    if (event.type === 'message_delta') {
      const usage = {
        ...(isRecord(message.usage) ? message.usage : {}),
        ...(isRecord(event.usage) ? event.usage : {}),
      };
      message = { ...message, ...(isRecord(event.delta) ? event.delta : {}), usage };
    }
    const block = blocks[index];
    const delta = event.type === 'content_block_delta' && isRecord(event.delta) ? event.delta : undefined;
    if (!block || !delta) continue;
    if (delta.type === 'text_delta') block.text = `${block.text ?? ''}${delta.text ?? ''}`;
    if (delta.type === 'thinking_delta') block.thinking = `${block.thinking ?? ''}${delta.thinking ?? ''}`;
    if (delta.type === 'signature_delta') block.signature = delta.signature ?? null;
    if (delta.type === 'input_json_delta') inputs.set(index, `${inputs.get(index) ?? ''}${delta.partial_json ?? ''}`);
  }
  for (const [index, json] of inputs) if (blocks[index] && json) blocks[index].input = parseJson(json) ?? json;
  return { ...message, content: blocks.filter(Boolean) };
}

/** Gemini `streamGenerateContent` chunks as one `GenerateContentResponse`, each candidate's text joined. */
function geminiResponse(chunks: readonly Json[]): Json {
  let response: Record<string, Json> = {};
  const candidates = new Map<number, Record<string, Json> & { parts: Json[] }>();
  const plainText = (part: Json | undefined): part is Record<string, Json> =>
    isRecord(part) &&
    typeof part.text === 'string' &&
    Object.keys(part).every((key) => key === 'text' || key === 'thought');
  for (const chunk of chunks) {
    if (!isRecord(chunk)) continue;
    response = { ...response, ...chunk };
    for (const candidate of Array.isArray(chunk.candidates) ? chunk.candidates : []) {
      if (!isRecord(candidate)) continue;
      const index = typeof candidate.index === 'number' ? candidate.index : 0;
      const { content, ...fields } = candidate;
      const merged = { ...(candidates.get(index) ?? { parts: [] }), ...fields };
      if (isRecord(content)) merged.role = content.role ?? null;
      for (const part of isRecord(content) && Array.isArray(content.parts) ? content.parts : []) {
        const last = merged.parts.at(-1);
        if (plainText(part) && plainText(last) && last.thought === part.thought) last.text = `${last.text}${part.text}`;
        else merged.parts.push(isRecord(part) ? { ...part } : part);
      }
      candidates.set(index, merged);
    }
  }
  const merged = [...candidates.values()].map(({ parts, role, ...fields }) => ({
    ...fields,
    content: { role: role ?? 'model', parts },
  }));
  return { ...response, candidates: merged };
}
