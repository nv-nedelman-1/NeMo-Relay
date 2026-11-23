// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Relay LLM calls for OpenCode's model requests, shared by the HTTP and WebSocket transports.
 *
 * Relay receives request bodies without headers, so provider credentials stay in OpenCode.
 */
import { Effect } from 'effect';

import type { Json, LlmCodec, LlmRequestShape, RelayRuntime } from './modules.js';
import type { StreamOutcome } from './stream.js';
import type { Turn } from './turns.js';
import { failed, guarded, isRecord, message, otelStatus, rejectionReason, sameJson, settle } from './util.js';

export type Route = {
  name: string;
  decode?: (request: LlmRequestShape) => Json;
  encode?: (request: { annotated: Json; original: LlmRequestShape }) => LlmRequestShape;
};

export type LlmCall = {
  turn: Turn;
  handle: ReturnType<RelayRuntime['llmCall']>;
  metadata: Record<string, Json>;
  /** The model the request names, which can differ from OpenCode's model ID. */
  model: string;
  started: number;
  firstChunk?: number;
  ended?: boolean;
  /** Stop the turn from ending the call, once it has ended. */
  release?: () => void;
};

// OpenTelemetry GenAI semantic-convention bucket boundaries for token counts and durations.
const TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864];
const SECONDS_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92];

export type LlmCalls = ReturnType<typeof llmCalls>;

export function llmCalls(relay: RelayRuntime) {
  const chat = codecRoute('openai.chat.completions', new relay.OpenAIChatCodec());
  const responses = codecRoute('openai.responses', new relay.OpenAIResponsesCodec());
  /** Open calls no turn ends: OpenCode makes title and `generate` requests without an execution waiting for them. */
  const detached = new Set<LlmCall>();

  /**
   * GenAI client metrics for a finished call. Relay derives time to first chunk only for streams it
   * executes itself, so the integration reports the client metrics for the calls it observes.
   */
  function measurements(call: LlmCall, response: Json, outcome: StreamOutcome, failure: Failure | undefined) {
    // Gemini names its usage and model fields differently from OpenAI and Anthropic.
    const { usage, usageMetadata, model, modelVersion } = isRecord(response) ? response : {};
    const tokens = isRecord(usage) ? usage : isRecord(usageMetadata) ? usageMetadata : {};
    const responseModel = model ?? modelVersion;
    const attributes = {
      'gen_ai.operation.name': 'chat',
      'gen_ai.provider.name': String(call.metadata.provider),
      'gen_ai.request.model': call.model,
      ...(typeof responseModel === 'string' ? { 'gen_ai.response.model': responseModel } : {}),
      ...(outcome === 'success' ? {} : { 'error.type': failure?.type ?? outcome }),
    };
    const histogram = (name: string, value: number, unit: string, boundaries: number[], extra = {}) => ({
      name,
      kind: relay.MetricKind.Histogram,
      valueType: relay.MetricValueType.F64,
      value,
      unit,
      boundaries,
      attributes: { ...attributes, ...extra },
    });
    const seconds = (at: number) => (at - call.started) / 1000;
    const input = tokens.prompt_tokens ?? tokens.input_tokens ?? tokens.promptTokenCount;
    const output = tokens.completion_tokens ?? tokens.output_tokens ?? tokens.candidatesTokenCount;
    return [
      histogram('gen_ai.client.operation.duration', seconds(performance.now()), 's', SECONDS_BUCKETS),
      ...(typeof input === 'number'
        ? [histogram('gen_ai.client.token.usage', input, '{token}', TOKEN_BUCKETS, { 'gen_ai.token.type': 'input' })]
        : []),
      ...(typeof output === 'number'
        ? [histogram('gen_ai.client.token.usage', output, '{token}', TOKEN_BUCKETS, { 'gen_ai.token.type': 'output' })]
        : []),
      ...(call.firstChunk === undefined
        ? []
        : [histogram('gen_ai.client.operation.time_to_first_chunk', seconds(call.firstChunk), 's', SECONDS_BUCKETS)]),
    ];
  }

  /** Record a request Relay blocked as an `opencode.llm_blocked` mark; the reason, for the caller to answer with. */
  function blocked(turn: Turn, metadata: Record<string, Json>, reason: string) {
    return Effect.as(
      guarded('mark', () =>
        turn.run(() =>
          relay.event('opencode.llm_blocked', turn.scope, { model: metadata.model ?? null, reason }, metadata),
        ),
      ),
      reason,
    );
  }

  /** End a call once. It can run from a stream OpenCode is reading or when the turn ends, so it never throws. */
  function end(call: LlmCall, response: Json, outcome: StreamOutcome) {
    if (call.ended) return;
    call.ended = true;
    call.release?.();
    detached.delete(call);
    const failure = outcome === 'error' ? failureOf(response) : undefined;
    const status = otelStatus(failure ?? (outcome === 'cancelled' ? 'neutral' : undefined));
    try {
      call.turn.run(() => {
        relay.llmCallEnd(call.handle, response, null, { ...call.metadata, outcome, ...status });
        relay.metric('gen_ai.client.operation', measurements(call, response, outcome, failure), call.turn.scope);
      });
    } catch (error) {
      console.warn(`NeMo Relay could not record the end of an LLM call: ${message(error)}`);
    }
  }

  return {
    /** The codec-backed Relay route for a provider endpoint path. */
    routeOf(pathname: string): Route {
      if (pathname.endsWith('/chat/completions')) return chat;
      if (pathname.endsWith('/responses')) return responses;
      return { name: pathname.endsWith('/messages') ? 'anthropic.messages' : 'opencode.http' };
    },

    /**
     * Run Relay's request intercepts: the headers they set, which include Relay's own session and trace
     * headers, and the content when they changed it. An intercept that throws blocks the request.
     */
    intercept(turn: Turn, route: Route, content: Record<string, Json>, metadata: Record<string, Json>) {
      return Effect.gen(function* () {
        const intercepted = yield* settle(() =>
          turn.run(() => relay.llmRequestIntercepts(route.name, { headers: {}, content })),
        );
        if ('error' in intercepted)
          return {
            reason: yield* blocked(turn, metadata, yield* failed('an LLM request intercept', intercepted.error)),
          };
        const rewritten = intercepted.value.request as { headers?: Record<string, string>; content?: Json };
        if (!isRecord(rewritten.content)) return undefined;
        return {
          headers: rewritten.headers ?? {},
          content: sameJson(rewritten.content, content) ? undefined : rewritten.content,
        };
      });
    },

    /** Run Relay's LLM guardrails: why the request is blocked, or `undefined` to proceed. A guardrail that throws blocks it. */
    check(turn: Turn, content: Record<string, Json>, metadata: Record<string, Json>) {
      return Effect.gen(function* () {
        const checked = yield* settle(() => turn.run(() => relay.llmConditionalExecution({ headers: {}, content })));
        if (!('error' in checked)) return undefined;
        const reason = rejectionReason(checked.error) ?? (yield* failed('an LLM guardrail', checked.error));
        return yield* blocked(turn, metadata, reason);
      });
    },

    /**
     * Start a call. One still open when the turn ends, such as a request whose connection failed, ends
     * then, except a title or `generate` call, which ends with its own response or at shutdown.
     */
    start(turn: Turn, route: Route, content: Record<string, Json>, metadata: Record<string, Json>) {
      const model = typeof content.model === 'string' ? content.model : String(metadata.model);
      return Effect.map(
        guarded('LLM call', () =>
          turn.run(() =>
            relay.llmCall(
              route.name,
              { headers: {}, content },
              turn.scope,
              null,
              null,
              metadata,
              model,
              null,
              route.decode,
              route.encode,
            ),
          ),
        ),
        (handle): LlmCall | undefined => {
          if (!handle) return undefined;
          const call: LlmCall = { turn, handle, metadata, model, started: performance.now() };
          if (metadata.kind === 'title' || metadata.kind === 'generate') detached.add(call);
          else
            call.release = turn.onEnd(() =>
              end(call, { error: { message: 'The turn ended before the response completed' } }, 'cancelled'),
            );
          return call;
        },
      );
    },

    end,

    /** End the title and `generate` calls still open, for plugin shutdown. */
    endDetached() {
      for (const call of [...detached])
        end(call, { error: { message: 'NeMo Relay stopped before the response completed' } }, 'cancelled');
    },
  };
}

type Failure = { type: string; description?: string | undefined };

/**
 * How a failed call failed, from what the integration records: an HTTP error status and body, a failed
 * OpenAI Responses object, or a stream's error event.
 */
function failureOf(response: Json): Failure {
  const reported = Array.isArray(response)
    ? response.find((event) => isRecord(event) && (event.type === 'error' || isRecord(event.error)))
    : response;
  const error = isRecord(reported) && isRecord(reported.error) ? reported.error : reported;
  if (!isRecord(error)) return { type: 'error' };
  const body = isRecord(error.body) && isRecord(error.body.error) ? error.body.error : {};
  const text = (...values: unknown[]) => values.find((value): value is string => typeof value === 'string');
  return {
    type: typeof error.status === 'number' ? String(error.status) : (text(error.code, error.type) ?? 'error'),
    description: text(body.message, error.message),
  };
}

function codecRoute(name: string, codec: LlmCodec): Route {
  return {
    name,
    decode: (request) => codec.decode(request),
    encode: ({ annotated, original }) => codec.encode(annotated, original),
  };
}
