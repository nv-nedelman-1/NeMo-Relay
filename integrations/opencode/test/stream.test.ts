// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Streamed response tap and provider response assembly.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { assembleResponse, tapEventStream, type StreamOutcome } from '../src/stream.js';

const encoder = new TextEncoder();

function eventStream(...chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

describe('tapEventStream', () => {
  it('passes the body through unchanged and reports each event and the end of the stream', async () => {
    const body = 'data: {"a":1}\n\ndata: {"b"\ndata: :2}\r\n\r\n: keepalive\n\ndata: [DONE]\n\n';
    const data: string[] = [];
    const outcomes: StreamOutcome[] = [];
    const tapped = tapEventStream(
      eventStream(body.slice(0, 9), body.slice(9, 27), body.slice(27)),
      (payload) => data.push(payload),
      (outcome) => outcomes.push(outcome),
    );

    assert.equal(await tapped.text(), body);
    assert.deepEqual(data, ['{"a":1}', '{"b"\n:2}', '[DONE]']);
    assert.deepEqual(outcomes, ['success']);
    assert.equal(tapped.headers.get('content-type'), 'text/event-stream');
  });

  it('reports a consumer that stops reading as cancelled', async () => {
    const outcomes: StreamOutcome[] = [];
    const tapped = tapEventStream(
      eventStream('data: 1\n\n', 'data: 2\n\n'),
      () => undefined,
      (outcome) => outcomes.push(outcome),
    );
    const reader = tapped.body!.getReader();
    await reader.read();
    await reader.cancel();

    assert.deepEqual(outcomes, ['cancelled']);
  });
});

describe('assembleResponse', () => {
  it('merges Chat Completions chunks into one chat.completion', () => {
    const { response, failed } = assembleResponse([
      { id: 'c1', created: 1, model: 'm', choices: [{ index: 0, delta: { role: 'assistant', content: 'Look' } }] },
      {
        choices: [
          { index: 0, delta: { content: 'ing', tool_calls: [{ index: 0, id: 't1', function: { name: 're' } }] } },
        ],
      },
      {
        choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { name: 'ad', arguments: '{"path":' } }] } }],
      },
      {
        choices: [
          {
            index: 0,
            delta: { tool_calls: [{ index: 0, function: { arguments: '"a"}' } }] },
            finish_reason: 'tool_calls',
          },
        ],
      },
      { choices: [], usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 } },
    ]);

    assert.deepEqual(response, {
      id: 'c1',
      created: 1,
      model: 'm',
      object: 'chat.completion',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'Looking',
            tool_calls: [{ id: 't1', type: 'function', function: { name: 'read', arguments: '{"path":"a"}' } }],
          },
          finish_reason: 'tool_calls',
        },
      ],
      usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
    });
    assert.equal(failed, false);
  });

  it('returns the terminal response of an OpenAI Responses stream, and whether it failed', () => {
    const completed = { id: 'r1', status: 'completed', output: [], usage: { input_tokens: 4, output_tokens: 1 } };
    const incomplete = { id: 'r2', status: 'incomplete', incomplete_details: { reason: 'max_output_tokens' } };
    const failed = { id: 'r3', status: 'failed', error: { code: 'server_error', message: 'boom' } };
    const delta = { type: 'response.output_text.delta', delta: 'x' };

    assert.deepEqual(assembleResponse([delta, { type: 'response.completed', response: completed }]), {
      response: completed,
      failed: false,
    });
    assert.deepEqual(assembleResponse([delta, { type: 'response.incomplete', response: incomplete }]), {
      response: incomplete,
      failed: false,
    });
    assert.deepEqual(assembleResponse([delta, { type: 'response.failed', response: failed }]), {
      response: failed,
      failed: true,
    });
  });

  it('assembles an Anthropic Messages stream into the message it builds', () => {
    const { response, failed } = assembleResponse([
      {
        type: 'message_start',
        message: {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          model: 'claude',
          content: [],
          usage: { input_tokens: 12, output_tokens: 1 },
        },
      },
      { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Read ' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'it.' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Reading ' } },
      { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'a.txt' } },
      { type: 'content_block_stop', index: 1 },
      {
        type: 'content_block_start',
        index: 2,
        content_block: { type: 'tool_use', id: 'toolu_1', name: 'read', input: {} },
      },
      { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"path":' } },
      { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '"a.txt"}' } },
      { type: 'content_block_stop', index: 2 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 30 } },
      { type: 'message_stop' },
    ]);

    assert.deepEqual(response, {
      id: 'msg_1',
      type: 'message',
      role: 'assistant',
      model: 'claude',
      content: [
        { type: 'thinking', thinking: 'Read it.', signature: 'sig' },
        { type: 'text', text: 'Reading a.txt' },
        { type: 'tool_use', id: 'toolu_1', name: 'read', input: { path: 'a.txt' } },
      ],
      stop_reason: 'tool_use',
      stop_sequence: null,
      usage: { input_tokens: 12, output_tokens: 30 },
    });
    assert.equal(failed, false);
  });

  it('merges Gemini stream chunks into one response', () => {
    const { response } = assembleResponse([
      {
        candidates: [{ index: 0, content: { role: 'model', parts: [{ text: 'Reading ' }] } }],
        modelVersion: 'gemini-2.5-pro',
      },
      {
        candidates: [
          {
            index: 0,
            content: {
              role: 'model',
              parts: [{ text: 'a.txt' }, { functionCall: { name: 'read', args: { path: 'a.txt' } } }],
            },
            finishReason: 'STOP',
          },
        ],
        usageMetadata: { promptTokenCount: 9, candidatesTokenCount: 4, totalTokenCount: 13 },
        modelVersion: 'gemini-2.5-pro',
      },
    ]);

    assert.deepEqual(response, {
      candidates: [
        {
          index: 0,
          finishReason: 'STOP',
          content: {
            role: 'model',
            parts: [{ text: 'Reading a.txt' }, { functionCall: { name: 'read', args: { path: 'a.txt' } } }],
          },
        },
      ],
      usageMetadata: { promptTokenCount: 9, candidatesTokenCount: 4, totalTokenCount: 13 },
      modelVersion: 'gemini-2.5-pro',
    });
  });

  it('returns the events of other providers, and of a failed stream, as sent', () => {
    const cohere = [
      { type: 'message-start', id: 'm1' },
      { type: 'content-delta', index: 0, delta: { message: { content: { text: 'Hi' } } } },
      { type: 'message-end', delta: { finish_reason: 'COMPLETE' } },
    ];
    const anthropic = [
      { type: 'message_start', message: { id: 'msg_1', usage: { input_tokens: 3 } } },
      { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } },
    ];

    assert.deepEqual(assembleResponse(cohere), { response: cohere, failed: false });
    assert.deepEqual(assembleResponse(anthropic), { response: anthropic, failed: true });
  });
});
