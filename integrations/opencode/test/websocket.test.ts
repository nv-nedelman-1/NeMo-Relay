// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Model calls over OpenCode's WebSocket transport, against the Relay binding.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { Cause, Effect, Exit } from 'effect';

import { MAX_SOCKETS } from '../src/websocket.js';
import { fixture, harness } from './harness.js';

const scope = { sessionID: 'ses_ws', agent: 'build', kind: 'primary', model: { id: 'gpt-5', providerID: 'openai' } };
const frame = (body: unknown) => ({ ...scope, frame: JSON.stringify(body) });
const create = (body: Record<string, unknown>) => frame({ type: 'response.create', model: 'gpt-5', ...body });

describe('nemo-relay OpenCode plugin over WebSocket', () => {
  it('records each response and rebuilds the full input of incremental requests', async () => {
    const test = harness('');
    const user = { role: 'user', content: [{ type: 'input_text', text: 'Read a.txt' }] };
    const call = { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'read', arguments: '{"path":"a.txt"}' };
    const result = { type: 'function_call_output', call_id: 'call_1', output: 'hello' };
    const message = { type: 'message', id: 'msg_1', role: 'assistant', content: [{ type: 'output_text', text: 'Hi' }] };
    const second = { id: 'resp_2', model: 'gpt-5', output: [message], usage: { input_tokens: 30, output_tokens: 1 } };
    await test.run(function* () {
      yield* test.hook('experimental.ws.handshake', {
        ...scope,
        url: 'wss://api.example.test/v1/responses',
        headers: {},
      });
      yield* test.hook('experimental.ws.send', create({ instructions: 'Be brief.', input: [user] }));
      for (const event of [
        { type: 'response.created', response: { id: 'resp_1' } },
        { type: 'response.output_item.added', output_index: 0, item: { ...call, arguments: '' } },
        { type: 'response.function_call_arguments.delta', item_id: 'fc_1', delta: call.arguments },
        { type: 'response.output_item.done', output_index: 0, item: call },
        {
          type: 'response.completed',
          response: { id: 'resp_1', output: [call], usage: { input_tokens: 20, output_tokens: 5 } },
        },
      ])
        yield* test.hook('experimental.ws.receive', frame(event));

      yield* test.hook('experimental.ws.send', create({ previous_response_id: 'resp_1', input: [result] }));
      yield* test.hook(
        'experimental.ws.receive',
        frame({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'Hi' }),
      );
      yield* test.hook('experimental.ws.receive', frame({ type: 'response.completed', response: second }));
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_ws' } });
      yield* Effect.sleep('50 millis');
    });

    const events = test.events();
    const llm = events.filter((event) => event.kind === 'scope' && event.category === 'llm');
    assert.deepEqual(
      llm.map((event) => `${event.scope_category} ${event.name}`),
      ['start openai.responses', 'end openai.responses', 'start openai.responses', 'end openai.responses'],
    );
    const request = (llm[2]!.data as { content: Record<string, unknown> }).content;
    assert.deepEqual(request, { model: 'gpt-5', instructions: 'Be brief.', input: [user, call, result] });
    assert.equal(llm[2]!.metadata?.transport, 'websocket');
    assert.equal(llm[2]!.metadata?.session_id, 'ses_ws');
    assert.equal(llm[2]!.metadata?.['server.address'], 'api.example.test');
    assert.deepEqual(llm[3]!.data, second);
    const metrics = events.filter((event) => event.kind === 'mark' && event.name === 'gen_ai.client.operation');
    assert.equal(metrics.length, 2);
    for (const mark of metrics)
      assert.ok(
        (mark.data as { measurements: { name: string }[] }).measurements.some(
          ({ name }) => name === 'gen_ai.client.operation.time_to_first_chunk',
        ),
      );
  });

  it('masks outbound frames after guardrails check them, and closes unfinished responses', async () => {
    const test = harness(
      `
[[components]]
kind = "test.mask"
enabled = true

[components.config]
secret = "sk-live-"

[[components]]
kind = "test.policy"
enabled = true

[components.config]
command = "rm -rf"
secret = "sk-live-999"
`,
      { relayPlugins: [fixture('mask'), fixture('policy')] },
    );
    await test.run(function* () {
      const masked = create({ input: [{ role: 'user', content: 'use key sk-live-123' }] });
      yield* test.hook('experimental.ws.send', masked);
      assert.ok(masked.frame.includes('[MASKED]123') && !masked.frame.includes('sk-live-'), 'the frame sent is masked');
      assert.equal(JSON.parse(masked.frame).type, 'response.create');

      const rejected = yield* Effect.exit(
        test.hook('experimental.ws.send', create({ input: [{ role: 'user', content: 'use key sk-live-999' }] })),
      );
      assert.ok(
        Exit.isFailure(rejected) && Cause.hasDies(rejected.cause),
        'the guardrail sees the key before the mask hides it, and the request is not sent',
      );
      test.publish({ type: 'session.execution.interrupted', data: { sessionID: 'ses_ws' } });
      yield* Effect.sleep('50 millis');
    });

    const llm = test.events().filter((event) => event.kind === 'scope' && event.category === 'llm');
    assert.deepEqual(
      llm.map((event) => `${event.scope_category} ${String(event.metadata?.outcome ?? '')}`),
      ['start ', 'end cancelled'],
    );
    assert.ok(JSON.stringify(llm[0]!.data).includes('[MASKED]123'));
  });

  it('keeps the last exchange of only the most recently used sockets', async () => {
    const test = harness('');
    const first = { role: 'user', content: 'Read a.txt' };
    const second = { role: 'user', content: 'Now b.txt' };
    const on = (sessionID: string, body: unknown) => ({ ...scope, sessionID, frame: JSON.stringify(body) });
    await test.run(function* () {
      for (let index = 0; index <= MAX_SOCKETS; index++) {
        const sessionID = `ses_${index}`;
        const full = { type: 'response.create', model: 'gpt-5', input: [first] };
        yield* test.hook('experimental.ws.send', on(sessionID, full));
        const completed = { type: 'response.completed', response: { id: `resp_${index}`, output: [] } };
        yield* test.hook('experimental.ws.receive', on(sessionID, completed));
      }
      for (const index of [0, MAX_SOCKETS]) {
        const next = {
          type: 'response.create',
          model: 'gpt-5',
          previous_response_id: `resp_${index}`,
          input: [second],
        };
        yield* test.hook('experimental.ws.send', on(`ses_${index}`, next));
      }
    });

    const starts = test
      .events()
      .filter((event) => event.kind === 'scope' && event.category === 'llm' && event.scope_category === 'start');
    const lastRequest = (sessionID: string) => {
      const matching = starts.filter((event) => event.metadata?.session_id === sessionID);
      return (matching[matching.length - 1]!.data as { content: Record<string, unknown> }).content;
    };
    // Relay records only what a request adds to the previous call, so check which continuation was rebuilt.
    assert.equal(lastRequest(`ses_${MAX_SOCKETS}`).previous_response_id, undefined, 'a kept socket is rebuilt');
    assert.deepEqual(
      lastRequest('ses_0'),
      { model: 'gpt-5', previous_response_id: 'resp_0', input: [second] },
      'a continuation on a forgotten socket is recorded as sent',
    );
  });
});
