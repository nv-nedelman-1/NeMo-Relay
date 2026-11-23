// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin behavior over HTTP and for tools, against the Relay binding.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { describe, it } from 'node:test';

import { Cause, Effect, Exit } from 'effect';
import * as relay from 'nemo-relay-node';

import type { HttpRequestEvent, ToolBeforeEvent } from '../src/opencode.js';
import { Turns } from '../src/turns.js';
import { fixture, harness } from './harness.js';

function modelRequest(content: unknown, url = 'http://127.0.0.1:9/v1/chat/completions'): HttpRequestEvent {
  return {
    sessionID: 'ses_1',
    agent: 'build',
    kind: 'primary',
    model: { id: 'm', providerID: 'mock' },
    request: new Request(url, {
      method: 'POST',
      headers: { authorization: 'Bearer sk-test-credential', 'content-type': 'application/json' },
      body: JSON.stringify(content),
    }),
  };
}

function eventStream(...events: unknown[]): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n';
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

function toolCall(tool: string, input: unknown): ToolBeforeEvent {
  return { tool, sessionID: 'ses_1', agent: 'build', messageID: 'msg_1', id: `call_${tool}`, input };
}

describe('nemo-relay OpenCode plugin', () => {
  it('records a turn with its model and tool calls', async () => {
    const test = harness('');
    await test.run(function* () {
      const request = {
        ...modelRequest({ model: 'm', stream: true, messages: [{ role: 'user', content: 'Read a.txt' }] }),
        model: { id: 'fast', providerID: 'mock' },
      };
      yield* test.hook('http.request', request);
      assert.equal(request.request.headers.get('x-dynamo-session-id'), 'ses_1');
      const response = {
        ...request,
        response: eventStream(
          { id: 'c1', model: 'm', choices: [{ index: 0, delta: { role: 'assistant', content: 'Hel' } }] },
          {
            choices: [{ index: 0, delta: { content: 'lo' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          },
        ),
      };
      yield* test.hook('http.response', response);
      yield* Effect.promise(() => response.response.text());

      const read = toolCall('read', { path: 'a.txt' });
      yield* test.hook('execute.before', read);
      yield* test.hook('execute.after', {
        ...read,
        status: 'completed',
        result: { output: 'hello', content: [{ type: 'text', text: 'hello' }] },
      });
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
    });

    const events = test.events();
    const scopes = events.filter((event) => event.kind === 'scope');
    assert.deepEqual(
      scopes.map((event) => `${event.scope_category} ${event.category} ${event.name}`),
      [
        'start agent opencode',
        'start llm openai.chat.completions',
        'end llm openai.chat.completions',
        'start tool read',
        'end tool read',
        'end agent opencode',
      ],
    );
    assert.deepEqual(scopes[2]!.data, {
      id: 'c1',
      model: 'm',
      object: 'chat.completion',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Hello' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
    });
    // Each turn is its own Relay run, so Relay exports its agent span as a trace root.
    assert.equal(scopes[0]!.propagation_root_uuid, scopes[0]!.uuid);
    assert.equal(scopes[1]!.parent_uuid, scopes[0]!.uuid);
    // The call carries the model its request names; OpenCode's own model ID stays in the metadata.
    assert.equal(scopes[1]!.category_profile?.model_name, 'm');
    assert.equal(scopes[1]!.metadata?.model, 'fast');
    assert.equal(scopes[3]!.metadata?.toolCallId, 'call_read');
    assert.equal(scopes[5]!.metadata?.outcome, 'succeeded');
    for (const start of scopes.filter((event) => event.scope_category === 'start'))
      assert.deepEqual(
        [start.metadata?.session_id, start.metadata?.agent_kind, start.metadata?.agent_version],
        ['ses_1', 'opencode', '2.2.5-test'],
        start.name,
      );
    const metrics = events.filter((event) => event.kind === 'mark' && event.name === 'gen_ai.client.operation');
    const measurements = (mark: (typeof metrics)[number]) =>
      (mark.data as { measurements: { name: string; attributes: Record<string, unknown> }[] }).measurements;
    assert.equal(measurements(metrics[0]!)[0]!.attributes['gen_ai.request.model'], 'm');
    assert.deepEqual(
      metrics.map((mark) => measurements(mark).map(({ name }) => name)),
      [
        [
          'gen_ai.client.operation.duration',
          'gen_ai.client.token.usage',
          'gen_ai.client.token.usage',
          'gen_ai.client.operation.time_to_first_chunk',
        ],
      ],
    );
    assert.ok(!JSON.stringify(events).includes('sk-test-credential'), 'provider credentials stay in OpenCode');
  });

  it('marks failed calls and turns as OpenTelemetry errors', async () => {
    const test = harness('');
    await test.run(function* () {
      const request = modelRequest({ model: 'm', messages: [{ role: 'user', content: 'hi' }] });
      yield* test.hook('http.request', request);
      const failed = Response.json({ error: { message: 'upstream exploded' } }, { status: 500 });
      yield* test.hook('http.response', { ...request, response: failed });
      yield* Effect.sleep('20 millis');
      const read = toolCall('read', { path: 'a.txt' });
      yield* test.hook('execute.before', read);
      yield* test.hook('execute.after', { ...read, status: 'error', error: { message: 'ENOENT: no such file' } });
      test.publish({
        type: 'session.execution.failed',
        data: { sessionID: 'ses_1', error: { type: 'provider', message: 'upstream exploded' } },
      });
      yield* Effect.sleep('50 millis');
    });

    const ends = test.events().filter((event) => event.kind === 'scope' && event.scope_category === 'end');
    assert.deepEqual(
      ends.map((event) => [
        event.category,
        event.metadata?.['otel.status_code'],
        event.metadata?.['error.type'],
        event.metadata?.['otel.status_description'],
      ]),
      [
        ['llm', 'ERROR', '500', 'upstream exploded'],
        ['tool', 'ERROR', 'tool_execution_failed', 'ENOENT: no such file'],
        ['agent', 'ERROR', 'provider', 'upstream exploded'],
      ],
    );
  });

  it('runs guardrails before request intercepts, and sends the body as built unless one changes it', async () => {
    const test = harness(
      `
[[components]]
kind = "test.mask"
enabled = true

[components.config]
secret = "sk-"

[[components]]
kind = "test.policy"
enabled = true

[components.config]
command = "rm -rf"
secret = "sk-live-"
`,
      { relayPlugins: [fixture('mask'), fixture('policy')] },
    );
    await test.run(function* () {
      const live: HttpRequestEvent & { response: Response | undefined } = {
        ...modelRequest({ messages: [{ role: 'user', content: 'use key sk-live-123' }] }),
        response: undefined,
      };
      yield* test.hook('http.request', live);
      assert.equal(live.response?.status, 403, 'the guardrail sees the key before the mask hides it');

      const masked = modelRequest({ messages: [{ role: 'user', content: 'use key sk-test-456' }] });
      yield* test.hook('http.request', masked);
      assert.match(yield* Effect.promise(() => masked.request.text()), /use key \[MASKED\]test-456/);

      const plain = modelRequest({ messages: [{ role: 'user', content: 'hello' }] });
      const built = plain.request;
      yield* test.hook('http.request', plain);
      assert.equal(plain.request, built, 'a request no intercept changed is sent as OpenCode built it');
      assert.equal(plain.request.headers.get('x-dynamo-session-id'), 'ses_1');
    });
  });

  it('blocks a call when a guardrail or request intercept throws', async () => {
    const test = harness(
      `
[[components]]
kind = "test.broken"
enabled = true
`,
      { relayPlugins: [fixture('broken')] },
    );
    const answered = (content: string): HttpRequestEvent & { response: Response | undefined } => ({
      ...modelRequest({ messages: [{ role: 'user', content }] }),
      response: undefined,
    });
    await test.run(function* () {
      for (const word of ['LLM-GUARDRAIL', 'LLM-INTERCEPT']) {
        const request = answered(`mention ${word}`);
        yield* test.hook('http.request', request);
        assert.equal(request.response?.status, 403, word);
        assert.match(
          ((yield* Effect.promise(() => request.response!.json())) as { error: { message: string } }).error.message,
          new RegExp(`${word} broke`),
        );
      }
      for (const word of ['TOOL-GUARDRAIL', 'TOOL-INTERCEPT']) {
        const failure = yield* Effect.flip(test.hook('execute.before', toolCall('shell', { command: `echo ${word}` })));
        assert.equal((failure as { _tag?: unknown })._tag, 'Tool.Error', word);
        assert.match((failure as Error).message, new RegExp(`Blocked by NeMo Relay policy: .*${word} broke`));
      }
      const fine = answered('hello');
      yield* test.hook('http.request', fine);
      assert.equal(fine.response, undefined, 'a request no callback breaks on is sent');
    });

    const blocked = test.events().filter((event) => event.kind === 'mark' && event.name.endsWith('_blocked'));
    assert.deepEqual(
      blocked.map((mark) => mark.name),
      ['opencode.llm_blocked', 'opencode.llm_blocked', 'opencode.tool_blocked', 'opencode.tool_blocked'],
    );
  });

  it('with failClosed, refuses model requests and tool calls when Relay cannot start', async () => {
    const test = harness('', { failClosed: true, relayPlugins: [path.join(path.sep, 'missing', 'policy.js')] });
    await test.run(function* () {
      const request: HttpRequestEvent & { response: Response | undefined } = {
        ...modelRequest({ messages: [{ role: 'user', content: 'hello' }] }),
        response: undefined,
      };
      yield* test.hook('http.request', request);
      assert.equal(request.response?.status, 403);
      assert.match(
        ((yield* Effect.promise(() => request.response!.json())) as { error: { message: string } }).error.message,
        /^Blocked because NeMo Relay is not active: /,
      );

      const failure = yield* Effect.flip(test.hook('execute.before', toolCall('read', { path: 'a.txt' })));
      assert.equal((failure as { _tag?: unknown })._tag, 'Tool.Error');

      const frame = { sessionID: 'ses_1', agent: 'build', kind: 'primary', model: { id: 'm', providerID: 'mock' } };
      const sent = yield* Effect.exit(
        test.hook('experimental.ws.send', { ...frame, frame: '{"type":"response.create"}' }),
      );
      assert.ok(Exit.isFailure(sent) && Cause.hasDies(sent.cause), 'a WebSocket request is not sent');
    });
  });

  it('lets a title call outlive its turn, and ends one still open at shutdown', async () => {
    const test = harness('');
    const title = (sessionID: string) => ({
      ...modelRequest({ model: 'small', stream: true, messages: [{ role: 'user', content: 'Title this' }] }),
      sessionID,
      kind: 'title',
    });
    await test.run(function* () {
      const late = title('ses_1');
      yield* test.hook('http.request', late);
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
      const answered = {
        ...late,
        response: eventStream({ choices: [{ index: 0, delta: { content: 'Greeting' }, finish_reason: 'stop' }] }),
      };
      yield* test.hook('http.response', answered);
      yield* Effect.promise(() => answered.response.text());
      yield* test.hook('http.request', title('ses_2'));
    });

    const ends = test
      .events()
      .filter((event) => event.kind === 'scope' && event.category === 'llm' && event.scope_category === 'end');
    assert.deepEqual(
      ends.map((event) => [event.metadata?.session_id, event.metadata?.outcome]),
      [
        ['ses_1', 'success'],
        ['ses_2', 'cancelled'],
      ],
    );
    assert.match(JSON.stringify(ends[0]!.data), /Greeting/);
  });

  it("records a generate request outside the session's turns", async () => {
    const test = harness('');
    await test.run(function* () {
      // OpenCode warms the prompt cache with `generate` requests while a session is idle.
      const ping = {
        ...modelRequest({ model: 'm', stream: true, messages: [{ role: 'user', content: 'warm' }] }),
        kind: 'generate',
      };
      yield* test.hook('http.request', ping);
      const warmed = { ...ping, response: eventStream({ choices: [{ index: 0, delta: { content: 'ok' } }] }) };
      yield* test.hook('http.response', warmed);
      yield* Effect.promise(() => warmed.response.text());

      yield* test.hook(
        'http.request',
        modelRequest({ model: 'm', stream: true, messages: [{ role: 'user', content: 'hi' }] }),
      );
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
    });

    const scopes = test.events().filter((event) => event.kind === 'scope' && event.scope_category === 'start');
    const agents = scopes.filter((event) => event.category === 'agent');
    assert.equal(agents.length, 1, 'the ping opens no turn');
    assert.deepEqual(
      scopes
        .filter((event) => event.category === 'llm')
        .map((event) => [event.metadata?.kind, event.parent_uuid === agents[0]!.uuid]),
      [
        ['generate', false],
        ['primary', true],
      ],
    );
  });

  it('resolves relative paths in its options against the project when OpenCode runs in a subdirectory', async () => {
    const test = harness('', {}, path.join('packages', 'api'));
    await test.run(function* () {
      yield* test.hook('http.request', modelRequest({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }));
    });

    // The relative `plugins.toml` the harness configures is the project's, so Relay records the call.
    assert.equal(test.events().filter((event) => event.category === 'llm').length, 2);
  });

  it('lets a request it cannot read go ahead unrecorded', async () => {
    const test = harness('');
    await test.run(function* () {
      const body = new ReadableStream<Uint8Array>({ pull: (controller) => controller.error(new Error('broken body')) });
      const unreadable = new Request('http://127.0.0.1:9/v1/chat/completions', {
        method: 'POST',
        body,
        duplex: 'half',
      } as RequestInit);
      const exit = yield* Effect.exit(test.hook('http.request', { ...modelRequest({}), request: unreadable }));
      assert.ok(Exit.isSuccess(exit), 'the request is not failed');
    });

    assert.equal(test.events().filter((event) => event.category === 'llm').length, 0);
  });

  it('ends the model and tool calls still open when the turn ends', async () => {
    const test = harness('');
    await test.run(function* () {
      yield* test.hook('http.request', modelRequest({ model: 'm', messages: [] }));
      yield* test.hook('execute.before', toolCall('read', { path: 'a.txt' }));
      test.publish({ type: 'session.execution.interrupted', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
    });

    const scopes = test.events().filter((event) => event.kind === 'scope');
    assert.deepEqual(
      scopes.map(
        (event) =>
          `${event.scope_category} ${event.name} ${String(event.metadata?.outcome ?? event.metadata?.status ?? '')}`,
      ),
      [
        'start opencode ',
        'start openai.chat.completions ',
        'start read ',
        'end openai.chat.completions cancelled',
        'end read cancelled',
        'end opencode interrupted',
      ],
    );
    assert.deepEqual(
      scopes.filter((event) => event.scope_category === 'end').map((event) => event.metadata?.['otel.status_code']),
      ['UNSET', 'UNSET', 'UNSET'],
      'a cancelled call or interrupted turn is neither OK nor an error',
    );
  });

  it('assembles Anthropic and Gemini streams, and records failed Responses streams as errors', async () => {
    const test = harness('');
    // stream.test.ts covers the assembly itself; this covers the HTTP hooks and the client metrics.
    const anthropic = [
      { type: 'message_start', message: { id: 'msg_1', content: [], usage: { input_tokens: 3, output_tokens: 1 } } },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 2 } },
    ];
    const gemini = [
      {
        candidates: [{ content: { parts: [{ text: 'Hi' }] } }],
        usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 1 },
      },
    ];
    const failed = { id: 'resp_1', status: 'failed', error: { code: 'server_error', message: 'boom' } };
    await test.run(function* () {
      for (const [url, content, events] of [
        ['http://127.0.0.1:9/v1/messages', { model: 'claude', stream: true, messages: [] }, anthropic],
        ['http://127.0.0.1:9/v1beta/models/g:streamGenerateContent?alt=sse', { contents: [] }, gemini],
        [
          'http://127.0.0.1:9/v1/responses',
          { model: 'm', stream: true, input: [] },
          [{ type: 'response.failed', response: failed }],
        ],
      ] as const) {
        const request = modelRequest(content, url);
        yield* test.hook('http.request', request);
        const answered = { ...request, response: eventStream(...events) };
        yield* test.hook('http.response', answered);
        yield* Effect.promise(() => answered.response.text());
      }
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
    });

    const ends = test
      .events()
      .filter((event) => event.kind === 'scope' && event.category === 'llm' && event.scope_category === 'end');
    assert.deepEqual(
      ends.map((event) => [event.name, event.metadata?.outcome]),
      [
        ['anthropic.messages', 'success'],
        ['opencode.http', 'success'],
        ['openai.responses', 'error'],
      ],
    );
    const tokens = test
      .events()
      .filter((event) => event.kind === 'mark' && event.name === 'gen_ai.client.operation')
      .map((mark) =>
        (mark.data as { measurements: { name: string; value: number }[] }).measurements
          .filter(({ name }) => name === 'gen_ai.client.token.usage')
          .map(({ value }) => value),
      );
    assert.deepEqual(tokens, [[3, 2], [4, 1], []]);
  });

  it('passes a binary stream to OpenCode without waiting for it to end', { timeout: 10_000 }, async () => {
    const test = harness('');
    await test.run(function* () {
      const request = modelRequest({ modelId: 'm', messages: [] }, 'http://127.0.0.1:9/model/m/converse-stream');
      yield* test.hook('http.request', request);
      let close = () => {};
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([0, 1, 2]));
          close = () => controller.close();
        },
      });
      const headers = { 'content-type': 'application/vnd.amazon.eventstream' };
      const answered = { ...request, response: new Response(body, { headers }) };
      yield* test.hook('http.response', answered);
      close();
      const bytes = yield* Effect.promise(() => answered.response.arrayBuffer());
      assert.deepEqual(new Uint8Array(bytes), new Uint8Array([0, 1, 2]));
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
    });

    const llm = test.events().filter((event) => event.kind === 'scope' && event.category === 'llm');
    assert.deepEqual(
      llm.map((event) => `${event.scope_category} ${event.name} ${String(event.metadata?.outcome ?? '')}`),
      ['start opencode.http ', 'end opencode.http success'],
    );
  });

  it('shares one Relay activation among the OpenCode locations in a process', async () => {
    const first = harness('');
    const second = harness('', { pluginsToml: path.join(first.directory, 'plugins.toml') });
    const other = harness('');
    const turn = (location: typeof first, sessionID: string) =>
      Effect.gen(function* () {
        const request = { ...modelRequest({ model: 'm', stream: true, messages: [] }), sessionID };
        yield* location.hook('http.request', request);
        const answered = { ...request, response: eventStream({ choices: [{ index: 0, delta: { content: 'ok' } }] }) };
        yield* location.hook('http.response', answered);
        yield* Effect.promise(() => answered.response.text());
        location.publish({ type: 'session.execution.succeeded', data: { sessionID } });
        yield* Effect.sleep('50 millis');
      });
    await first.run(function* () {
      yield* Effect.promise(() =>
        second.run(function* () {
          yield* turn(second, 'ses_2');
        }),
      );
      yield* Effect.promise(() =>
        other.run(function* () {
          assert.ok(!other.active(), 'a location with different options runs without Relay');
        }),
      );
      // The location that closed first released its hold without stopping Relay for this one.
      yield* turn(first, 'ses_1');
    });

    const scopes = first.events().filter((event) => event.kind === 'scope');
    assert.deepEqual(
      scopes
        .filter((event) => event.scope_category === 'start')
        .map((event) => `${event.category} ${String(event.metadata?.session_id)}`),
      ['agent ses_2', 'llm ses_2', 'agent ses_1', 'llm ses_1'],
    );
    assert.equal(scopes.filter((event) => event.scope_category === 'end').length, 4);
  });

  it('blocks tool calls and model requests that a Relay guardrail rejects', async () => {
    const test = harness(
      `
[[components]]
kind = "test.policy"
enabled = true

[components.config]
command = "rm -rf"
secret = "sk-live-"
`,
      { relayPlugins: [fixture('policy')] },
    );
    await test.run(function* () {
      const blocked = yield* Effect.flip(test.hook('execute.before', toolCall('shell', { command: 'rm -rf build' })));
      assert.equal((blocked as { _tag?: unknown })._tag, 'Tool.Error');
      assert.equal((blocked as Error).message, 'Blocked by NeMo Relay policy: rm -rf is not allowed');
      yield* test.hook('execute.before', toolCall('shell', { command: 'ls' }));

      const secret = { messages: [{ role: 'user', content: 'use key sk-live-123' }] };
      const answered: HttpRequestEvent & { response: Response | undefined } = {
        ...modelRequest(secret),
        response: undefined,
      };
      yield* test.hook('http.request', answered);
      assert.equal(answered.response?.status, 403);
      assert.deepEqual(yield* Effect.promise(() => answered.response!.json()), {
        error: {
          message: 'Blocked by NeMo Relay policy: the request contains a secret',
          type: 'policy_violation',
          code: 'nemo_relay_policy',
        },
      });

      const failed = yield* Effect.exit(test.hook('http.request', modelRequest(secret)));
      assert.ok(Exit.isFailure(failed) && Cause.hasDies(failed.cause), 'hosts without `response` fail the request');
    });

    const blocked = test.events().filter((event) => event.kind === 'mark' && event.name.endsWith('_blocked'));
    assert.deepEqual(
      blocked.map((mark) => [mark.name, (mark.data as { reason: string }).reason, mark.metadata?.session_id]),
      [
        ['opencode.tool_blocked', 'rm -rf is not allowed', 'ses_1'],
        ['opencode.llm_blocked', 'the request contains a secret', 'ses_1'],
        ['opencode.llm_blocked', 'the request contains a secret', 'ses_1'],
      ],
    );
  });

  it('opens a turn when the session starts executing, and leaves a late title request outside it', async () => {
    const test = harness('');
    const title = (sessionID: string) => ({
      ...modelRequest({ model: 'small', messages: [{ role: 'user', content: 'Title this' }] }),
      sessionID,
      kind: 'title',
    });
    await test.run(function* () {
      test.publish({ type: 'session.execution.started', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
      yield* test.hook('http.request', title('ses_1'));
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_1' } });
      yield* Effect.sleep('50 millis');
      yield* test.hook('http.request', title('ses_1'));
    });

    const events = test.events();
    const agents = events.filter((event) => event.category === 'agent' && event.scope_category === 'start');
    assert.equal(agents.length, 1, 'a title request after the turn ended opens no turn of its own');
    const [during, after] = events.filter((event) => event.category === 'llm' && event.scope_category === 'start');
    assert.equal(during!.parent_uuid, agents[0]!.uuid, 'a title request during the turn is part of it');
    assert.notEqual(after!.parent_uuid, agents[0]!.uuid);
  });

  it('keeps tool calls with the same ID in different sessions apart', async () => {
    const test = harness('');
    await test.run(function* () {
      const call = (sessionID: string) => ({ ...toolCall('read', { path: sessionID }), sessionID, id: 'call_0' });
      const done = (sessionID: string) => ({
        ...call(sessionID),
        status: 'completed' as const,
        result: { output: `result of ${sessionID}`, content: [] },
      });
      yield* test.hook('execute.before', call('ses_A'));
      yield* test.hook('execute.before', call('ses_B'));
      yield* test.hook('execute.after', done('ses_A'));
      yield* test.hook('execute.after', done('ses_B'));
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_A' } });
      test.publish({ type: 'session.execution.succeeded', data: { sessionID: 'ses_B' } });
      yield* Effect.sleep('50 millis');
    });

    const ends = test.events().filter((event) => event.category === 'tool' && event.scope_category === 'end');
    assert.deepEqual(
      ends.map((event) => [event.metadata?.session_id, event.metadata?.status]),
      [
        ['ses_A', 'completed'],
        ['ses_B', 'completed'],
      ],
    );
    assert.match(JSON.stringify(ends[1]!.data), /result of ses_B/);
  });

  it('does not run the end callback of a call that already ended', () => {
    const turns = new Turns(relay);
    const turn = turns.get('ses_1');
    let ran = 0;
    turn.onEnd(() => ran++)();
    turn.onEnd(() => ran++);
    turns.end('ses_1', 'succeeded');
    assert.equal(ran, 1);
  });
});
