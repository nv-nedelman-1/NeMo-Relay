// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Model calls over HTTP, from OpenCode's `http.request` and `http.response` hooks.
 *
 * Each HTTP request OpenCode sends to a model provider becomes one Relay LLM call. Relay's
 * conditional-execution guardrails and then its request intercepts run before the request leaves the
 * machine, and the call ends with the provider's response, assembled from the stream when the
 * provider streams.
 */
import { Effect, type Scope } from 'effect';

import type { LlmCall, LlmCalls } from './calls.js';
import type { Json } from './modules.js';
import type { Context } from './opencode.js';
import { assembleResponse, tapEventStream } from './stream.js';
import type { Turns } from './turns.js';
import { guarded, isRecord, message, parseJson, settle } from './util.js';

/** Register the HTTP model-call hooks. */
export function registerHttpHooks(ctx: Context, llm: LlmCalls, turns: Turns): Effect.Effect<void, never, Scope.Scope> {
  const calls = new WeakMap<Request, LlmCall>();

  return Effect.gen(function* () {
    yield* ctx.session.hook('http.request', (event) =>
      Effect.gen(function* () {
        const body = yield* settle(() => event.request.clone().text());
        if ('error' in body)
          return yield* Effect.logWarning(`NeMo Relay could not read a model request: ${message(body.error)}`);
        let content = parseJson(body.value);
        if (!isRecord(content)) return;
        // A title request can come after the turn it belongs to has ended, so it joins the turn only if one
        // is open, rather than opening one nothing would close.
        const turn = yield* guarded('turn', () => {
          if (event.kind === 'generate') return turns.outside(event.sessionID);
          if (event.kind === 'title') return turns.peek(event.sessionID) ?? turns.outside(event.sessionID);
          return turns.get(event.sessionID);
        });
        if (!turn) return;
        const url = new URL(event.request.url);
        const route = llm.routeOf(url.pathname);
        const metadata: Record<string, Json> = {
          source: 'opencode.http.request',
          ...turn.identity,
          agent: event.agent,
          kind: event.kind,
          provider: event.model.providerID,
          model: event.model.id,
          'server.address': url.hostname,
          ...(url.port ? { 'server.port': Number(url.port) } : {}),
        };

        const reason = yield* llm.check(turn, content, metadata);
        if (reason !== undefined) return yield* block(event, reason);

        const rewritten = yield* llm.intercept(turn, route, content, metadata);
        if (rewritten && 'reason' in rewritten) return yield* block(event, rewritten.reason);
        // Headers go on the request OpenCode built, so it sends the body bytes it built, which a signed
        // request such as Bedrock's depends on, unless an intercept changed the body.
        for (const [name, value] of Object.entries(rewritten?.headers ?? {})) event.request.headers.set(name, value);
        if (rewritten?.content) {
          content = rewritten.content;
          const headers = new Headers(event.request.headers);
          headers.delete('content-length');
          event.request = new Request(event.request, { headers, body: JSON.stringify(content) });
        }

        const call = yield* llm.start(turn, route, content, metadata);
        if (call) calls.set(event.request, call);
      }),
    );

    yield* ctx.session.hook('http.response', (event) =>
      Effect.sync(() => {
        const call = calls.get(event.request);
        if (!call) return;
        calls.delete(event.request);
        const { response } = event;
        const type = response.headers.get('content-type') ?? '';
        if (response.ok && response.body && type.includes('text/event-stream')) {
          const events: Json[] = [];
          event.response = tapEventStream(
            response,
            (data) => {
              call.firstChunk ??= performance.now();
              const parsed = parseJson(data);
              if (parsed !== undefined) events.push(parsed);
            },
            (outcome) => {
              const assembled = assembleResponse(events);
              // OpenCode stops reading once the response is complete, cancelling a stream the provider keeps open.
              const ended = outcome === 'cancelled' && assembled.finished ? 'success' : outcome;
              llm.end(call, assembled.response, assembled.failed && ended === 'success' ? 'error' : ended);
            },
          );
          return;
        }
        if (response.ok && response.body && type !== '' && !/json|text/.test(type)) {
          // A binary stream, such as Bedrock's event stream, is passed through without decoding.
          event.response = tapEventStream(response, undefined, (outcome) => llm.end(call, null, outcome));
          return;
        }
        // Read a copy, so OpenCode gets the response without waiting for the whole body.
        void response
          .clone()
          .text()
          .then(
            (text) => {
              const body = parseJson(text) ?? text;
              if (response.ok) llm.end(call, body, 'success');
              else llm.end(call, { error: { status: response.status, body } }, 'error');
            },
            (error: unknown) => llm.end(call, { error: { status: response.status, message: message(error) } }, 'error'),
          );
      }),
    );
  });
}

/** Stop a request Relay blocked. */
function block(event: { request: Request }, reason: string): Effect.Effect<void> {
  return refuse(event, `Blocked by NeMo Relay policy: ${reason}`);
}

/**
 * Stop a request without sending it. An OpenCode host whose `http.request` hook accepts a `response`
 * answers the request with a policy error; older hosts can only fail the request, which OpenCode
 * reports as a session error.
 */
export function refuse(event: { request: Request }, text: string): Effect.Effect<void> {
  if (!('response' in event)) return Effect.die(new Error(text));
  (event as { response?: Response }).response = Response.json(
    { error: { message: text, type: 'policy_violation', code: 'nemo_relay_policy' } },
    { status: 403 },
  );
  return Effect.void;
}
