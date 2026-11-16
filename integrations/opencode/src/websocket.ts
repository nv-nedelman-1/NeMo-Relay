// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Model calls over OpenCode's WebSocket transport, from its experimental `ws.*` hooks.
 *
 * OpenCode keeps one socket per session and runs one response at a time on it, so each outbound
 * `response.create` frame starts a Relay LLM call and the frames that follow end it at the first
 * terminal event. After the first step OpenCode usually sends only the new input with
 * `previous_response_id`; the integration rebuilds the full request from the previous request and
 * its output, so Relay records each call's complete input, as it does over HTTP.
 *
 * Guardrails check the full request, then request intercepts rewrite the outbound frame. A request a
 * guardrail rejects is not sent, but the WebSocket hooks cannot answer it, so OpenCode reports a
 * session error; use the HTTP transport where a blocked request should end with a policy error instead.
 */
import { Effect, type Scope } from 'effect';

import type { LlmCall, LlmCalls } from './calls.js';
import type { Json } from './modules.js';
import type { Context } from './opencode.js';
import type { Turns } from './turns.js';
import { guarded, isRecord, parseJson } from './util.js';

const TERMINAL = new Set(['response.completed', 'response.incomplete', 'response.failed', 'error']);
/**
 * Sockets whose last exchange is kept. OpenCode does not report a socket closing, so the least
 * recently used one beyond this many is forgotten; a continuation on it is recorded as sent.
 */
export const MAX_SOCKETS = 32;

/** The last completed exchange on a session's socket: what an incremental frame continues. */
type Exchange = { id: string; request: Record<string, Json>; output: Json[] };
type Socket = {
  url?: URL;
  last?: Exchange | undefined;
  active?: { call: LlmCall; request: Record<string, Json> } | undefined;
};

/** Register the WebSocket model-call hooks. */
export function registerWebSocketHooks(
  ctx: Context,
  llm: LlmCalls,
  turns: Turns,
): Effect.Effect<void, never, Scope.Scope> {
  const sockets = new Map<string, Socket>();
  const socketOf = (sessionID: string): Socket => {
    const socket = sockets.get(sessionID) ?? {};
    sockets.delete(sessionID);
    sockets.set(sessionID, socket);
    if (sockets.size > MAX_SOCKETS) sockets.delete(sockets.keys().next().value!);
    return socket;
  };
  const route = llm.routeOf('/responses');

  return Effect.gen(function* () {
    yield* ctx.session.hook('experimental.ws.handshake', (event) =>
      Effect.sync(() => {
        if (URL.canParse(event.url)) socketOf(event.sessionID).url = new URL(event.url);
      }),
    );

    yield* ctx.session.hook('experimental.ws.send', (event) =>
      Effect.gen(function* () {
        const frame = parseJson(event.frame);
        if (!isRecord(frame) || frame.type !== 'response.create') return;
        const turn = yield* guarded('turn', () => turns.get(event.sessionID));
        if (!turn) return;
        const socket = socketOf(event.sessionID);
        const { type, ...body } = frame;
        const metadata: Record<string, Json> = {
          source: 'opencode.ws.send',
          ...turn.identity,
          agent: event.agent,
          kind: event.kind,
          provider: event.model.providerID,
          model: event.model.id,
          transport: 'websocket',
          ...(socket.url ? { 'server.address': socket.url.hostname } : {}),
          ...(socket.url?.port ? { 'server.port': Number(socket.url.port) } : {}),
        };

        const block = (reason: string) => Effect.die(new Error(`Blocked by NeMo Relay policy: ${reason}`));
        const reason = yield* llm.check(turn, expand(body, socket.last), metadata);
        if (reason !== undefined) return yield* block(reason);

        const rewritten = yield* llm.intercept(turn, route, body, metadata);
        if (rewritten && 'reason' in rewritten) return yield* block(rewritten.reason);
        const sent = rewritten?.content ?? body;
        if (rewritten?.content) event.frame = JSON.stringify({ type, ...sent });
        const request = expand(sent, socket.last);

        if (socket.active) {
          llm.end(socket.active.call, { error: { message: 'A new request replaced this response' } }, 'cancelled');
          socket.active = undefined;
        }
        const call = yield* llm.start(turn, route, request, metadata);
        if (call) socket.active = { call, request };
      }),
    );

    yield* ctx.session.hook('experimental.ws.receive', (event) =>
      Effect.sync(() => {
        const socket = sockets.get(event.sessionID);
        const active = socket?.active;
        if (!socket || !active) return;
        const data = parseJson(event.frame);
        if (!isRecord(data) || typeof data.type !== 'string') return;
        if (data.type.endsWith('.delta')) active.call.firstChunk ??= performance.now();
        if (!TERMINAL.has(data.type)) return;
        socket.active = undefined;
        const response = isRecord(data.response) ? data.response : undefined;
        // OpenCode continues only a completed response; anything else makes its next frame a full one.
        socket.last =
          data.type === 'response.completed' && response && typeof response.id === 'string'
            ? {
                id: response.id,
                request: active.request,
                output: Array.isArray(response.output) ? response.output : [],
              }
            : undefined;
        const succeeded = data.type === 'response.completed' || data.type === 'response.incomplete';
        llm.end(active.call, response ?? data, succeeded ? 'success' : 'error');
      }),
    );
  });
}

/** The full request an incremental frame continues, or the frame as sent when it is not a continuation. */
function expand(body: Record<string, Json>, last: Exchange | undefined): Record<string, Json> {
  const { previous_response_id: previous, input, ...fields } = body;
  if (typeof previous !== 'string' || last?.id !== previous) return body;
  return { ...last.request, ...fields, input: [...list(last.request.input), ...last.output, ...list(input)] };
}

function list(value: Json | undefined): Json[] {
  if (Array.isArray(value)) return value;
  return value === undefined ? [] : [value];
}
