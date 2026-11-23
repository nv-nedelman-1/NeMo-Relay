// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin wiring: parse options, start Relay, and register the OpenCode hooks.
 *
 * Every failure before the hooks are registered leaves the plugin inactive and OpenCode running, unless
 * `failClosed` asks it to refuse model requests and tool calls instead. Inside a hook, a guardrail or
 * request intercept that throws blocks its call, as in Relay's managed calls; Relay's other failures
 * are logged and the call proceeds.
 */
import { Tool } from '@opencode/schema/tool';
import { Effect, Stream, type Scope } from 'effect';

import { llmCalls } from './calls.js';
import { parseConfig } from './config.js';
import { refuse, registerHttpHooks } from './http.js';
import { loadRelay } from './modules.js';
import type { Context } from './opencode.js';
import { acquireRelay } from './runtime.js';
import { registerToolHooks } from './tool.js';
import { Turns } from './turns.js';
import { guarded, message, settle } from './util.js';
import { registerWebSocketHooks } from './websocket.js';

const TURN_EVENTS = [
  'session.execution.started',
  'session.execution.succeeded',
  'session.execution.failed',
  'session.execution.interrupted',
] as const;

/** The OpenCode plugin effect: active for the lifetime of the plugin's scope. */
export const nemoRelay = (ctx: Context): Effect.Effect<void, never, Scope.Scope> =>
  Effect.gen(function* () {
    // Read before validation, so a location whose options are invalid still fails closed when asked to.
    const failClosed = ctx.options.failClosed === true;
    const inactive = (reason: string, log = Effect.logError) =>
      Effect.andThen(log(`NeMo Relay is inactive: ${reason}`), failClosed ? refuseCalls(ctx, reason) : Effect.void);

    let config;
    try {
      // OpenCode finds the project's configuration from a subdirectory too, so paths in it resolve from the project.
      config = parseConfig(ctx.options, ctx.location.project.directory);
    } catch (error) {
      return yield* inactive(`invalid plugin options: ${message(error)}`);
    }
    if (!config.enabled) return;

    const loaded = yield* settle(loadRelay);
    if ('error' in loaded)
      return yield* inactive(`the Node binding did not load: ${message(loaded.error)}`, Effect.logWarning);
    const { relay, host } = loaded.value;
    const started = yield* settle(() => acquireRelay(config, host));
    if ('error' in started) return yield* inactive(message(started.error));
    const activation = started.value;
    yield* Effect.addFinalizer(() => Effect.promise(() => activation.close().catch(() => undefined)));
    for (const diagnostic of activation.diagnostics) yield* Effect.logWarning(`NeMo Relay: ${diagnostic.message}`);

    const turns = new Turns(relay, ctx.app?.version);
    yield* Effect.addFinalizer(() => guarded('shutdown', () => turns.endAll('shutdown')));
    yield* ctx.event.subscribe({ types: TURN_EVENTS }).pipe(
      Stream.runForEach((event) =>
        guarded('turn event', () => {
          const outcome = event.type.slice('session.execution.'.length);
          if (outcome === 'started') turns.get(event.data.sessionID);
          else turns.end(event.data.sessionID, outcome, event.data.error);
        }),
      ),
      Effect.tapError((error) =>
        Effect.logError(
          `NeMo Relay stopped receiving session events, so turns end only at shutdown: ${message(error)}`,
        ),
      ),
      Effect.ignore,
      // The subscription starts when the stream runs, so start it now rather than on a later tick.
      Effect.forkScoped({ startImmediately: true }),
    );

    const llm = llmCalls(relay);
    yield* Effect.addFinalizer(() => Effect.sync(() => llm.endDetached()));
    yield* registerHttpHooks(ctx, llm, turns);
    yield* registerWebSocketHooks(ctx, llm, turns);
    yield* registerToolHooks(ctx, relay, turns);
    yield* Effect.logInfo('NeMo Relay is active');
  });

/** For `failClosed`: refuse every model request and tool call in a location where Relay is not active. */
function refuseCalls(ctx: Context, reason: string): Effect.Effect<void, never, Scope.Scope> {
  const text = `Blocked because NeMo Relay is not active: ${reason}`;
  return Effect.gen(function* () {
    yield* ctx.session.hook('http.request', (event) => refuse(event, text));
    yield* ctx.session.hook('experimental.ws.send', () => Effect.die(new Error(text)));
    yield* ctx.tool.hook('execute.before', () => Effect.fail(new Tool.Error({ message: text })));
    yield* Effect.logError('NeMo Relay fails closed: model requests and tool calls are blocked');
  });
}
