// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool calls from OpenCode's `execute.before` and `execute.after` hooks.
 *
 * Relay's tool conditional-execution guardrails run before OpenCode executes a tool. A rejection, or
 * a guardrail or intercept that throws, fails that call alone, and the model receives the reason as
 * the tool result. Tool request intercepts can rewrite the arguments, and calls that run become Relay
 * tool spans.
 */
import { Tool } from '@opencode/schema/tool';
import { Effect, type Scope } from 'effect';

import type { Json, RelayRuntime } from './modules.js';
import type { Context } from './opencode.js';
import type { Turn, Turns } from './turns.js';
import { failed, guarded, isRecord, message, otelStatus, rejectionReason, sameJson, settle, toJson } from './util.js';

type Call = { turn: Turn; handle: ReturnType<RelayRuntime['toolCall']>; metadata: Record<string, Json> };

/** Register the tool-call hooks. */
export function registerToolHooks(
  ctx: Context,
  relay: RelayRuntime,
  turns: Turns,
): Effect.Effect<void, never, Scope.Scope> {
  const calls = new Map<string, Call>();
  /** End a call once. It can run when the turn ends, outside any hook, so it never throws. */
  const end = (id: string, result: Json, status: string) => {
    const call = calls.get(id);
    if (!call) return;
    calls.delete(id);
    const description = isRecord(result) && typeof result.error === 'string' ? result.error : undefined;
    const otel = otelStatus(
      status === 'error'
        ? { type: 'tool_execution_failed', description }
        : status === 'cancelled'
          ? 'neutral'
          : undefined,
    );
    try {
      call.turn.run(() => relay.toolCallEnd(call.handle, { result }, null, { ...call.metadata, status, ...otel }));
    } catch (error) {
      console.warn(`NeMo Relay could not record the end of a tool call: ${message(error)}`);
    }
  };

  return Effect.gen(function* () {
    yield* ctx.tool.hook('execute.before', (event) =>
      Effect.gen(function* () {
        const turn = yield* guarded('turn', () => turns.get(event.sessionID));
        if (!turn) return;
        const args = toJson(event.input);
        const metadata: Record<string, Json> = {
          source: 'opencode.execute',
          ...turn.identity,
          agent: event.agent,
          messageId: event.messageID,
          toolCallId: event.id,
        };

        const block = (reason: string) =>
          Effect.andThen(
            guarded('mark', () =>
              turn.run(() =>
                relay.event(
                  'opencode.tool_blocked',
                  turn.scope,
                  { tool: event.tool, arguments: args, reason },
                  metadata,
                ),
              ),
            ),
            Effect.fail(new Tool.Error({ message: `Blocked by NeMo Relay policy: ${reason}` })),
          );

        const checked = yield* settle(() => turn.run(() => relay.toolConditionalExecution(event.tool, args)));
        if ('error' in checked)
          return yield* block(rejectionReason(checked.error) ?? (yield* failed('a tool guardrail', checked.error)));

        const rewritten = yield* settle(() => turn.run(() => relay.toolRequestIntercepts(event.tool, args)));
        if ('error' in rewritten) return yield* block(yield* failed('a tool request intercept', rewritten.error));
        const input = sameJson(rewritten.value, args) ? args : rewritten.value;
        if (input !== args) event.input = input;

        const handle = yield* guarded('tool call', () =>
          turn.run(() => relay.toolCall(event.tool, input, turn.scope, null, null, metadata, event.id)),
        );
        if (!handle) return;
        calls.set(event.id, { turn, handle, metadata });
        // OpenCode skips `execute.after` when the tool is interrupted or a later `execute.before` hook fails.
        turn.onEnd(() => end(event.id, { error: 'The turn ended before the tool finished' }, 'cancelled'));
      }),
    );

    yield* ctx.tool.hook('execute.after', (event) =>
      Effect.sync(() => {
        if (!calls.has(event.id)) return;
        const result =
          event.status === 'completed'
            ? toJson({
                output: event.result.output,
                content: event.result.content.map((part) => (part.type === 'text' ? part : { type: part.type })),
              })
            : { error: event.error.message };
        end(event.id, result, event.status);
      }),
    );
  });
}
