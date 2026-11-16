// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * One Relay agent scope per OpenCode session turn.
 *
 * A turn opens on the first hook OpenCode runs for a session and closes when the session finishes
 * executing. Each turn is its own Relay run and its own trace, rooted at the agent span.
 */
import type { Json, RelayRuntime, ScopeHandle } from './modules.js';
import { otelStatus } from './util.js';

export type Turn = {
  /** The turn's Relay agent scope, the parent of its LLM and tool calls; `null` for calls outside a turn. */
  readonly scope: ScopeHandle | null;
  /** Relay's session-identity metadata keys, for every event the turn records. */
  readonly identity: Readonly<Record<string, Json>>;
  /** Run `callback` with the turn's scope stack active. */
  run<T>(callback: () => T): T;
  /** Run `callback` when the turn ends, before its agent scope closes. */
  onEnd(callback: () => void): void;
};

type OpenTurn = Turn & {
  readonly scope: ScopeHandle;
  readonly closers: Array<() => void>;
};

export class Turns {
  readonly #open = new Map<string, OpenTurn>();

  /** `version` is OpenCode's, recorded as Relay's `agent_version` when OpenCode provides it. */
  constructor(
    private readonly relay: RelayRuntime,
    private readonly version?: string,
  ) {}

  /** The open turn for `sessionID`, opening one if needed. */
  get(sessionID: string): Turn {
    const current = this.#open.get(sessionID);
    if (current) return current;
    const stack = this.relay.createScopeStack();
    const run = <T>(callback: () => T): T => this.relay.withScopeStack(stack, callback) as T;
    const identity = this.#identity(sessionID);
    const scope = run(() =>
      this.relay.pushScope('opencode', this.relay.ScopeType.Agent, null, null, null, {
        source: 'opencode',
        ...identity,
      }),
    );
    const closers: Array<() => void> = [];
    const turn: OpenTurn = { scope, identity, closers, run, onEnd: (callback) => closers.push(callback) };
    this.#open.set(sessionID, turn);
    return turn;
  }

  /**
   * For a request OpenCode makes outside the session's executions, such as a cache-warming `generate`
   * request while the session is idle: its calls get no agent scope, and no turn waits for them.
   */
  outside(sessionID: string): Turn {
    const stack = this.relay.createScopeStack();
    return {
      scope: null,
      identity: this.#identity(sessionID),
      run: <T>(callback: () => T): T => this.relay.withScopeStack(stack, callback) as T,
      onEnd: () => undefined,
    };
  }

  /** Close the open turn for `sessionID`, recording how it ended and, if it failed, OpenCode's error. */
  end(sessionID: string, outcome: string, error?: { readonly type: string; readonly message: string }): void {
    const turn = this.#open.get(sessionID);
    if (!turn) return;
    this.#open.delete(sessionID);
    for (const close of turn.closers.splice(0)) close();
    const status = otelStatus(
      outcome === 'failed'
        ? { type: error?.type ?? 'execution_failed', description: error?.message }
        : outcome === 'succeeded'
          ? undefined
          : 'neutral',
    );
    turn.run(() => this.relay.popScope(turn.scope, null, null, { outcome, ...status }));
  }

  /** Close every open turn, for plugin shutdown. */
  endAll(outcome: string): void {
    for (const sessionID of [...this.#open.keys()]) this.end(sessionID, outcome);
  }

  #identity(sessionID: string): Record<string, Json> {
    return {
      session_id: sessionID,
      agent_kind: 'opencode',
      ...(this.version ? { agent_version: this.version } : {}),
    };
  }
}
