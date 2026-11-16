// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The part of OpenCode's Effect plugin API (`@opencode/plugin/effect`) this integration uses.
 *
 * OpenCode supplies the API to plugins at runtime, so the package declares only the shape it relies
 * on rather than installing `@opencode/plugin`, whose dependencies include OpenCode's client stack.
 */
import type { Tool } from '@opencode/schema/tool';
import type { Effect, Scope, Stream } from 'effect';

type Registration = { readonly dispose: Effect.Effect<void> };
type Hook<Event, Failure = never> = (event: Event) => Effect.Effect<void, Failure>;
type Registered = Effect.Effect<Registration, never, Scope.Scope>;

export type ModelRef = { readonly id: string; readonly providerID: string; readonly variant?: string };

export type HttpRequestEvent = {
  readonly sessionID: string;
  readonly agent: string;
  readonly model: ModelRef;
  readonly kind: string;
  request: Request;
};

export type HttpResponseEvent = {
  readonly sessionID: string;
  readonly agent: string;
  readonly model: ModelRef;
  readonly kind: string;
  readonly request: Request;
  response: Response;
};

export type WebSocketHandshakeEvent = {
  readonly sessionID: string;
  readonly agent: string;
  readonly model: ModelRef;
  readonly kind: string;
  url: string;
  headers: Record<string, string>;
};

export type WebSocketFrameEvent = {
  readonly sessionID: string;
  readonly agent: string;
  readonly model: ModelRef;
  readonly kind: string;
  frame: string;
};

export type ToolBeforeEvent = {
  tool: string;
  readonly sessionID: string;
  readonly agent: string;
  readonly messageID: string;
  readonly id: string;
  input: unknown;
};

export type ToolAfterEvent = {
  readonly tool: string;
  readonly sessionID: string;
  readonly agent: string;
  readonly messageID: string;
  readonly id: string;
  readonly input: unknown;
} & (
  | { readonly status: 'completed'; result: { output?: unknown; content: ReadonlyArray<{ readonly type: string }> } }
  | { readonly status: 'error'; error: { readonly message: string } }
);

export type SessionEvent = {
  readonly type: string;
  /** `error` is set on `session.execution.failed`. */
  readonly data: { readonly sessionID: string; readonly error?: { readonly type: string; readonly message: string } };
};

export interface Context {
  readonly app?: { readonly version: string };
  readonly options: Readonly<Record<string, unknown>>;
  readonly location: { readonly directory: string; readonly project: { readonly directory: string } };
  readonly session: {
    hook(name: 'http.request', callback: Hook<HttpRequestEvent>): Registered;
    hook(name: 'http.response', callback: Hook<HttpResponseEvent>): Registered;
    hook(name: 'experimental.ws.handshake', callback: Hook<WebSocketHandshakeEvent>): Registered;
    hook(name: 'experimental.ws.send' | 'experimental.ws.receive', callback: Hook<WebSocketFrameEvent>): Registered;
  };
  readonly tool: {
    hook(name: 'execute.before', callback: Hook<ToolBeforeEvent, Tool.Error>): Registered;
    hook(name: 'execute.after', callback: Hook<ToolAfterEvent>): Registered;
  };
  readonly event: {
    subscribe(input: { readonly types: ReadonlyArray<string> }): Stream.Stream<SessionEvent, unknown>;
  };
}

export type Plugin = {
  readonly id: string;
  readonly effect: (context: Context) => Effect.Effect<void, never, Scope.Scope>;
};
