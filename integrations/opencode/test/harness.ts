// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A stand-in for OpenCode's plugin context that records the plugin's hooks, publishes session events
 * on demand, and reads back the events Relay wrote to an ATOF file sink.
 */
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Effect, Stream } from 'effect';

import type { Context, SessionEvent } from '../src/opencode.js';
import { nemoRelay } from '../src/plugin.js';

export type AtofEvent = {
  kind: string;
  uuid: string;
  parent_uuid?: string | null;
  propagation_root_uuid?: string | null;
  scope_category?: string;
  category?: string;
  category_profile?: { model_name?: string };
  name: string;
  data?: unknown;
  metadata?: Record<string, unknown>;
};

type Hook = (event: unknown) => Effect.Effect<void, unknown>;

export const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}.js`, import.meta.url));

/** `subdirectory`, relative to the project directory, is where OpenCode runs: its location. */
export function harness(components: string, options: Record<string, unknown> = {}, subdirectory = '') {
  const directory = mkdtempSync(path.join(tmpdir(), 'nemo-relay-opencode-'));
  writeFileSync(
    path.join(directory, 'plugins.toml'),
    `version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = ${JSON.stringify(directory)}
filename = "events.jsonl"
mode = "overwrite"
${components}`,
  );
  const hooks = new Map<string, Hook>();
  const register = (name: string, callback: Hook) =>
    Effect.sync(() => {
      hooks.set(name, callback);
      return { dispose: Effect.void };
    });
  const published: SessionEvent[] = [];
  let wake = () => {};
  async function* sessionEvents(): AsyncGenerator<SessionEvent> {
    for (;;) {
      while (published.length > 0) yield published.shift()!;
      await new Promise<void>((resolve) => (wake = resolve));
    }
  }
  const context = {
    app: { name: 'opencode', version: '2.2.5-test', channel: 'test' },
    options: { pluginsToml: 'plugins.toml', ...options },
    location: { directory: path.join(directory, subdirectory), project: { directory } },
    session: { hook: register },
    tool: { hook: register },
    event: { subscribe: () => Stream.fromAsyncIterable(sessionEvents(), (error) => error) },
  } as unknown as Context;
  return {
    directory,
    /** Whether the plugin registered its hooks, which it does only when Relay is active. */
    active: () => hooks.size > 0,
    hook: (name: string, event: unknown) => hooks.get(name)!(event),
    publish: (event: SessionEvent) => {
      published.push(event);
      wake();
    },
    /** Run `body` with the plugin active, then shut it down so Relay flushes its sinks. */
    run: (body: () => Generator<Effect.Effect<unknown, unknown>, void, unknown>) =>
      Effect.runPromise(
        Effect.scoped(
          Effect.gen(function* () {
            yield* nemoRelay(context);
            yield* Effect.gen(body);
          }),
        ),
      ),
    events: (): AtofEvent[] =>
      readFileSync(path.join(directory, 'events.jsonl'), 'utf8')
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as AtofEvent),
  };
}
