// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Relay plugin-host lifetime for one OpenCode process.
 *
 * Language-binding plugins listed in `relayPlugins` are registered before activation, so a
 * `plugins.toml` component can enable and configure them like any built-in component.
 *
 * Relay's plugin host is process-wide, but OpenCode runs a plugin instance per location (project
 * directory or worktree) and loads each with fresh module state, so the activation lives on the
 * process global: locations with the same options share it, and the last to close tears it down.
 */
import { pathToFileURL } from 'node:url';

import type { ConfigDiagnostic, Plugin } from 'nemo-relay-node/plugin';

import type { NemoRelayOpenCodeConfig } from './config.js';
import type { RelayPluginHost } from './modules.js';

/** Default export of a `relayPlugins` module: a Relay language-binding plugin and its kind. */
export type RelayPluginModule = Plugin & { kind: string };

export type RelayActivation = {
  /** Configuration warnings and errors Relay reported while activating. */
  readonly diagnostics: readonly ConfigDiagnostic[];
  /** Release this location's hold; the last release tears down Relay plugins and flushes exporters. */
  close(): Promise<void>;
};

const SHARED = Symbol.for('nemo-relay.opencode.activation.v1');

type Shared = {
  queue: Promise<unknown>;
  active?: { key: string; holders: number; activation: RelayActivation };
};

/** Hold the process's Relay activation, starting it if no location holds it yet. */
export function acquireRelay(config: NemoRelayOpenCodeConfig, host: RelayPluginHost): Promise<RelayActivation> {
  const key = JSON.stringify([config.pluginsToml, config.plugins, config.relayPlugins]);
  return serialized(async (shared) => {
    if (shared.active && shared.active.key !== key)
      throw new Error('another OpenCode location in this process runs NeMo Relay with different options');
    shared.active ??= { key, holders: 0, activation: await startRelay(config, host) };
    const active = shared.active;
    active.holders += 1;
    let released = false;
    return {
      diagnostics: active.activation.diagnostics,
      close: () => {
        if (released) return Promise.resolve();
        released = true;
        return serialized(async (current) => {
          if (--active.holders > 0) return;
          if (current.active === active) delete current.active;
          await active.activation.close();
        });
      },
    };
  });
}

/** Run `step` after every earlier start and release in this process, so activations never overlap. */
function serialized<T>(step: (shared: Shared) => Promise<T>): Promise<T> {
  const registry = globalThis as unknown as Record<symbol, Shared | undefined>;
  const shared = (registry[SHARED] ??= { queue: Promise.resolve() });
  const result = shared.queue.then(() => step(shared));
  shared.queue = result.catch(() => undefined);
  return result;
}

/** Register `relayPlugins` and activate the Relay plugin host. */
async function startRelay(config: NemoRelayOpenCodeConfig, host: RelayPluginHost): Promise<RelayActivation> {
  const kinds: string[] = [];
  const deregister = () => kinds.splice(0).forEach((kind) => host.deregister(kind));
  try {
    for (const file of config.relayPlugins) {
      const plugin = ((await import(pathToFileURL(file).href)) as { default?: unknown }).default;
      if (!isRelayPluginModule(plugin))
        throw new Error(`${file} must default-export a Relay plugin with a string kind and a register function`);
      host.register(plugin.kind, plugin);
      kinds.push(plugin.kind);
    }
    const activation = await host.initialize(config.plugins, config.pluginsToml);
    return {
      diagnostics: activation.report.config.diagnostics,
      close: async () => {
        await activation.close();
        deregister();
      },
    };
  } catch (error) {
    deregister();
    throw error;
  }
}

function isRelayPluginModule(value: unknown): value is RelayPluginModule {
  if (typeof value !== 'object' || value === null) return false;
  const { kind, register } = value as Record<string, unknown>;
  return typeof kind === 'string' && typeof register === 'function';
}
