// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Options from the plugin's `opencode.json` entry.
 *
 * Relative paths resolve against the OpenCode project directory, so a project can keep its Relay
 * configuration next to its OpenCode configuration.
 */
import path from 'node:path';

import type { PluginConfig } from 'nemo-relay-node/plugin';

export type NemoRelayOpenCodeConfig = {
  /** Turns the integration off without removing it from `opencode.json`. */
  enabled: boolean;
  /** Explicit `plugins.toml` layer, applied over the user and system files Relay discovers. */
  pluginsToml: string | undefined;
  /** Programmatic plugin-host configuration; it overrides every file layer. */
  plugins: PluginConfig;
  /** Modules whose default export is a Relay language-binding plugin with its `kind`. */
  relayPlugins: string[];
};

const KEYS = new Set(['enabled', 'failClosed', 'pluginsToml', 'plugins', 'relayPlugins']);

/** Validate plugin options and resolve their paths against `directory`. */
export function parseConfig(options: Record<string, unknown>, directory: string): NemoRelayOpenCodeConfig {
  const unknown = Object.keys(options).filter((key) => !KEYS.has(key));
  if (unknown.length > 0) throw new Error(`unknown option(s): ${unknown.join(', ')}`);
  const {
    enabled = true,
    failClosed = false,
    pluginsToml,
    plugins = { version: 1, components: [] },
    relayPlugins = [],
  } = options;
  if (typeof enabled !== 'boolean') throw new Error('enabled must be a boolean');
  // The plugin reads `failClosed` before parsing, so invalid options can still fail closed; it is only checked here.
  if (typeof failClosed !== 'boolean') throw new Error('failClosed must be a boolean');
  if (pluginsToml !== undefined && typeof pluginsToml !== 'string') throw new Error('pluginsToml must be a path');
  if (!isPluginConfig(plugins)) throw new Error('plugins must be a { version, components } document');
  if (!Array.isArray(relayPlugins) || !relayPlugins.every((entry) => typeof entry === 'string'))
    throw new Error('relayPlugins must be an array of module paths');
  return {
    enabled,
    pluginsToml: pluginsToml === undefined ? undefined : path.resolve(directory, pluginsToml),
    plugins,
    relayPlugins: relayPlugins.map((entry) => path.resolve(directory, entry)),
  };
}

function isPluginConfig(value: unknown): value is PluginConfig {
  if (typeof value !== 'object' || value === null) return false;
  const { version, components } = value as Record<string, unknown>;
  return (
    typeof version === 'number' &&
    Array.isArray(components) &&
    components.every((component) => typeof (component as { kind?: unknown } | null)?.kind === 'string')
  );
}
