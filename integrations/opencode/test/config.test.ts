// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin option parsing.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';

const directory = path.resolve('/projects/demo');

describe('parseConfig', () => {
  it('defaults to an enabled plugin with no programmatic components', () => {
    assert.deepEqual(parseConfig({}, directory), {
      enabled: true,
      pluginsToml: undefined,
      plugins: { version: 1, components: [] },
      relayPlugins: [],
    });
  });

  it('resolves paths against the project directory', () => {
    const config = parseConfig(
      { pluginsToml: 'relay/plugins.toml', relayPlugins: ['./relay/policy.mjs', '/opt/relay/audit.mjs'] },
      directory,
    );

    assert.equal(config.pluginsToml, path.join(directory, 'relay/plugins.toml'));
    assert.deepEqual(config.relayPlugins, [
      path.join(directory, 'relay/policy.mjs'),
      path.resolve('/opt/relay/audit.mjs'),
    ]);
  });

  it('rejects unknown options and wrong types', () => {
    assert.throws(() => parseConfig({ pluginToml: 'x' }, directory), /unknown option\(s\): pluginToml/);
    assert.throws(() => parseConfig({ enabled: 'yes' }, directory), /enabled must be a boolean/);
    assert.throws(() => parseConfig({ failClosed: 'yes' }, directory), /failClosed must be a boolean/);
    assert.throws(() => parseConfig({ plugins: { version: 1, components: [{}] } }, directory), /plugins must be/);
    assert.throws(() => parseConfig({ relayPlugins: 'policy.mjs' }, directory), /relayPlugins must be/);
  });
});
