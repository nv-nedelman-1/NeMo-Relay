// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Test `relayPlugins` module: blocks shell commands and model requests that contain configured text.
 */
import type { Plugin } from 'nemo-relay-node/plugin';

const policy: Plugin & { kind: string } = {
  kind: 'test.policy',
  register(config, context) {
    const command = String(config.command);
    const secret = String(config.secret);
    context.registerToolConditionalExecutionGuardrail('commands', 10, (tool, args) =>
      tool === 'shell' && JSON.stringify(args).includes(command) ? `${command} is not allowed` : null,
    );
    context.registerLlmConditionalExecutionGuardrail('secrets', 10, (request) =>
      JSON.stringify(request).includes(secret) ? 'the request contains a secret' : null,
    );
  },
};

export default policy;
