// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Test `relayPlugins` module: an LLM request intercept that masks configured text before a request is sent.
 */
import type { Plugin } from 'nemo-relay-node/plugin';

const mask: Plugin & { kind: string } = {
  kind: 'test.mask',
  register(config, context) {
    const secret = String(config.secret);
    context.registerLlmRequestIntercept('mask', 10, false, ({ request }) => ({
      request: JSON.parse(JSON.stringify(request).replaceAll(secret, '[MASKED]')),
    }));
  },
};

export default mask;
