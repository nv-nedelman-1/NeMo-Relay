// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Test `relayPlugins` module: guardrails and request intercepts that throw on requests naming them.
 */
import type { Plugin } from 'nemo-relay-node/plugin';

const breakOn = (word: string, value: unknown) => {
  if (JSON.stringify(value).includes(word)) throw new Error(`${word} broke`);
};

const broken: Plugin & { kind: string } = {
  kind: 'test.broken',
  register(_config, context) {
    context.registerLlmConditionalExecutionGuardrail('broken', 10, (request) => {
      breakOn('LLM-GUARDRAIL', request);
      return null;
    });
    context.registerLlmRequestIntercept('broken', 10, false, ({ request }) => {
      breakOn('LLM-INTERCEPT', request);
      return { request };
    });
    context.registerToolConditionalExecutionGuardrail('broken', 10, (_tool, args) => {
      breakOn('TOOL-GUARDRAIL', args);
      return null;
    });
    context.registerToolRequestIntercept('broken', 10, false, (_tool, args) => {
      breakOn('TOOL-INTERCEPT', args);
      return args;
    });
  },
};

export default broken;
