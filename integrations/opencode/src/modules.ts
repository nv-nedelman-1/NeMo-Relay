// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dynamic module loading boundary for the NeMo Relay Node binding.
 *
 * Keeping imports behind this loader lets OpenCode load the plugin even when the native binding is
 * unavailable; the plugin then stays inactive instead of failing OpenCode's startup.
 */
import type * as NemoRelayRuntime from 'nemo-relay-node';
import type * as NemoRelayPluginHost from 'nemo-relay-node/plugin';
import type { JsonValue, LlmCodec, LlmRequestShape } from 'nemo-relay-node/typed';

type RuntimeKeys =
  | 'ScopeType'
  | 'createScopeStack'
  | 'withScopeStack'
  | 'pushScope'
  | 'popScope'
  | 'event'
  | 'metric'
  | 'MetricKind'
  | 'MetricValueType'
  | 'llmCall'
  | 'llmCallEnd'
  | 'llmConditionalExecution'
  | 'llmRequestIntercepts'
  | 'toolCall'
  | 'toolCallEnd'
  | 'toolConditionalExecution'
  | 'toolRequestIntercepts'
  | 'OpenAIChatCodec'
  | 'OpenAIResponsesCodec';

/** @internal Subset of `nemo-relay-node` used by this integration. */
export type RelayRuntime = Pick<typeof NemoRelayRuntime, RuntimeKeys>;

/** @internal Subset of `nemo-relay-node/plugin` used by this integration. */
export type RelayPluginHost = Pick<typeof NemoRelayPluginHost, 'initialize' | 'register' | 'deregister'>;

export type RelayModules = { relay: RelayRuntime; host: RelayPluginHost };

export type Json = JsonValue;
export type { LlmCodec, LlmRequestShape };
export type ScopeHandle = NemoRelayRuntime.ScopeHandle;

/** Load the runtime and plugin-host modules. */
export async function loadRelay(): Promise<RelayModules> {
  const [relay, host] = await Promise.all([import('nemo-relay-node'), import('nemo-relay-node/plugin')]);
  return { relay: relay as RelayRuntime, host: host as RelayPluginHost };
}
