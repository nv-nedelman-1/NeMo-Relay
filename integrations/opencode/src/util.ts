// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isDeepStrictEqual } from 'node:util';

import { Effect } from 'effect';

import type { Json } from './modules.js';

/** `text` parsed as JSON, or `undefined` when it is not JSON. */
export function parseJson(text: string): Json | undefined {
  try {
    return JSON.parse(text) as Json;
  } catch {
    return undefined;
  }
}

/** A JSON-safe copy of `value`, dropping what JSON cannot carry. */
export function toJson(value: unknown): Json {
  try {
    return (JSON.parse(JSON.stringify(value ?? null)) as Json) ?? null;
  } catch {
    return String(value);
  }
}

/** Whether two JSON values are equal. Relay returns objects with their keys sorted, so key order is ignored. */
export function sameJson(left: unknown, right: unknown): boolean {
  return isDeepStrictEqual(left, right);
}

export function isRecord(value: unknown): value is Record<string, Json> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Run a Relay call from a hook; a failure is logged and the hook carries on without it. */
export function guarded<T>(label: string, call: () => T): Effect.Effect<T | undefined> {
  return Effect.suspend(() => {
    try {
      return Effect.succeed(call());
    } catch (error) {
      return Effect.as(Effect.logWarning(`NeMo Relay ${label} failed: ${message(error)}`), undefined);
    }
  });
}

/** Run `promise` without failing: the effect succeeds with its value or its rejection. */
export function settle<T>(promise: () => Promise<T>): Effect.Effect<{ value: T } | { error: unknown }> {
  return Effect.promise(() =>
    Promise.resolve()
      .then(promise)
      .then(
        (value) => ({ value }),
        (error: unknown) => ({ error }),
      ),
  );
}

/** Why a call is blocked when a Relay guardrail or intercept throws, as Relay's managed calls are; logged too. */
export function failed(callback: string, error: unknown): Effect.Effect<string> {
  const reason = `${callback} failed: ${message(error)}`;
  return Effect.as(Effect.logWarning(`NeMo Relay blocked a call: ${reason}`), reason);
}

/**
 * End metadata for a call or turn Relay recorded but did not run, which Relay reports as OK unless
 * told otherwise: a failure, with its type and description, or a neutral end, such as a cancellation.
 */
export function otelStatus(failure: { type: string; description?: string | undefined } | 'neutral' | undefined) {
  if (failure === undefined) return {};
  if (failure === 'neutral') return { 'otel.status_code': 'UNSET' };
  return {
    'otel.status_code': 'ERROR',
    'error.type': failure.type,
    ...(failure.description ? { 'otel.status_description': failure.description } : {}),
  };
}

/** The reason a Relay conditional-execution guardrail gave, or `undefined` for any other failure. */
export function rejectionReason(error: unknown): string | undefined {
  return /^guardrail rejected:\s*(.*)$/s.exec(message(error))?.[1];
}
