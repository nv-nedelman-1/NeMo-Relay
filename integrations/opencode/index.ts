// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * OpenCode plugin entry point.
 *
 * OpenCode loads this package from the directory listed in `opencode.json` `plugins`. This file
 * stays small; the wiring lives in `src/plugin.ts`.
 */
import type { Plugin } from './src/opencode.js';
import { nemoRelay } from './src/plugin.js';

const plugin: Plugin = { id: 'nemo-relay', effect: nemoRelay };

export default plugin;
