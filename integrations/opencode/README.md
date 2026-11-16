<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay for OpenCode

`nemo-relay-opencode` is an OpenCode plugin that runs NeMo Relay inside OpenCode. It records each
session turn as Relay agent, LLM, and tool calls, and applies Relay's request intercepts and
conditional-execution guardrails to OpenCode's model requests and tool calls. It uses OpenCode's
existing plugin hooks (OpenCode v2, `@opencode/plugin` 2.2 or later), so OpenCode needs no changes.

## What It Records

- **One agent scope per turn.** Each OpenCode session execution becomes a Relay agent scope and its own
  trace, rooted at the agent span, separate from OpenCode's own traces. A model or tool call still
  open when the turn ends, such as a request whose connection failed, ends with it as cancelled.
  Title generation, which OpenCode does not make the turn wait for, is the exception: its call ends
  with its own response. Requests OpenCode makes outside a turn, such as cache-warming `generate`
  requests while a session is idle, are LLM calls of their own, with no agent scope.
- **One LLM call per provider request.** Every request OpenCode sends to a model provider, including
  title generation and compaction, with the request body and the provider's response. Over HTTP, a
  streamed response reaches OpenCode unchanged while the plugin assembles the final response for
  Relay. Over OpenCode's WebSocket transport (the default for its OpenAI, xAI, and Azure providers),
  each `response.create` frame is one call that ends at the response's terminal event; when OpenCode
  continues a response with only the new input and `previous_response_id`, the plugin rebuilds the
  full input, so Relay receives the complete request either way. Provider credentials stay in
  OpenCode: Relay receives request bodies, not headers.
- **One tool call per executed tool,** with its arguments and result, keyed by OpenCode's tool-call ID.
- **The OpenCode session ID on every event,** under Relay's session-identity metadata keys: `session_id`,
  with `agent_kind` set to `opencode`. Relay exports the ID as `gen_ai.conversation.id` on `gen_ai`
  spans, and as `session.id` on the trace roots of its `full` and `openinference` projections. OpenCode
  releases after 2.2.5 put the same ID on their own spans as `session.id`, so Relay's traces and
  OpenCode's can be matched by session in separate backends. Over HTTP, Relay also sends the ID to the
  provider as `x-dynamo-session-id`, its agent-lineage header for NVIDIA Dynamo, on the calls made
  within a turn.
- **GenAI client metrics** for each model call: `gen_ai.client.token.usage`,
  `gen_ai.client.operation.duration`, and, for streams, `gen_ai.client.operation.time_to_first_chunk`.

Exporters, trajectories (ATIF), the event log (ATOF), redaction, and pricing come from Relay's plugin
configuration, as for any Relay host.

## Policy

- **Tools.** Tool conditional-execution guardrails run before OpenCode executes a tool. A rejection
  fails that call alone; the model receives `Blocked by NeMo Relay policy: <reason>` as the tool result.
  Tool request intercepts can rewrite the arguments.
- **Model requests.** LLM conditional-execution guardrails see each request as OpenCode built it and
  can block it; a blocked request is never sent. LLM request intercepts then run and can rewrite its
  body, or its outbound WebSocket frame, for example to mask secrets. How OpenCode reports a block
  depends on the transport:
  - Over HTTP, where OpenCode's `http.request` hook accepts a `response`, the plugin answers with an
    HTTP 403 policy error that OpenCode reports as a provider error. Older OpenCode versions offer no
    way to answer a request, so the plugin fails it instead, which OpenCode reports as a session error.
  - Over WebSocket, OpenCode's hooks cannot answer a request either, so OpenCode reports a generic
    stream failure. Set `"transport": "http"` on providers whose blocked requests should end with the
    policy reason.
- A guardrail or request intercept that throws blocks its call, as in Relay's managed calls, with a
  reason that names the failure. A masking intercept that fails therefore never lets its request out.
- Each block is recorded as an `opencode.tool_blocked` or `opencode.llm_blocked` mark with the reason.

## Use

Build the Node binding and the plugin from the repository root:

```bash
npm ci --ignore-scripts
npm run build --workspace=nemo-relay-node
npm run build --workspace=nemo-relay-opencode
```

Then list the package directory in `opencode.json`:

```json
{
  "plugins": [
    {
      "package": "/path/to/NeMo-Relay/integrations/opencode",
      "options": { "pluginsToml": "relay/plugins.toml" }
    }
  ]
}
```

A `plugins.toml` that writes trajectories and redacts emails and API keys from everything Relay exports:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atif]
enabled = true
agent_name = "opencode"
output_directory = "relay-output"
filename_template = "trajectory-{session_id}.json"

[[components]]
kind = "pii_redaction"
enabled = true

[[components.config.profiles]]
mode = "builtin"

[components.config.profiles.builtin]
action = "redact"
detector = "api_key"
```

### Options

| Option | Default | Description |
|---|---|---|
| `enabled` | `true` | Set `false` to turn the plugin off without removing it. |
| `failClosed` | `false` | Set `true` to block model requests and tool calls wherever Relay is not active, for example because the Node binding or a `relayPlugins` module failed to load, instead of running without Relay. |
| `pluginsToml` | none | An explicit `plugins.toml` layer, applied over the user and system files Relay discovers. |
| `plugins` | `{ "version": 1, "components": [] }` | Programmatic plugin configuration; it overrides every file layer. |
| `relayPlugins` | `[]` | Modules to register as Relay language-binding plugins before Relay starts. |

Relative paths resolve against the OpenCode project directory. Unknown options are rejected.

A `relayPlugins` module default-exports a Relay language-binding plugin with its `kind`. A component of
that kind in `plugins.toml` then enables and configures it like any built-in component:

```js
export default {
  kind: 'example.policy',
  register(config, context) {
    context.registerToolConditionalExecutionGuardrail('no-recursive-delete', 10, (_tool, args) =>
      /\brm\s+-[a-zA-Z]*r/.test(args?.command ?? '') ? 'recursive deletes are not allowed' : null,
    );
  },
};
```

### OpenTelemetry

OpenCode and Relay read the same `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_RESOURCE_ATTRIBUTES`. OpenCode
exports its own spans there, and Relay's automatic exporter sends its GenAI spans, logs, and metrics to
the same collector with the same resource attributes. Both describe the same model and tool calls, so
keep them in separate projects or backends; one that totals tokens across both counts each call twice.

To give Relay a destination of its own, configure an endpoint in Relay's `opentelemetry` section with
its own `service_name` or `resource_attributes`, and set `OTEL_TRACES_EXPORTER`, `OTEL_LOGS_EXPORTER`,
and `OTEL_METRICS_EXPORTER` to `none`, which turn off Relay's automatic exporter and which OpenCode
does not read. Relay does not start its own endpoints while `OTEL_EXPORTER_OTLP_HEADERS` is set; if
OpenCode needs that variable, send both to a local collector and separate them there. Refer to Relay's
OpenTelemetry configuration guide.

## Limits

- WebSocket support uses OpenCode's `experimental.ws.*` hooks, which OpenCode may still change.
  Request intercepts can rewrite a WebSocket frame's body but not its headers. To rebuild incremental
  requests, the plugin keeps the last exchange of the 32 most recently used sockets; a continuation
  on an older socket is recorded as sent.
- Relay decodes OpenAI Chat Completions and Responses, Anthropic Messages, and Gemini requests and
  responses. The plugin assembles a streamed response in these formats into the response Relay
  decodes; streams from other providers, and failed streams, are recorded as the list of their events.
  A binary stream, such as Bedrock's, reaches OpenCode unchanged but is recorded without its body.
- OpenCode signs Bedrock requests that use SigV4 authentication over their body before plugin hooks
  run, so a request intercept that changes such a body, for example to mask a secret, invalidates
  its signature. Relay's own headers do not.
- Relay's plugin host is process-wide, and one OpenCode process can serve several locations (project
  directories and worktrees). Locations whose options match, once relative paths resolve, share one
  Relay activation; a location whose options differ runs without Relay and logs why.
- If the Node binding is unavailable or Relay fails to start, the plugin logs why and stays inactive,
  and OpenCode runs without Relay, unless `failClosed` is set. Failing to record a call is logged and
  the call proceeds. `failClosed` guards against Relay failing, not against a user who controls their
  OpenCode configuration removing the plugin.

## Development

```bash
npm run typecheck --workspace=nemo-relay-opencode
npm test --workspace=nemo-relay-opencode
```
