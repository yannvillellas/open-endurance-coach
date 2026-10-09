# Safety model: writes, approvals and the prompt boundary

How the app keeps untrusted content from breaking out of its data, corrupting the
terminal, or being written to Intervals.icu differently from what the operator approved.
Implements #61; every protection below is covered by tests.

## Threat model

Untrusted text enters from four places: Intervals.icu data (activity names, interval
labels, event and race descriptions), the athlete's current message, the stored
conversation, and the model's own output. Hub content is not necessarily authored by this
user: devices, imported plans or a shared coach account can plant text, so it is treated
as externally influenced. The literal `yes` gate bounds the blast radius of a successful
persuasion: the model can propose, never write.

## Prompt structure

- Athlete data is wrapped in `<athlete_data>…</athlete_data>` and the current message in
  `<athlete_message>…</athlete_message>`; every `<`/`>` in their content is escaped to
  `\u003c`/`\u003e`, so no value can close a block.
- The conversation is sent as **native role messages** (`system`, then `user`/`assistant`
  turns, then the final `user` request), not as text with `user:`/`assistant:` prefixes.
  A turn's content cannot impersonate another role: the role is a protocol field.
- The conversation starts with a `user` turn (leading `assistant` turns are dropped) and
  never contains a `system` turn, so any chat API can map it.

## Characters

`sanitize_text` replaces dangerous characters with U+FFFD and normalises CRLF/CR to LF.
It keeps printable text, newlines and tabs, and keeps weak direction marks (LRM/RLM/ALM,
legitimate typography for non-Latin text). Replaced: C0/C1 controls, unpaired surrogates,
strong bidi overrides/isolates (spoofing) and line/paragraph separators (line forging).

Applied at every consumption boundary: prompt payload, focus, profile, tone and history
turns; terminal rendering (`escape()`, which also escapes rich markup); assistant content
before persistence; mutation `description`/`type` at the schema.

Mutation **names** are trimmed of surrounding whitespace and rejected — never replaced —
when they are then blank or still contain a newline, tab or control character: a name is
an identity (the writer matches on `(name, date)`), and a replaced character would
silently change it.

## Writes, approvals and display

- Writes are validated at the schema (`extra="forbid"`), so what the plan displays is
  exactly what is written.
- The proposal gate binds a `yes` to the displayed plan: `approve(expect=…)` compares a
  SHA-256 fingerprint of the mutations and refuses to apply a plan changed by another
  session, re-displaying it instead. Subsets (`yes 1 3`) index the approved snapshot, and
  `apply()` refuses any proposal that was never approved.
- The writer is idempotent by `(name, date)`: identical content is `unchanged`, but a
  create that collides with a different existing event is **refused** rather than
  overwriting it. Category-less events are adopted, never duplicated.
- Untrusted values (names, ids, hub/provider error text) are logged with `%r` and error
  details are collapsed to a single line, so a value cannot forge a log line. Provider and
  hub `Retry-After` hints are capped, so a server cannot stall the process.

## Accepted residuals

- **Semantic persuasion.** Delimiters and roles bind structure, not meaning: the model
  still reads injected sentences inside a block or a past `user` turn. The gate and the
  schema remain the backstop.
- **Terminal emulators.** OSC and other exotic sequences are neutralised by removing
  control characters, but behaviour differs across emulators and cannot be exhaustively
  verified.
- **Gate phrase in the panel.** The literal confirmation phrase can appear as plain text
  inside the plan panel; it cannot execute anything, and the real prompt sits outside it.
- **CR display.** A CR in a hub description is displayed as a line break (normalised) but
  stays escaped in the JSON payload.

## Provider adapter contract

The core only produces `list[LlmMessage]`; a new provider implements the `LlmProvider`
protocol and registers in `build_registry`, with no change above the client layer.
OpenAI-compatible providers (DeepSeek, OVH/Qwen) send the messages as-is; consecutive
`user` turns are already exercised by the chat-only fallback. An adapter that flattens
messages into plain text instead of mapping roles must tag-escape history content like
the data blocks, since a turn then becomes text again.

Mapping for a non-OpenAI API (e.g. Anthropic):

| Ours                                                        | Anthropic                                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `system` message                                            | top-level `system` parameter (no `system` role in `messages`)                                      |
| `user` / `assistant` turns                                  | unchanged; consecutive turns are combined by the API                                               |
| `json_mode` (`response_format`)                             | prompt-based JSON — the contract and example live in the system prompt — or `output_config.format` |
| `thinking`                                                  | `{"type": "enabled", "budget_tokens": …}`                                                          |
| `finish_reason`                                             | `stop_reason`                                                                                      |
| `prompt_tokens` / `completion_tokens` (token-drift warning) | `input_tokens` / `output_tokens`                                                                   |
