# Architecture

What the coach reads, how it reasons, and how a change reaches Intervals.icu. The
safety rules are in [safety-model.md](safety-model.md); the chat behaviour is in
[chat.md](chat.md).

## Data extraction

Two scopes (`extractors/standard.py`, `extractors/deep.py`), both budgeted to fit the
model's token limit (`extractors/budget.py`):

- **Standard** — recent activities, a ±14-day calendar window split into recent and
  upcoming events, wellness, sport settings, goal races over a 120-day horizon with
  macro phase anchors, and a 90-day weekly training rollup with CTL/ATL/ramp and
  per-sport load.
- **Deep historical** — trend queries such as "heart rate improvement on hills over the
  last 3 months". It carries the same goal-race and rollup context so a trend answer can
  be tied back to the race being trained for, and pins the referenced window (±3 days)
  so the evidence survives context trimming.

When the budget is exceeded, the extractor evicts data by priority until the context
fits, and never touches the current message. Goal-race descriptions are trimmed before
a whole race is dropped, and activities/events are evicted before the sports settings.

## Analysis

The model answers through a strict JSON schema (invalid output is retried, then
rejected; `schemas/decisions.py`). The prompt enforces Joe Friel's periodization
principles and Dr. Andrew Coggan's power analytics: executed training against planned
targets, and current readiness (CTL/ATL, HRV, sleep). The answer is one of _chat_,
_analysis_ or _plan_, and anything material the coach is missing is asked for before a
proposal is finalized.

The decision loop is explicit and ordered: validate planned against executed, solicit
and inject the athlete's qualitative context (RPE, fatigue, schedule constraints)
before finalizing, modulate the upcoming load, then generate the structural changes.
User context is always injected before any calendar update. Missing race details
(distance, climbing, expected load) are asked for, never estimated.

## Proposals

An analysis that proposes calendar changes is stored as a pending proposal. The athlete
sees the exact plan (day-grouped, numbered) and approves with a literal `yes`,
optionally a subset. The approval is bound to a fingerprint of the displayed plan, so a
plan changed by another session is refused and re-displayed instead of applied. The
gate behaviour is documented in [chat.md](chat.md).

## Store & schema

State lives in one local SQLite file (`store/db.py`): `proposals`, `messages` and
`seen_activities`. The schema is versioned with `PRAGMA user_version`. At init
`CoachStore` applies the ordered, idempotent migrations from the file's recorded version
up to `SCHEMA_VERSION` (version 1 is the base schema); each step runs in a transaction
that commits its DDL and the new version together, so a failed migration rolls back
whole. A fresh database and an existing one converge on the same schema; a database
newer than the code is refused instead of being silently mis-read. A schema change is a
new numbered entry in `MIGRATIONS`, not an ad-hoc `ALTER TABLE`. The version is readable
via `CoachStore.schema_version`.

## Calendar writer

Approved proposals are applied by `writer/calendar.py`:

- creates and updates are idempotent by name+date: identical content is `unchanged`;
- workout mutations only touch `WORKOUT` events, race mutations only `RACE_*`; an
  unlabelled event matched by name+date is adopted by the mutation's family;
- a create that would overwrite a different existing event is **refused**;
- updates and deletes never cross families, and every write is read back and compared,
  with any drift reported to the operator.

## Interfaces

The CLI (`coach`) drives the loop today. Webhook triggers (activity uploaded/analyzed,
calendar updated) and a wellness poller are planned behind the same engine; an
Intervals.icu OAuth app has been created for that step.

## Source map

    src/open_endurance_coach/
      cli/          commands, chat loop, rendering, confirmation
      chat/         history, gate, session state
      engine/       orchestration: analyse, approve, apply, feedback
      extractors/   data scopes and budget eviction
      clients/      Intervals.icu and LLM providers (adapter contract in safety-model.md)
      prompts/      system prompt, JSON contract, rendering rules
      schemas/      pydantic models for hub data and LLM output
      store/        SQLite state: proposals, messages, seen activities
      writer/       calendar mutations
      sanitize.py   character and single-line sanitisation
      tokens.py     token estimation and budgets
