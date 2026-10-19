# Architecture

What the coach reads, how it reasons, and how a change reaches Intervals.icu. The
safety rules are in [safety-model.md](safety-model.md); the chat behaviour is in
[chat.md](chat.md).

## Data extraction

One gather (`extractors/context.py`) driven by a declared context plan, budgeted to fit
the model's token limit (`extractors/budget.py`).

Before each new question the analysis window is resolved with a small model call that
sees the recent conversation and the active athlete facts, ranked and capped at the same
25 as the memory injection, and is itself fitted to the input budget
(`extractors/resolution.py`): a strict JSON reply gives the exact past window or the
cited past days (or none for tactical questions), the sport, the metric, whether one
activity must be detailed,
whether the question reviews past execution against the plan (`review_execution` —
then a review without its own period reads the whole planning horizon), and the declared
athlete state (illness, elite). The code validates the dates, caps the
windows (activities 3650 days, wellness and events 365 days) and announces what was read
with one `Fetch:` line (`state:` included when declared); a resolution failure falls back
to the standard windows with a warning. The day/period caps apply to a resolved period;
cited-day windows are bounded by count (`MAX_ACTIVITY_WINDOWS`) instead, and a very
tight budget still trims data oldest-first. A refinement sent while a proposal is open
rebuilds that proposal's snapshot instead, with no resolution call.

- **Tactical plan** — recent activities (14 days), wellness (7 days), a ±14-day
  calendar window split into recent and upcoming events, sport settings, goal races and
  a weekly training rollup with CTL/ATL/ramp and per-sport load, both over a 224-day
  horizon (`PLAN_HORIZON_DAYS`).
- **Question window** — activities and wellness cover exactly the resolved window(s);
  when the question cites several scattered past days, each day gets its own ±3-day
  window (touching windows merge, at most ten keep the most recent), so the fetch stays
  exact instead of spanning the gap. Events add the tactical ±14-day window when the
  two overlap, so a mixed question
  ("that session three weeks ago, should I repeat it next week?") sees both, and goal
  races and the rollup stay tactical. A named sport ranks matching activities first and
  guides the detail pin without dropping any activity; a detailed activity carries its
  streams and per-kilometre splits, pinned ±3 days so the evidence survives context
  trimming.

A pure periodization skeleton (`extractors/periodization.py`, [periodization.md](periodization.md))
derives the race block from the anchor race, the week grid and the executed rollup: one
`block_plan` row per week (phase, role, target TSS band, status), injected right after
`goal_races` and rendered in its own `<block_plan>` prompt block as a compact Markdown
table. Same inputs, same weeks: the block is race-anchored and recomputed each turn, never
persisted; only `today` positions the current week. Contingency adjustments (rest, repeat,
step_back, omitted) travel as row statuses, roles and adjusted bands.

On a review turn (`review_execution` from the resolver), the same gather computes an
execution review over the block — matching, weekly aggregation and the efficacy
comparison against the adjusted block rows (`analysis/execution.py`,
[execution-review.md](execution-review.md)); the digest is injected as the compact
`execution_review` section and rendered in its own `<execution_review>` prompt block.

When the budget is exceeded, the extractor evicts data by priority until the context
fits, and never touches the current message. Goal-race descriptions are trimmed before
a whole race is dropped, the block plan is dropped only after the goal races, the
execution review only after the block plan, and activities/events are evicted before
the sports settings.

## Analysis

The model answers through a strict JSON schema (invalid output is retried, then
rejected; `schemas/decisions.py`). The prompt enforces Joe Friel's periodization
principles and Dr. Andrew Coggan's power analytics: executed training against planned
targets and the `block_plan` rows, and current readiness (CTL/ATL, HRV, sleep). The
answer is one of _chat_, _analysis_ or _plan_, and anything material the coach is missing
is asked for before a proposal is finalized.

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
via `CoachStore.schema_version`. Migration 2 adds the indexes for the filtered reads
(`proposals.status`, `proposals.created_at`, `messages.created_at`,
`seen_activities.seen_at`).

Retention follows purpose, not a single age knob. Messages are pruned at startup to
`CHAT_HISTORY_MAX_AGE_DAYS` — the maximum age at which a stored turn can still be
seeded into a session — so one setting defines both what is read and what is kept.
Proposals are lifecycle-bound and never pruned by age: apply and reject delete the row,
orphaned `pending` drafts are discarded at startup, an invalid `unapplied` is discarded
on drift or staleness, and a valid one is deliberately re-offered until resolved.
`seen_activities` markers are pruned to the widest fetch window
(`MAX_ACTIVITY_WINDOW_DAYS`): a marker is deleted only once its activity can no longer be
returned by any `list_activities` call, so it never causes a false "New activities
since last review" and is never kept once it cannot be read again. The tests pin the
boundaries: old messages and unfetchable markers are pruned while proposals and
still-fetchable markers survive (`tests/test_store.py`).

### Coach memory

`coach_memory` holds durable athlete facts across two provenance classes (declared,
observed) and seven categories. The model writes them on every turn through the
report's `memory` operations — `remember`, `update` (only the changed fields; an
explicit null clears one), `close` and `forget` — outside the calendar approval gate.
Only durable facts are accepted: one-off requests, single-session decisions, questions,
hypotheticals and lone examples stay in the calendar or the reply, a
`preferences_tolerance` fact needs an explicit preference, and a `forget` request
targets every active fact matching the description (asking when ambiguous). Closure
follows the category in code: `injury_niggle` and `health_medical` are kept as
`resolved` history (with `valid_to`), everything else is deleted, and a passed
`expires_on` is swept at startup. Memory is never pruned by age; `/forget all` erases
it. Injection happens in the engine after extraction: active facts are ranked by
trigger (focus keywords or reached date), then category priority (health, injury,
constraints, profile, goals, preferences, observed), then least recently surfaced;
full facts fit a ~350-token allowance (25 max), the rest appear as a compact
id/category/label index (~150 tokens), and injected facts get `last_surfaced_at`.

## Calendar writer

Approved proposals are applied by `writer/calendar.py`:

- creates and updates are idempotent by name+date: identical content is `unchanged`;
- workout mutations only touch `WORKOUT` events, race mutations only `RACE_*`; an
  unlabelled event matched by name+date is adopted by the mutation's family;
- a create that would overwrite a different existing event is **refused**;
- writes are **serialized per writer instance** (the CLI creates one per process), so
  two concurrent approvals on the same writer cannot interleave their find and create;
  when a create fails ambiguously
  (lost response or 5xx) the writer **re-lists** by name+date and reconciles with the
  event that did land instead of duplicating it — cross-process writes are the store
  concurrency model (#69);
- updates and deletes never cross families, and every write is read back and compared,
  with any drift reported to the operator.

## Interfaces

The CLI (`coach`) drives the loop today. Webhook triggers (activity uploaded/analyzed,
calendar updated) and a wellness poller are planned behind the same engine; an
Intervals.icu OAuth app has been created for that step. The LLM client keeps a
per-provider circuit breaker: after N consecutive failures it fails fast, and a
half-open probe closes it again once the provider recovers.

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
