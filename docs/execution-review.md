# Execution review

A review turn measures, over past weeks, whether the athlete did what the calendar
prescribed and whether the intended periodization was accompanied by the expected
movement. It is read-only: no calendar write, no private fetch path, no stored state.

## Trigger and window

The window resolver (#91) sets `review_execution` when the question compares past
execution to the plan, whatever the wording — never by keyword. Without an explicit
period the context plan reads the whole planning horizon (224 days) and the default
review clips at the block start; an explicit period is honoured as resolved. The review
stops at yesterday: a session dated today is judged once its activity is done, an unmatched
session dated today is pending — never missed — and an unmatched activity dated today is
not reported as unplanned.

## Matching

The plan is the whole calendar: every `WORKOUT` and `RACE_*` event, whoever wrote it.
The hub's pairing leads — an activity whose `paired_event_id` points at a planned event
is that session whatever the dates, and an activity paired outside the review is never
reused. The remaining sessions take the nearest unused unpaired activity within one day,
by type family (ride, Run, Swim, Walk/Hike; an unknown type matches only itself, and a
planned event without a type accepts any sport).

The hub's `Activity.compliance` scores its own pairing: it reads first, never
recomputed, when the activity is paired to the session. The load, duration and distance
ratios are the fallback for sessions the hub does not score (unpaired or fallback
matches; a `0.0` sentinel on an unpaired activity is not trusted). The
classes are `completed` (≥ 80 %), `partial` (20-80 %) and `missed` (< 20 % or no match);
an executed activity with no planned session is `unplanned`.

## Aggregation

Per week, on the block's week grid (the hub rollup cadence, Monday in practice):
planned and executed load, the completed/partial/missed counts, adherence
(`completed / judged`), unplanned load and its share of the executed load — all
rendered in the digest. A week with nothing due keeps its row — a prescribed stop is not a failure — and
the current week is marked as such. Per session the digest always carries the exact
planned/actual numbers (load, then duration or distance when missing) and the hub
compliance when present; no home-made percentage is ever computed.

## Efficacy

Each week joins its adjusted `block_plan` row (phase, status, band) and the hub's
rollup movement (CTL, ramp). The band and the status are read as-is: `rest`, `repeat`,
`step_back` and `omitted` weeks are the plan, not failures. The report says whether the
intended trajectory was accompanied by a comparable real movement — suggestive, never
causal, with no composite score.

## Injection

The digest is computed in code on the full fetched data, before budget eviction, so old
weeks never depend on what survives it; it is injected as the compact `execution_review`
section (a weekly table plus one line per session, rendered in its own
`<execution_review>` prompt block), evicted only after the block plan, and skipped when
no race anchors a block. The model
narrates it; it never computes it.
