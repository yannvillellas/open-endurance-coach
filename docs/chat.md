# Chat

`coach` opens a single conversation with the coach. Everything happens in the
conversation; the only commands are session commands.

## Message handling

Every message is classified from the data snapshot as _chat_, _analysis_ or _plan_:

- a fresh snapshot is extracted when needed (first message, requests like
  "analyze/review/check my week", or trend questions), otherwise the cached snapshot is
  reused;
- a full analysis is reused for follow-ups — there is no re-analysis until you ask for
  one or the question needs historical depth;
- if a change would help during a chat, the coach offers it instead of interrupting you
  with a confirmation gate.

## Questions and assumptions

Material questions block proposals, for any plan — a training block, a race event, or
both. If an answer would change the plan (available training days, constraints, injury,
RPE, race duration/climbing/expected load), the coach asks and does _not_ propose
calendar changes until you answer — or you say `proceed with assumptions` and the
assumption is stated in the plan.

## Approval gate

When the coach proposes calendar changes it asks: "Apply this to Intervals.icu: …".
Reply with exactly `yes` and the changes are validated, approved and written in one
step.

- Items are numbered: `yes 1 3` approves only those items, `yes except 2` all the
  others. An item can also be picked by weekday (`yes except thursday`) or ISO date.
- A selection that matches no proposed item writes nothing and keeps the proposal open.
- Unapproved items are skipped and never written.
- `no` declines. Anything else is a change request: the coach re-analyzes with your
  words and proposes again.
- A proposal changed by another session between display and `yes` is refused and the
  updated plan is re-displayed.
- If an approved proposal fails to write, say `retry`. An unapplied proposal is offered
  again at the next startup, or discarded with a notice once its dates have passed.
- Ctrl-C during confirmation cancels safely.

## Memory

Sessions are seeded with recent exchanges (last 10 messages from the last 90 days;
`CHAT_HISTORY_TURNS`, `CHAT_HISTORY_MAX_AGE_DAYS`). The in-session cap follows the
active model: the derived input budget minus the system prompt, the reserved
athlete-data budget and prompt overhead. The coach warns at 50% and 75% of that cap
and, at 80%, drops the oldest exchanges (keeping the newest) and reports the new
context size. With the default model-window budgets those thresholds sit far above a
normal session, so they matter mainly when `LLM_INPUT_BUDGET` is set lower; the cap
assumes the full athlete-data reserve and no current message, so a very large message
can trim slightly before the notice. Stored history is pruned automatically to
`HISTORY_DAYS` (default 180) at startup.

`/forget` deletes the conversation (the `messages` log) now, `/forget N` deletes only
the messages older than N days, and `/forget all` deletes all local state
(conversation, proposals, dedup markers) after a literal `yes`; the plan already
applied and your races stay in Intervals.icu.

## Commands

`/provider [name]` and `/model [name]` show or switch the LLM, `/forget [days]` or
`/forget all`, `/help`, `/exit`. Everything else is conversation.
