# Open Endurance Coach

A self-hosted, AI-driven endurance coaching system.

Open Endurance Coach integrates multi-sport telemetry from Intervals.icu with Large Language Model analysis to validate training execution and adjust future workouts. Every proposed calendar change is reviewed and approved by the athlete before anything is written.

## Current Capabilities

- **Data Extraction:** Standard scope (recent activities, a ±14-day calendar window split into recent and upcoming events, wellness, sport settings, goal races over a 120-day horizon with macro phase anchors, and a 90-day weekly training rollup with CTL/ATL/ramp and per-sport load) and deep-historical scope (trend queries such as "heart rate improvement on hills over the last 3 months"), which carries the same goal-race and rollup context so a trend answer can be tied back to the race being trained for and pins the referenced window (±3 days) so evidence survives context trimming. Both are budgeted to fit the model's token limit.
- **Analysis:** OVHcloud AI Endpoints' free tier (Qwen3.5-397B-A17B, JSON mode, thinking enabled; no API key) — or DeepSeek — enforces Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics, comparing executed training against planned targets and current readiness (CTL/ATL, HRV, sleep).
- **Proposal & Review Loop:** Every analysis produces a validated proposal under a strict schema — invalid LLM output is retried, then rejected. The coach asks for anything material it is missing (availability, constraints, injury, RPE, race details) and re-analyzes with your answer before anything can be approved.
- **Race-Aware Planning:** With a goal race in the calendar the coach plans backwards from it — states the macro phases (Base, Build, Peak, Taper) with weekly load targets, then proposes concrete workouts for the next 7–14 days. Races are first-class events (`RACE_A/B/C`) it can create or adjust; missing race details (distance, climbing, expected load) are asked for, never estimated.
- **Calendar Writer:** Approved proposals are applied to Intervals.icu with idempotent create/update and strict category guards — workout mutations only touch `WORKOUT` events, race mutations only touch `RACE_A/B/C` events (an unlabelled event matched by name+date is adopted by the mutation's family), and updates/deletes never cross between the two families.
- **Manual-First Triggers:** The CLI drives the loop today. Webhook triggers (activity uploaded/analyzed, calendar updated) and a wellness poller are on the roadmap behind the same engine; an Intervals.icu OAuth app has been created for that step.

## Chat mode (the main interface)

`coach` opens a single conversation with the coach. You just talk:

- **Free text runs a full analysis.** Every message is classified from the data snapshot: a fresh snapshot is extracted when needed (first message, requests like "analyze/review/check my week", or trend questions), otherwise the cached snapshot is reused.
- **The coach decides what you need.** Every message is classified as _chat_ (answer from the analysis already in the session), _analysis_ (review executed training), or _plan_ (propose calendar changes). A full analysis is reused for follow-ups — there is no re-analysis until you ask for one or the question needs historical depth. If a change would help during a chat, he offers it instead of interrupting you with a confirmation gate.
- **Material questions block proposals, for any plan** — a training block, a race event, or both. If an answer would change the plan (available training days, constraints, injury, RPE, race duration/climbing/expected load), the coach asks and does _not_ propose calendar changes until you answer — or say `proceed with assumptions` and he states the assumption in the plan.
- **When he proposes calendar changes**, he asks: "Apply this to Intervals.icu: …". Reply with exactly `yes` and the changes are validated, approved, and written in one step. The proposal items are numbered, so `yes` can also approve a subset: `yes 1 3` approves only those items and `yes except 2` all the others; an item can also be picked by weekday (`yes except thursday`) or ISO date. Unapproved items are skipped and never written, and a selection that matches no proposed item writes nothing and keeps the proposal open. `no` declines, and **anything else is a change request** — he re-analyzes with your words and proposes again. Nothing is ever written without a literal yes. If a proposal is approved but the write fails, say `retry`; an unapplied proposal is offered again at the next startup, or discarded with a notice once its dates have passed.
- **Memory**: sessions are seeded with recent exchanges (last 10 messages from the last 90 days). The in-session cap follows the active model: it is the derived input budget minus the system prompt, the reserved athlete-data budget and prompt overhead, so old turns are trimmed only when the context is full. The coach warns at 50% and 75% of that cap and, at 80%, drops the oldest exchanges (keeping the newest) and reports the new context size; with the default model-window budgets those thresholds sit far above a normal session, so they matter mainly when `LLM_INPUT_BUDGET` is set lower; the cap assumes the full athlete-data reserve and no current message, so a very large message can trim slightly before the notice. Stored history is pruned automatically to `HISTORY_DAYS` (default 180) at startup; `/forget` wipes it now, or `/forget N` keeps only the last N days.
- Session commands only: `/provider` and `/model` show or switch the LLM, `/forget [days]`, `/help`, `/exit` — everything else is conversation.

```text
$ coach
Chat with the coach.

you: how was my week?
Coach:
  Race is 10 days out; the taper shape is correct.
Evidence:
  - Load: 236 last week, 77 so far this week.
Open questions:
  ? Do you want the rest week on the calendar?
Confirm? Reply exactly yes to apply, no to discard.
Or describe a change to revise the plan.
  (Proposal panel with the day-grouped create/update/delete list)
you: make it 45 minutes instead
Coach: (re-analyzes, shows the updated proposal)
you: yes
Applied:
  - create -> created event 130130106
```

## Approvals

Calendar changes never happen silently: when the coach proposes adjustments he asks for a literal `yes` before anything is written, and `no` declines the proposal. `yes 1 3` or `yes except 2` approves only part of it; the rest is never written. Everything happens inside the conversation.

## Modes

- **Analysis** (free text in chat when data is needed): extracts a fresh data snapshot (recent activities, wellness, upcoming events, sport settings — or a 90-day filtered window for trend questions), runs the strict JSON analysis, saves the report internally as a proposal, and marks the analyzed activities as seen so they surface as "New activities since last review" only once. Nothing is written to the calendar here.
- **Chat** (any other free text): answers from the cached snapshot through the same JSON analysis; it never writes to the calendar and only proposes changes it classified as _plan_.
- **Proposal gate**: every calendar change is proposed with the exact plan and requires a literal `yes`; any other answer is treated as a change request or discussion and writes nothing. Mid-confirmation Ctrl-C cancels safely.

## Configuration

All settings come from environment variables or a `.env` file (see `.env.example`). Essentials: `INTERVALS_API_KEY`, `INTERVALS_ATHLETE_ID`. No LLM API key is required by default: the coach uses OVHcloud AI Endpoints' anonymous free tier (Qwen3.5-397B-A17B), which is IP-rate-limited to roughly 2 requests/minute and shared with any other OVH free-tier usage from the same IP. On a 429 the coach reports the limit and the options (wait, set `OVH_API_KEY`, or select another provider with `--provider <name>`). Optional knobs:

| Variable                           | Default          | Purpose                                                                                                                          |
| ---------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                     | `ovh`            | LLM provider (`ovh` or `deepseek`)                                                                                               |
| `LLM_MODEL`                        | provider default | Model override; defaults to `Qwen3.5-397B-A17B` (ovh) or `deepseek-flash` (deepseek). Any other model needs `LLM_CONTEXT_WINDOW` |
| `LLM_THINKING`                     | `true`           | Reasoning mode (DeepSeek flag; OVH reasons server-side and ignores it)                                                           |
| `OVH_API_KEY` / `DEEPSEEK_API_KEY` | empty            | Only for the OVH paid tier / the DeepSeek provider                                                                               |
| `LLM_MAX_TOKENS`                   | `32768`          | Output budget shared by reasoning and the analysis JSON (reasoning models count reasoning against it)                            |
| `LLM_INPUT_BUDGET`                 | unset            | Optional lower cap on the request input budget (system prompt + athlete data + history); unset uses the model's usable window    |
| `LLM_CONTEXT_WINDOW`               | provider default | Override the model's context window; required for a model the app does not know (rejected without it)                            |
| `LLM_MAX_OUTPUT_TOKENS`            | provider default | Override the model's output cap (only needed for a model the app does not know)                                                  |
| `LLM_TIMEOUT_SECONDS`              | `180`            | Per-call timeout                                                                                                                 |
| `APP_TIMEZONE`                     | `Europe/Paris`   | Training-day boundaries; must match the Intervals.icu account timezone                                                           |
| `DATABASE_PATH`                    | `data/coach.db`  | Local SQLite state (proposals, messages, seen activities)                                                                                 |
| `MAX_RETRIES` / `RETRY_BASE_DELAY` | `3` / `1`        | HTTP retry policy                                                                                                                |
| `REQUESTS_PER_SECOND`              | `8`              | Intervals.icu rate-limit throttle                                                                                                |
| `ATHLETE_PROFILE` / `COACH_TONE`   | configurable     | Persona injected into every prompt                                                                                               |
| `CHAT_HISTORY_TURNS`               | `10`             | Feedback rows loaded as chat memory (>= 1)                                                                                       |
| `CHAT_HISTORY_MAX_AGE_DAYS`        | `90`             | Cutoff age for messages loaded as chat memory (>= 1)                                                                        |
| `HISTORY_DAYS`                     | `180`            | Stored history kept (messages, proposals, seen activities); 0 = keep forever (>= 0)                                      |

To use DeepSeek instead, either set `LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY` in `.env`, or override a single run without editing anything: `coach -p deepseek` (`-p` for short; the matching default model is selected automatically; add `--model`/`-m` to force one). DeepSeek's model is `deepseek-flash` (DeepSeek-V4.1-Flash, the default); selecting any other model requires `LLM_CONTEXT_WINDOW`. Inside the chat, the active provider and model are printed on startup and `/provider [name]` / `/model [name]` switch them mid-session. A provider can only be selected when it is usable: an unknown name (e.g. `ova`) lists the available providers with their credential status, and DeepSeek without a key reports `No API key for provider 'deepseek'; set DEEPSEEK_API_KEY` immediately.

```text
Unknown LLM provider: 'ova'
Available providers:
  deepseek  not ready (needs DEEPSEEK_API_KEY)
  ovh       ready (no API key needed)
```

## Safety model

Changes reach Intervals.icu only after: strict schema validation (`extra="forbid"`), a pending-only approval, and a proposal gate restating the exact plan that requires a literal `yes`. The writer resolves creates by name+date (no duplicates; race matches span any `RACE_*` priority) and refuses to update or delete anything outside the mutation's own family (workout → `WORKOUT` only, race → `RACE_*` only). No personal data is ever committed: `ATHLETE_PROFILE` lives only in `.env`, and no personal details belong in public docs, code, fixtures or issues.

## Coaching Methodology

The system is engineered to act as an elite endurance coach enforcing Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics. It prioritizes objective execution validation and autonomous fatigue modulation over generic encouragement.

The decision loop is explicit and ordered: validate planned against executed, solicit and inject the athlete's qualitative context (RPE, fatigue, schedule constraints) before finalizing, modulate the upcoming load, then generate the structural changes. User context is always injected before any calendar update.

## Initial Setup Requirements

To operate this system, the following external credentials are required:

1. **Intervals.icu API Access:** An Athlete ID and Developer API Key (requires HTTP Basic Authentication).
2. **LLM access:** None by default. The coach uses OVHcloud AI Endpoints' anonymous free tier (kepler endpoint) with Qwen3.5-397B-A17B — 397B parameters, 262K context, reasoning mode — and no API key, rate-limited to about 2 requests/minute per IP. An `OVH_API_KEY` is only needed for the paid tier. DeepSeek remains available as an alternative (`LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`, or per run with `--provider deepseek`). The LLM layer is provider-agnostic; adding another provider means adding a provider class behind the same interface.
