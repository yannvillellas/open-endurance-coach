# Open Endurance Coach

A self-hosted, AI-driven endurance coaching system.

Open Endurance Coach integrates multi-sport telemetry from Intervals.icu with Large Language Model analysis to validate training execution and adjust future workouts. Every proposed calendar change is reviewed and approved by the athlete before anything is written.

## Current Capabilities

- **Data Extraction:** Standard scope (recent activities, wellness, upcoming events, sport settings, goal races over a 120-day horizon, and a 90-day weekly training rollup with CTL/ATL/ramp and per-sport load) and deep-historical scope (trend queries such as "heart rate improvement on hills over the last 3 months"), both budgeted to fit the model's token limit.
- **Analysis:** OVHcloud AI Endpoints' free tier (Qwen3.5-397B-A17B, JSON mode, thinking enabled; no API key) — or DeepSeek — enforces Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics, comparing executed training against planned targets and current readiness (CTL/ATL, HRV, sleep).
- **Draft & Review Loop:** Every analysis produces a validated draft under a strict schema — invalid LLM output is retried, then rejected. The coach solicits missing RPE/fueling data and re-analyzes with the athlete's feedback before anything can be approved.
- **Race-Aware Planning:** With a goal race in the calendar the coach plans backwards from it — states the macro phases (Base, Build, Peak, Taper) with weekly load targets, then proposes concrete workouts for the next 7–14 days. Races are first-class events (`RACE_A/B/C`) it can create or adjust; missing race details (distance, climbing, expected load) are asked for, never estimated.
- **Calendar Writer:** Approved decisions are applied to Intervals.icu with idempotent create/update and strict category guards — workout mutations only touch `WORKOUT` events, race mutations only touch `RACE_A/B/C` events, and updates/deletes never cross between the two families. Applying defaults to a dry-run.
- **Manual-First Triggers:** The CLI drives the loop today. Webhook triggers (activity uploaded/analyzed, calendar updated) and a wellness poller are on the roadmap behind the same engine; an Intervals.icu OAuth app has been created for that step.

## Chat mode (the main interface)

`coach` opens a single conversation with the coach. You just talk:

- **Free text runs the right thing automatically.** When fresh data is needed (first message, trend questions, or requests like "analyze/review/check my week"), the coach runs a full analysis and answers with the report. Otherwise he answers conversationally from the same data snapshot.
- **The coach decides what you need.** Every message is classified as _chat_ (answer from the analysis already in the session), _analysis_ (review executed training), or _plan_ (propose calendar changes). A full analysis is reused for follow-ups — there is no re-analysis until you ask for one or the question needs historical depth. If a change would help during a chat, he offers it instead of interrupting you with a confirmation gate.
- **Material questions block proposals, for any plan** — a training block, a race event, or both. If an answer would change the plan (available training days, constraints, injury, RPE, race duration/climbing/expected load), the coach asks and does _not_ propose calendar changes until you answer — or say `proceed with assumptions` and he states the assumption in the plan.
- **When he proposes calendar changes**, he asks: "Apply this to Intervals.icu: …". Reply with exactly `yes` and the changes are validated, approved, and written in one step. `no` declines, and **anything else is a change request** — he re-analyzes with your words and proposes again. Nothing is ever written without a literal yes.
- **Memory**: sessions remember recent exchanges (last 10 feedback rows from the last 90 days, up to 2048 tokens, self-trimmed). Start fresh with `coach --fresh`, or say `/clear` at any time.
- Session commands only: `/provider` and `/model` show or switch the LLM, `/clear`, `/help`, `/exit` — everything else is conversation.

```text
$ coach
Chat with the coach. Remembering 2 past exchanges.

you: how was my week?
Coach: (full report — summary, findings, questions)
Coach: Apply this to Intervals.icu:
        - create Tempo Session on 2026-08-25
you: make it 45 minutes instead
Coach: (re-analyzes, shows the updated proposal)
you: yes
Coach: Applied: create created: 130130106
```

## Approvals

Calendar changes never happen silently: when the coach proposes adjustments he asks for a literal `yes` before anything is written, and `no` or `cancel` abandons the proposal. Everything happens inside the conversation.

## Configuration

All settings come from environment variables or a `.env` file (see `.env.example`). Essentials: `INTERVALS_API_KEY`, `INTERVALS_ATHLETE_ID`. No LLM API key is required by default: the coach uses OVHcloud AI Endpoints' anonymous free tier (Qwen3.5-397B-A17B), which is IP-rate-limited to roughly 2 requests/minute and shared with any other OVH free-tier usage from the same IP. On a 429 the coach reports the limit and the options (wait, set `OVH_API_KEY`, or select another provider with `--provider <name>`). Optional knobs:

| Variable                           | Default          | Purpose                                                                                                                       |
| ---------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                     | `ovh`            | LLM provider (`ovh` or `deepseek`)                                                                                            |
| `LLM_MODEL`                        | provider default | Model override; defaults to `Qwen3.5-397B-A17B` (ovh) or `deepseek-flash` (deepseek). DeepSeek also accepts `deepseek-v4-pro` |
| `LLM_THINKING`                     | `true`           | Reasoning mode (DeepSeek flag; OVH reasons server-side and ignores it)                                                        |
| `OVH_API_KEY` / `DEEPSEEK_API_KEY` | empty            | Only for the OVH paid tier / the DeepSeek provider                                                                            |
| `LLM_MAX_TOKENS`                   | `8192`           | Output budget for the analysis JSON                                                                                           |
| `LLM_TIMEOUT_SECONDS`              | `180`            | Per-call timeout                                                                                                              |
| `APP_TIMEZONE`                     | `Europe/Paris`   | Training-day boundaries; must match the Intervals.icu account timezone                                                        |
| `DATABASE_PATH`                    | `data/coach.db`  | Local SQLite state (drafts, decisions, feedback)                                                                              |
| `MAX_RETRIES` / `RETRY_BASE_DELAY` | `3` / `1`        | HTTP retry policy                                                                                                             |
| `REQUESTS_PER_SECOND`              | `8`              | Intervals.icu rate-limit throttle                                                                                             |
| `ATHLETE_PROFILE` / `COACH_TONE`   | configurable     | Persona injected into every prompt                                                                                            |
| `CHAT_HISTORY_TURNS`               | `10`             | Feedback rows loaded as chat memory (>= 1)                                                                                    |
| `CHAT_HISTORY_MAX_TOKENS`          | `2048`           | Chat memory budget (self-trimmed) (>= 1)                                                                                      |
| `CHAT_HISTORY_MAX_AGE_DAYS`        | `90`             | Cutoff age for feedback rows loaded as chat memory (>= 1)                                                                     |

To use DeepSeek instead, either set `LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY` in `.env`, or override a single run without editing anything: `coach -p deepseek` (`-p` for short; the matching default model is selected automatically; add `--model`/`-m` to force one). DeepSeek model IDs: `deepseek-flash` (DeepSeek-V4.1-Flash, the default) and `deepseek-v4-pro` (DeepSeek-V4-Pro) — e.g. `coach -p deepseek -m deepseek-v4-pro` or `LLM_MODEL=deepseek-v4-pro`; `/model deepseek-v4-pro` switches mid-session. Inside the chat, the active provider and model are printed on startup and `/provider [name]` / `/model [name]` switch them mid-session. A provider can only be selected when it is usable: an unknown name (e.g. `ova`) lists the available providers with their credential status, and DeepSeek without a key reports `No API key for provider 'deepseek'; set DEEPSEEK_API_KEY` immediately.

```text
Unknown LLM provider: 'ova'
Available providers:
  deepseek  not ready (needs DEEPSEEK_API_KEY)
  ovh       ready (no API key needed)
```

## Safety model

Changes reach Intervals.icu only after: strict schema validation (`extra="forbid"`), a pending-only approval, and a proposal gate restating the exact plan that requires a literal `yes`. The writer resolves creates by name+date (no duplicates; race matches span any `RACE_*` priority) and refuses to update or delete anything outside the mutation's own family (workout → `WORKOUT` only, race → `RACE_*` only). Applying defaults to a dry-run.

## Coaching Methodology

The system is engineered to act as an elite endurance coach enforcing Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics. It prioritizes objective execution validation and autonomous fatigue modulation over generic encouragement.

## Initial Setup Requirements

To operate this system, the following external credentials are required:

1. **Intervals.icu API Access:** An Athlete ID and Developer API Key (requires HTTP Basic Authentication).
2. **LLM access:** None by default. The coach uses OVHcloud AI Endpoints' anonymous free tier (kepler endpoint) with Qwen3.5-397B-A17B — 397B parameters, 262K context, reasoning mode — and no API key, rate-limited to about 2 requests/minute per IP. An `OVH_API_KEY` is only needed for the paid tier. DeepSeek remains available as an alternative (`LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`, or per run with `--provider deepseek`). The LLM layer is provider-agnostic; adding another provider means adding a provider class behind the same interface.
