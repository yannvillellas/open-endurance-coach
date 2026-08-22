# Open Endurance Coach

A self-hosted, AI-driven endurance coaching system.

Open Endurance Coach integrates multi-sport telemetry from Intervals.icu with Large Language Model analysis to validate training execution and adjust future workouts. Every proposed calendar change is reviewed and approved by the athlete before anything is written.

## Current Capabilities

- **Data Extraction:** Standard scope (recent activities, wellness, upcoming events, sport settings) and deep-historical scope (trend queries such as "heart rate improvement on hills over the last 3 months"), both budgeted to fit the model's token limit.
- **Analysis:** DeepSeek (JSON mode, thinking enabled) enforces Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics, comparing executed training against planned targets and current readiness (CTL/ATL, HRV, sleep).
- **Draft & Review Loop:** Every analysis produces a validated draft under a strict schema — invalid LLM output is retried, then rejected. The coach solicits missing RPE/fueling data and re-analyzes with the athlete's feedback before anything can be approved.
- **Calendar Writer:** Approved decisions are applied to Intervals.icu with idempotent create/update and a WORKOUT-only category guard — updates and deletes never touch race or non-workout events. Applying defaults to a dry-run.
- **Manual-First Triggers:** The CLI drives the loop today. Webhook triggers (activity uploaded/analyzed, calendar updated) and a wellness poller are on the roadmap behind the same engine; an Intervals.icu OAuth app has been created for that step.

## Chat mode (the main interface)

`coach chat` is a single conversation with the coach. You just talk:

- **Free text runs the right thing automatically.** When fresh data is needed (first message, trend questions, or requests like "analyze/review/check my week"), the coach runs a full analysis and answers with the report. Otherwise he answers conversationally from the same data snapshot.
- **When he proposes calendar changes**, he asks: "Apply this to Intervals.icu: …". Reply with exactly `yes` and the changes are validated, approved, and written in one step. `no` declines, and **anything else is a change request** — he re-analyzes with your words and proposes again. Nothing is ever written without a literal yes.
- **Memory**: sessions remember recent exchanges (last 10 feedback rows, up to 2048 tokens, self-trimmed). Start fresh with `coach chat --fresh`, or say `/clear` at any time.
- Commands are optional: `/analyze` forces a fresh analysis, `/help`, `/exit` — everything else is conversation.

```text
$ coach chat
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

## One-shot CLI (automation API)

The commands below are the scriptable surface — the chat above is the daily interface. Approve and apply stay two separate steps here because automation needs record-then-write.

```text
coach ask <question>          Ask the coach anything (trend queries supported)
coach analyze [focus]         Post-training analysis (default focus used when omitted)
coach review [draft_id]       List pending drafts or inspect one
coach feedback <id> <text>    Answer the coach's questions on a draft
coach approve <id>            Approve a draft (yes/no confirmation; --yes to skip)
coach reject <id>             Discard a draft (yes/no confirmation; --yes to skip)
coach apply [decision_id]     Dry-run by default; --write applies (needs confirmation)
```

Options: `--feedback` (inject subjective context into ask/analyze), `--mutations-file` (approve with your own workout mutations), `--write`/`--yes` (apply).

## Modes

- **Analysis** (free text in chat when data is needed, or `analyze` one-shot): extracts a fresh data snapshot (recent activities, wellness, upcoming events, sport settings — or a 90-day filtered window for trend questions), runs the strict JSON analysis, saves the report internally as a draft, and marks the analyzed activities as seen so they surface as "New activities since last review" only once. Nothing is written to the calendar here.
- **Conversation** (free text in chat otherwise): prose answers from the cached snapshot; never creates plans, never marks activities seen.
- **Proposal gate**: every calendar change is proposed with the exact plan and requires a literal `yes`; any other answer is treated as a change request or discussion and writes nothing. Mid-confirmation Ctrl-C cancels safely.

## Configuration

All settings come from environment variables or a `.env` file (see `.env.example`). Essentials: `INTERVALS_API_KEY`, `INTERVALS_ATHLETE_ID`, `DEEPSEEK_API_KEY`. Optional knobs:

| Variable                           | Default                    | Purpose                                                                |
| ---------------------------------- | -------------------------- | ---------------------------------------------------------------------- |
| `LLM_MODEL` / `LLM_THINKING`       | `deepseek-v4-pro` / `true` | Model and reasoning mode                                               |
| `LLM_MAX_TOKENS`                   | `8192`                     | Output budget for the analysis JSON                                    |
| `LLM_TIMEOUT_SECONDS`              | `180`                      | Per-call timeout                                                       |
| `APP_TIMEZONE`                     | `Europe/Paris`             | Training-day boundaries; must match the Intervals.icu account timezone |
| `DATABASE_PATH`                    | `data/coach.db`            | Local SQLite state (drafts, decisions, feedback)                       |
| `MAX_RETRIES` / `RETRY_BASE_DELAY` | `3` / `1`                  | HTTP retry policy                                                      |
| `REQUESTS_PER_SECOND`              | `8`                        | Intervals.icu rate-limit throttle                                      |
| `ATHLETE_PROFILE` / `COACH_TONE`   | configurable               | Persona injected into every prompt                                     |
| `CHAT_HISTORY_TURNS`               | `10`                       | Feedback rows loaded as chat memory                                    |
| `CHAT_HISTORY_MAX_TOKENS`          | `2048`                     | Chat memory budget (self-trimmed)                                      |

## Safety model

Changes reach Intervals.icu only after: strict schema validation (`extra="forbid"`), a pending-only approval, and a proposal gate restating the exact plan that requires a literal `yes` (or explicit `--yes` in scripts). The writer resolves creates by name+date (no duplicates) and refuses to update or delete anything that is not a WORKOUT-category event. Applying defaults to a dry-run.

## Coaching Methodology

The system is engineered to act as an elite endurance coach enforcing Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics. It prioritizes objective execution validation and autonomous fatigue modulation over generic encouragement.

## Initial Setup Requirements

To operate this system, the following external credentials are required:

1. **Intervals.icu API Access:** An Athlete ID and Developer API Key (requires HTTP Basic Authentication).
2. **DeepSeek API Key:** For LLM inference with JSON schema output. The LLM layer exposes a provider-agnostic interface; DeepSeek V4 Pro (thinking mode) is the current provider and models are switched via configuration. Adding another provider means adding a provider class behind the same interface.
