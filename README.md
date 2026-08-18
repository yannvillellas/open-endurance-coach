# Open Endurance Coach

A self-hosted, AI-driven endurance coaching system.

Open Endurance Coach integrates multi-sport telemetry from Intervals.icu with Large Language Model analysis to validate training execution and adjust future workouts. Every proposed calendar change is reviewed and approved by the athlete before anything is written.

## Current Capabilities

- **Data Extraction:** Standard scope (recent activities, wellness, upcoming events, sport settings) and deep-historical scope (trend queries such as "heart rate improvement on hills over the last 3 months"), both budgeted to fit the model's token limit.
- **Analysis:** DeepSeek (JSON mode, thinking enabled) enforces Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics, comparing executed training against planned targets and current readiness (CTL/ATL, HRV, sleep).
- **Draft & Review Loop:** Every analysis produces a validated draft under a strict schema — invalid LLM output is retried, then rejected. The coach solicits missing RPE/fueling data and re-analyzes with the athlete's feedback before anything can be approved.
- **Calendar Writer:** Approved decisions are applied to Intervals.icu with idempotent create/update and a WORKOUT-only category guard — updates and deletes never touch race or non-workout events. Applying defaults to a dry-run.
- **Manual-First Triggers:** The CLI drives the loop today. Webhook triggers (activity uploaded/analyzed, calendar updated) and a wellness poller are on the roadmap behind the same engine; an Intervals.icu OAuth app has been created for that step.

## CLI

```text
coach ask <question>          Ask the coach anything (trend queries supported)
coach analyze [focus]         Post-training analysis (optional --feedback)
coach review [draft_id]       List pending drafts or inspect one
coach feedback <id> <text>    Answer the coach's questions on a draft
coach approve <id>            Approve a draft (optional --mutations-file)
coach reject <id>             Discard a draft
coach apply [decision_id]     Dry-run by default; --write applies to the calendar
```

## Coaching Methodology

The system is engineered to act as an elite endurance coach enforcing Joe Friel's periodization principles and Dr. Andrew Coggan's power analytics. It prioritizes objective execution validation and autonomous fatigue modulation over generic encouragement.

## Initial Setup Requirements

To operate this system, the following external credentials are required:

1. **Intervals.icu API Access:** An Athlete ID and Developer API Key (requires HTTP Basic Authentication).
2. **DeepSeek API Key:** For LLM inference with JSON schema output. The LLM layer exposes a provider-agnostic interface; DeepSeek V4 Pro (thinking mode) is the current provider and models are switched via configuration. Adding another provider means adding a provider class behind the same interface.
