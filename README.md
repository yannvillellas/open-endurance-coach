# Open Endurance Coach

A self-hosted AI endurance coach. It reads your training from
[Intervals.icu](https://intervals.icu), reviews it against Joe Friel's periodization
principles and Coggan-style power analysis, and proposes calendar changes — which you
approve with a literal `yes` before anything is written.

Early, single-user and self-hosted. The CLI (`coach`) is the only interface today.

## Quick start

1. Create an Intervals.icu API key and note your athlete id (Intervals → Settings →
   Developer).
2. Install and configure:

   ```sh
   git clone https://github.com/yannvillellas/open-endurance-coach
   cd open-endurance-coach
   python3.12 -m venv .venv
   .venv/bin/pip install -e .
   cp .env.example .env     # set INTERVALS_API_KEY and INTERVALS_ATHLETE_ID
   ```

3. Start the coach:

   ```sh
   .venv/bin/coach
   ```

No LLM key is required by default: the coach uses OVHcloud's anonymous free tier
(Qwen3.5-397B-A17B, roughly 2 requests/minute per IP). Set `OVH_API_KEY` for the paid
tier, or `LLM_PROVIDER=deepseek` with `DEEPSEEK_API_KEY`. Every setting is in
[docs/configuration.md](docs/configuration.md).

Abridged example session:

```text
$ coach
Chat with the coach. /help lists commands.

you: how was my week?
Coach:
  Race is 10 days out; the taper shape is correct.
Evidence:
  - Load: 236 last week, 77 so far this week.
Open questions:
  ? Do you want the rest week on the calendar?
Confirm? Reply exactly yes to apply, no to discard.
Or describe a change to revise the plan.

you: make it 45 minutes instead
Coach: (re-analyzes and shows the updated plan)
you: yes
Applied:
  - create -> created event 130130106
```

## How it works

- **Snapshot** — each session extracts what the question needs from Intervals.icu
  (recent activities, a ±14-day calendar window, wellness, sport settings, goal races,
  a 90-day rollup), budgeted to the model's token limit; trend questions pull a wider,
  pinned window. See [docs/architecture.md](docs/architecture.md).
- **Analysis** — the model answers through a strict JSON schema (invalid output is
  retried, then rejected) using periodization and power analytics, comparing planned
  against executed and current readiness (CTL/ATL, HRV, sleep).
- **Questions before plans** — if something material is missing (availability,
  constraints, injury, RPE, race details) the coach asks and does not propose changes
  until you answer, or you say `proceed with assumptions`.
- **Approval gate** — calendar changes appear as a numbered plan and need a literal
  `yes`; subsets work (`yes 1 3`, `yes except 2`, `yes except thursday`), anything else
  is a change request, and nothing is written without approval. Details:
  [docs/chat.md](docs/chat.md).
- **Writer** — applied changes are idempotent by name+date and family-scoped (workouts
  touch only `WORKOUT`, races only `RACE_*`); a create that would overwrite a different
  existing event is refused.
- **Memory** — sessions keep recent exchanges under the model's input budget; stored
  history is pruned after `HISTORY_DAYS` (default 180). `/forget`, `/forget N` and
  `/forget all` work in chat.

## Safety

Nothing reaches Intervals.icu without a literal `yes` for the exact plan shown. Every
LLM answer is schema-validated, the writer never crosses event families, and untrusted
text is neutralised at the display, prompt and write boundaries. No personal data
belongs in the repo: `ATHLETE_PROFILE` and `COACH_TONE` live only in `.env`.

Threat model, accepted residuals and the provider adapter contract:
[docs/safety-model.md](docs/safety-model.md).

## Development

```sh
.venv/bin/pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src tests && pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).
MIT licensed.

## Documentation

- [docs/architecture.md](docs/architecture.md) — data scopes, analysis pipeline, writer
- [docs/chat.md](docs/chat.md) — chat behaviour, approval gate, memory and commands
- [docs/configuration.md](docs/configuration.md) — every environment variable
- [docs/safety-model.md](docs/safety-model.md) — threat model and hard limits
- [docs/api-capabilities.md](docs/api-capabilities.md) — hub and provider capabilities
- [docs/privacy-policy.md](docs/privacy-policy.md)
