# Contributing

Thanks for your interest in Open Endurance Coach.

## Before you start

- Work starts from a GitHub issue: read it before writing code. If there is none, open one (bug report or feature request).
- This is a self-hosted, single-athlete app that talks to Intervals.icu. You need an Intervals.icu API key to run it against real data; nothing reaches the calendar without a literal `yes` from the operator.
- Never include secrets, credentials, `.env` contents, or personal health data in issues, code, tests, fixtures, or commits.

## Development setup

```sh
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in INTERVALS_API_KEY and INTERVALS_ATHLETE_ID
```

## Checks (must pass before a pull request)

```sh
ruff check .
ruff format --check .
mypy src tests
pytest -q --cov=open_endurance_coach
pip-audit
```

The CI fails the tests job below **95% total coverage**.

The CI audits the dependencies on every pull request. Dependabot watches unchanged pins
(security alerts and version-update PRs), and every dependency PR goes through the same
checks.

## Pull requests

- One change per pull request, scoped to the linked issue; reference it with `Closes #<issue>`.
- Write or update tests before the implementation.
- Keep commits small and use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).
- Document new data flows and API usage (`README.md` or `docs/`).
- A review freezes the SHA it reviewed; post-review changes go on a new commit or branch.

## Agents

AI agents working in this repository also follow [`AGENTS.md`](AGENTS.md).
