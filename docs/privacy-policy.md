# Privacy Policy — Open Endurance Coach

Last updated: 2026-08-17

Open Endurance Coach ("the tool") is a self-hosted, single-user AI endurance
coaching application. This policy describes how it handles data.

## Data accessed

The tool reads and writes data from a single Intervals.icu account — the
account of the operator who installed and configured it — using the operator's
own credentials (personal API key, and optionally OAuth authorization granted
to this application). The following categories may be accessed:

- Activities (completed training sessions) and their telemetry summaries
- Wellness data (weight, resting heart rate, HRV, sleep, readiness)
- Calendar events and planned workouts
- Athlete profile and sport settings (FTP, zones)

## Data storage

All data is stored locally on the operator's machine (local files and a local
SQLite database). No data is transmitted to, or stored on, any server operated
by this project. There is no remote service component.

## Data sharing

- The operator's configured LLM provider (by default DeepSeek) receives
  excerpts of training data in prompts, solely to generate coaching analysis.
  Refer to the provider's own privacy policy for how they process requests.
- No other third party receives any data. The tool does not contain analytics,
  telemetry, or tracking.

## Operator control

The operator can revoke access at any time by regenerating the Intervals.icu
API key, disconnecting the OAuth application in Intervals.icu settings, or
deleting the local database and configuration.

## Contact

Questions about this policy: open an issue in the project repository
(<https://github.com/yannvillellas/open-endurance-coach>).
