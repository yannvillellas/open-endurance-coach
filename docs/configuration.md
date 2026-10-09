# Configuration

All settings come from environment variables or a `.env` file (see `.env.example`).
Essentials: `INTERVALS_API_KEY` and `INTERVALS_ATHLETE_ID`. No LLM API key is required
by default: the coach uses OVHcloud AI Endpoints' anonymous free tier
(Qwen3.5-397B-A17B), which is IP-rate-limited to roughly 2 requests/minute and shared
with any other OVH free-tier usage from the same IP. On a 429 the coach reports the
limit and the options (wait, set `OVH_API_KEY`, or select another provider with
`--provider <name>`).

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
| `DATABASE_PATH`                    | `data/coach.db`  | Local SQLite state (proposals, messages, seen activities)                                                                        |
| `MAX_RETRIES` / `RETRY_BASE_DELAY` | `3` / `1`        | HTTP retry policy                                                                                                                |
| `REQUESTS_PER_SECOND`              | `8`              | Intervals.icu rate-limit throttle                                                                                                |
| `ATHLETE_PROFILE` / `COACH_TONE`   | configurable     | Persona injected into every prompt                                                                                               |
| `CHAT_HISTORY_TURNS`               | `10`             | Feedback rows loaded as chat memory (>= 1)                                                                                       |
| `CHAT_HISTORY_MAX_AGE_DAYS`        | `90`             | Cutoff age for messages loaded as chat memory (>= 1)                                                                             |
| `HISTORY_DAYS`                     | `180`            | Stored history kept (messages, proposals, seen activities); 0 = keep forever (>= 0)                                              |

To use DeepSeek instead, either set `LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY` in
`.env`, or override a single run without editing anything: `coach -p deepseek` (`-p`
for short; the matching default model is selected automatically; add `--model`/`-m` to
force one). DeepSeek's model is `deepseek-flash` (DeepSeek-V4.1-Flash, the default);
selecting any other model requires `LLM_CONTEXT_WINDOW`. Inside the chat, the active
provider and model are printed on startup and `/provider [name]` / `/model [name]`
switch them mid-session. A provider can only be selected when it is usable: an unknown
name (e.g. `ova`) lists the available providers with their credential status, and
DeepSeek without a key reports `No API key for provider 'deepseek'; set
DEEPSEEK_API_KEY` immediately.

```text
Unknown LLM provider: 'ova'
Available providers:
  deepseek  not ready (needs DEEPSEEK_API_KEY)
  ovh       ready (no API key needed)
```
