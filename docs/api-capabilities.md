# External Capabilities Verification (Iteration 0 Spike)

Verified 2026-08-16 against official documentation: Intervals.icu API docs thread (forum topic 609), Intervals.icu API Integration Cookbook (topic 80090), Intervals.icu OpenAPI spec (`https://intervals.icu/api/v1/docs`), DeepSeek API docs (`api-docs.deepseek.com/guides/json_mode`), Cloudflare Tunnel docs.

## 1. Intervals.icu API

### Auth

- Basic auth for personal use: username `API_KEY`, password = the API key (generated in Settings → Developer Settings).
- Athlete id `0` in path = the athlete owning the key. Recommended to always use `0`.
- OAuth + Bearer tokens only needed for multi-user apps. Not in scope.

### Rate limits (API-key callers)

- **5000 requests/day** (reset midnight UTC), **2500/rolling 15 min**, **10 req/s per IP** (silent).
- Headers `X-RateLimit-Limit` / `X-RateLimit-Remaining` (`<15m>,<daily>`); over-limit → `429` + `Retry-After` seconds.
- Implication: polling cadence and deep-historical extraction must be budgeted. Worst-case daily loop (per webhook: activity detail + wellness + events read + event writes) stays well under limits; deep-historical queries can spike — need explicit call budgeting.

### Cloudflare note (critical)

- Intervals.icu sits behind Cloudflare; requests with library-default user agents (e.g. `python-urllib`) may be challenged/blocked.
- Implication: our HTTP client must send a browser-like `User-Agent`.

### Read endpoints (our scope)

| Purpose                                | Endpoint                                                                                  | Notes                                                                                                                                                                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Activity list                          | `GET /api/v1/athlete/{id}/activities?oldest=&newest=`                                     | Summary rows; `fields` param to select columns                                                                                                                                                                              |
| Activity detail + intervals            | `GET /api/v1/activity/{id}?intervals=true`                                                | Full telemetry summaries + `icu_intervals[]` (power/HR/cadence per detected interval)                                                                                                                                       |
| Wellness (HRV, sleep, weight, CTL/ATL) | `GET /api/v1/athlete/{id}/wellness?oldest=&newest=`                                       | 46 fields: `hrv`, `hrvSDNN`, `restingHR`, `sleepSecs`, `sleepQuality`, `fatigue`, `stress`, `soreness`, `readiness`, `weight`, `ctl`, `atl`, `rampRate`, `vo2max`, `lactate` …                                              |
| Calendar events                        | `GET /api/v1/athlete/{id}/events?oldest=&newest=&category=`                               | Planned workouts; `category` = comma-separated from: WORKOUT, RACE_A, RACE_B, RACE_C, NOTE, PLAN, HOLIDAY, SICK, INJURED, SET_EFTP, FITNESS_DAYS, SEASON_START, TARGET, SET_FITNESS (invalid values → 422); `resolve` param |
| Event detail                           | `GET /api/v1/athlete/{id}/events/{eventId}`                                               | Includes `icu_training_load`, `icu_ctl`, `icu_atl`, `workout_doc`                                                                                                                                                           |
| Sport settings (FTP, zones)            | `GET /api/v1/athlete/{id}/sport-settings`                                                 | Baseline values; rows carry mixed-type ids (`id` int, `athlete_id` string)                                                                                                                                                  |
| Athlete profile                        | `GET /api/v1/athlete/{id}`                                                                | Weight, FTP, athlete info                                                                                                                                                                                                   |
| Athlete summary                        | `GET /api/v1/athlete/{id}/athlete-summary`                                                | Returns a **list** of server-aggregated rows (verified 2026-08-17), not a dict                                                                                                                                              |
| Interval search (deep-historical)      | `GET /api/v1/athlete/{id}/activities/interval-search?minSecs=&minIntensity=&type=&limit=` | Filters detected intervals by duration/intensity — useful for trend queries without pulling full activities                                                                                                                 |
| Activity file                          | `GET /api/v1/activity/{id}/file`                                                          | gzip-compressed fit/gpx/tcx                                                                                                                                                                                                 |

- **Form/TSB:** no dedicated endpoint/field; compute `form = ctl - atl` from wellness (`ctl`/`atl` fields) or event payloads.

### Write endpoints (calendar)

| Purpose                   | Endpoint                                                                 | Notes                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Create event              | `POST /api/v1/athlete/{id}/events`                                       | Dates must be `T00:00:00` local. Description is parsed server-side → `icu_training_load` and time-in-zones computed. |
| Update event              | `PUT /api/v1/athlete/{id}/events/{eventId}`                              | Same parsing behavior                                                                                                |
| Delete event              | `DELETE /api/v1/athlete/{id}/events/{eventId}`                           |                                                                                                                      |
| Bulk create/update        | `POST /api/v1/athlete/{id}/events/bulk`                                  | Batch mutations                                                                                                      |
| Bulk delete               | `PUT /api/v1/athlete/{id}/events/bulk-delete`                            |                                                                                                                      |
| Upload structured workout | `POST /api/v1/athlete/{id}/events` with `file_contents` (.zwo/.mrc/.erg) | Server parses steps → `workout_doc`                                                                                  |

- Payload fields for a workout event: `category: WORKOUT`, `start_date_local`, `name`, `description`, `type` (e.g. `Ride`), `moving_time`, `icu_training_load`, optionally `color`, `folders_id`, etc.
- **Implication:** our writer can either set free-form `description` + estimated `moving_time`/load (server computes load), or emit full `.zwo` `file_contents` for structured workouts. Decision point for iteration 4: start with description-based, upgrade to .zwo later.

### Webhooks

- Configured in the app's **Manage App** page (OAuth apps) — **for personal API-key accounts, verify availability of a webhooks section under Settings → Developer Settings** (empirical check pending; docs only describe app-based webhooks).
- Payload shape:

  ```json
  { "secret": "...", "events": [ { "athlete_id": "...", "type": "ACTIVITY_UPLOADED", "timestamp": "...", "activity": {...} } ] }
  ```

- `secret` field enables sender verification.
- Event types documented: `ACTIVITY_UPLOADED`, `ACTIVITY_ANALYZED` (delivered after ~60 s delay to consolidate events), `CALENDAR_UPDATED` (contains `events[]` + `deleted_events[]`; preferred over legacy `CALENDAR_EVENT_UPDATED`/`CALENDAR_EVENT_DELETED`), `SPORT_SETTINGS_UPDATED` (FTP/zones changed).
- **No wellness-specific webhook type is documented.** Wellness trigger requires either verification in-app or a scheduled wellness poll (see open items).
- Activity webhooks are **not** delivered for Strava-synced activities (user uses Garmin — unaffected, but documented).
- **No missed-delivery notification exists.** A webhook can be lost on downtime with no indicator. Implication: periodic reconciliation sync (poll `activities` + `events` since last sync) is mandatory regardless of webhook reliability.

## 2. DeepSeek API

- Endpoint: `https://api.deepseek.com`, OpenAI-compatible client (`openai` package with `base_url`).
- Model line: `deepseek-chat` / `deepseek-reasoner` (docs show `deepseek-v4-pro` as current flagship).
- **JSON output mode (verified):**
  - `response_format: {"type": "json_object"}`.
  - The prompt **must contain the word "json"** and an example of the desired schema, otherwise JSON mode does not engage.
  - `max_tokens` must be set high enough to avoid mid-JSON truncation.
  - **Known issue:** API occasionally returns empty `content` — our parser must retry (bounded, e.g. 3 attempts) on empty/parse-failure.
- Context caching available (`guides/kv_cache`) — relevant if we send a large stable persona/system block repeatedly.
- Multi-round conversation + tool calls supported (relevant for the interactive CLI chat).

## 3. Cloudflare Tunnel (webhook ingress)

- `cloudflared` quickstart verified: `login` → `tunnel create <NAME>` → `config.yml` (`url: http://localhost:<port>`, tunnel UUID, credentials file) → `tunnel route dns <NAME> <hostname>` → `tunnel run <NAME>`.
- Can install as a Linux systemd service (`cloudflared service install`).
- **Prerequisite:** a domain added to Cloudflare (nameservers on Cloudflare). For ad-hoc testing without a domain: quick tunnels (`cloudflared tunnel --url http://localhost:8000`) provide a temporary public URL.
- Outbound needs port 7844 reachable to Cloudflare.

## 4. Open items (hands-on verification required)

1. **Personal webhook config:** RESOLVED 2026-08-16 — in-app check: Developer Settings contains only API key, athlete ID, and connected apps. No webhooks section for personal API-key accounts. Webhooks require an OAuth app (email <david@intervals.icu>). **Consequence: manual-first trigger architecture; webhooks optional later.**
2. **Wellness trigger:** moot under manual-first triggers; wellness data is pulled on demand and by an optional scheduled poll later. Revisit if an OAuth app is created.
3. **Exact webhook event fields** for `ACTIVITY_UPLOADED`/`ACTIVITY_ANALYZED` payloads (activity object shape) — only needed if a webhook adapter is added.
4. **DeepSeek model selection:** confirm current model id and context window on the pricing page; measure token usage of a typical prompt before finalizing extractor budgets.
5. **Tunnel hostname:** RESOLVED — a Cloudflare-managed domain is available for a permanent tunnel hostname.

## 5. Design implications recorded for later iterations

- HTTP client: browser-like `User-Agent`, Basic auth, retry with `Retry-After` respect, rate-limit header tracking (budgeting for deep-historical queries).
- Token budgets: prefer `fields` filtering, interval summaries over raw streams; wellness has 46 fields — select columns via `cols`/`fields` params.
- When a webhook adapter is added later (OAuth app), ack-first + in-process queue is confirmed correct: ACTIVITY_ANALYZED already has a 60 s consolidation delay, so processing after ack is the native pattern.
- **Polling cadence (rate-limit math, for the optional background poller):** activities list every 5 min ≈ 288 calls/day; wellness every 30 min ≈ 48/day; per-analysis extraction ≈ 5–10 calls; total stays far below the 5000/day limit. Polling is also self-healing for missed webhooks — no separate reconciliation job needed.
- Writer: start with description-based workouts (server computes load/time-in-zones), optional .zwo upgrade later.

## 6. Addendum (from intervals-icu-sync reference review, 2026-08-16)

- **Strava-sourced activities:** the API does not expose power/detailed metrics for activities synced via Strava (direct Garmin/device sync is required). Matches the documented webhook exclusion for Strava activities.
- **`ATHLETE_ID` location confirmed:** Settings → Developer Settings in Intervals.icu shows both the API key and the athlete ID (their `.env` uses `INTERVALS_API_KEY` + `ATHLETE_ID`).
- **Weekly load history:** the reference project reads `GET /api/v1/athlete/{id}/athlete-summary{ext}` for server-aggregated weekly training-load history — candidate for our standard extraction scope.
- **Upload idempotency pattern (reference):** index existing WORKOUT events by `(name, date)` in the target range → `PUT` update if found, else `POST` create; safe re-runs, no duplicates. Structured workouts upload as `.zwo` `file_contents` with step definitions; tags are supported on events.
- **Tunnel + localhost note:** the reference MCP setup requires an allowed-host configuration for the tunnel's public hostname (`Host` header) — any local receiver will need the same when wired to the tunnel.

## 7. Addendum (payload field shapes verified against live data, 2026-08-17/18)

Verified by recording real read-only payloads into anonymized test fixtures (`tests/fixtures/`) and cross-checking the OpenAPI spec (`https://intervals.icu/api/v1/docs`). Field names below differ from intuitive guesses:

- **Activity rows:** power is `icu_average_watts` / `icu_weighted_avg_watts` — there is **no `avg_power`**. RPE is `icu_rpe` (with `perceived_exertion` and `session_rpe` as separate fields). List rows carry no `icu_intervals`; only `GET /activity/{id}?intervals=true` returns them. Activities carry `group` (hash token) and generated `interval_summary` strings (e.g. `1x 15s 187bpm`).
- **Intervals (`icu_intervals[]`):** ids are ints; `start_time`/`end_time` are seconds within the activity, not epochs; `type`/`zone`/`intensity` are Intervals-generated classifications.
- **Wellness:** `id` is the date string (`YYYY-MM-DD`); 46 fields including nested `sportInfo[]` (`type`, `eftp`, `wPrime`, `pMax`); `cols`/`fields` params select columns.
- **Sport settings:** `id` is an int but `athlete_id` is a string; `power_zones`/`hr_zones` are 7-boundary numeric lists.
- **Athlete summary:** returns a list of server-aggregated rows, not a single object.

## 8. Workout description text format (structured workouts)

Verified 2026-08-22 against official Intervals.icu forum documentation. The `description` field of WORKOUT events is parsed server-side into structured steps (`workout_doc`, time-in-zones, training load). Quoted statements:

- API access guide (topic 609, david): "The workout description is now parsed. This means that workouts created or updated via the API have training load calculated and get 'time in zones' and so on."
- Uploading planned workouts guide (topic 63624, david): "you can use 'description' and supply native Intervals.icu workout text".

Server-rendered description of a parsed workout (topic 609, response to a `.zwo` upload — this is the format the server itself produces):

```text
- 20m 60% 90-100rpm

Main set 4x
- 8m 110%
- 8m 50%

- 10m 60%
```

Working API payload description (topic 63624, forum user example, Feb 2026 — posted with the event fields `"target": "POWER"` and `"workout_doc": {}`; the server fills `workout_doc`):

```text
- 15m 55% Warmup

3x
- 1m 150%
- 1m 50%

- 5m 50%
- 5m 120%
- 15m 55%
```

Format rules, from the official workout builder doc (topic 1163, david):

> - Create workout steps by starting a line with a `-` and using the following constructs: a duration "30s", "10m", "1m30" etc.; "100w, 80% (of FTP), 60% HR (of max heart rate), 100% LTHR (of threshold HR), 90 rpm (cadence)"; ranges "100-140w, 80-90% (of FTP) etc."; ramps "Ramp 100-200w" or "Ramp 60-80% (of FTP)"; "and whatever additional text you like".
> - "Create repeats by including '6x' or whatever in the line before a set of steps."
> - Zones (15 April 2022 update): "`- 60m Z2`" for zone 2 power, "`- 60m Z2 HR`" using heart rate, "Pace also works".
> - "Steps can have text prompts. All the text prior to the duration or power specification becomes the text for the step." (`Recovery 30s 50%` → prompt "Recovery").

Distance units (topic 9973, david): km, mi, mtr, meters, yrd, yards, y, miles, mile; a space is allowed ("1km", "1 km"). "The minutes are denoted `m`'s in Intervals... impossible to use meters as a unit" — plain `m` is minutes, never meters.

Absolute pace (topic 115846, david): `- 10m 7:15-7:00 Pace`, explicit units `/km`, `/mi`, `/100m`, `/500m`, `/100y`, `/400m`, `/250m`.

Parsed structure (topic 93737, david): `workout_doc` steps carry `text`, `duration`, `distance`, `reps` (+ nested `steps`), `warmup`/`cooldown`, and `power`/`hr`/`pace`/`cadence` values (`value`/`start`/`end`/`units`); `target` = POWER/HR/PACE.

### Live-verified additions (2026-08-22, real API write/read-back, not in the official docs)

| Construct                                                                      | Observed parse                                                                                 |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `- 10m20s Z2-Z3 HR`                                                            | `duration: 620`, `hr: {start: 2, end: 3, units: "hr_zone"}` (compound duration, HR zone range) |
| `- 2.5km Z2 HR`                                                                | `distance: 2500`, `hr: {units: "hr_zone", value: 2}`                                           |
| `- 400mtr Z1 HR`, `- 25mtr Z5 HR`, `- 0.1km Z2 HR`                             | distance steps in meters                                                                       |
| `- 0.1km 1:45/100m Pace`                                                       | `distance: 100`, `pace: {units: "secs/100m", value: 105}`                                      |
| `Main set 4x` + indented steps **with blank lines before and after the block** | `{reps: 4, steps: [...]}`, multiplied distance/duration/zoneTimes                              |
| Prose lines mixed with step lines                                              | prose ignored for steps (kept as `workout_doc.description`)                                    |

Live-confirmed pitfalls:

- Bare `100m` parses as **100 minutes** (6000s), not 100 meters — sub-km distances must be `mtr` or km fractions.
- A repeat block without blank-line separation loses its `reps` (steps stay flat, load not multiplied).
- A dash-prefixed repeat line (`- 4x`) is a step line, not a repeat header.
