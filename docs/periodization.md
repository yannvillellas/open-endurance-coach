# Periodization

The race block the coach plans against: phases, weekly roles, target load bands and the
status of each week. It is a pure, time-invariant derivation (`extractors/periodization.py`)
of the anchor race, the week grid and the executed history, recomputed every turn and never
persisted. The result travels as the `block_plan` context section, one row per week,
injected right after `goal_races` and rendered in its own `<block_plan>` prompt block as a
compact Markdown table; #66 compares executed weeks against its adjusted bands.

## Doctrine

### The macrocycle (K.I.S.S.) [1]

Prep 2-4 weeks (optional), Base 12, Build 8-9, Peak 1-2, Race 1 — 24 to 28 weeks for a
macrocycle. After an A race the athlete returns to Base; the bias is to _make it on the
side of too much Base_ [3].

### Mesocycles and recovery weeks [2]

Training is organised in 4-week mesocycles that end in a recovery week. The recovery
content is day-level: 2-3 short Z1 days plus 2-3 days of short intensity openers; there is
no fixed percentage cut, the load drop follows naturally.

### Load progression: the CTL ramp [3]

The driver is weekly load (TSS), read from the hub's CTL, never recomputed. A sustained
progression adds 5-8 CTL points per week; above that you are entering the crash range,
which Friel reserves for rare, short use ("keep such a high ramp rate to a week or
less"). Build raises CTL in small steps (0-4 per week). Race week must not lose more
than about 10% of CTL, and the peak aims at a positive TSB on race day.

### Peak and taper [4]

The Peak is one to two weeks of race-like sessions every 72 hours with Z1-only recovery.
An A race gets a multi-week taper through the Peak; a B race gets a 2-4 day mini-taper; a C
race is trained through.

### Missed workouts (interruptions) [5]

- **1-3 days**: return as if nothing happened; never make up the missed load.
- **4-7 days**: not due to illness, the lost time is a rest week; reduce the current block
  by one week (or Peak 2 → 1) to resynchronize. Due to illness, treat it as more than
  seven days missed (next bracket).
- **8-14 days**: step back (Build → Base 3; Base → Base 1, or Prep when illness), then
  repeat the last hard week until it goes well; omission priority: Build 2, then the first
  week of Peak.
- **More than 14 days**: from Build, return to Base 3; from Base, back up one block;
  omission priority: first week of Peak, then Build 2, then Build 1. Nothing changes when
  the interruption falls in the last week of Build 2 or in the Peak.
- Illness always resumes in zones 1-2 until heart rate and perceived effort are back to
  normal; the >2-week bracket has no illness variant, and an injury follows the same
  duration brackets — its specificity is modality and tolerance, handled in the sessions,
  not in the block.

### Time-crunched floors and short blocks [6]

When the available time is short, the Base has a floor of 8 weeks and the Build a floor of
4 weeks (one mesocycle). Below the floors the Base becomes hybrid and instructs early Z3 /
sweet-spot work (muscular endurance) from the first week, because linear volume
accumulation is no longer possible.

### Closely spaced A races [7]

There is no formula. A second A race less than 3 weeks after the first forms one block
with a prolonged Peak covering both; at 4 weeks or more the athlete returns to Base after
the first race and builds a new block — in v1 the block is detailed up to the anchored
race only, and the second block is left to a later turn instead of being hallucinated
months ahead.

## Design decisions

**Pure, race-anchored derivation.** The block ends on the anchor race week and starts at
the later of the first executed week and 24 training weeks before the race, so a week's
label is a function of its distance to the race: the same anchor, grid and history derive
the same weeks, and weeks once past are never re-labelled by later recomputations. Only
`today` positions the current week and judges execution. Late hub revisions may still
shift past rows; that is accepted.

**Anchor chain.** The anchor is the nearest A-priority race; with no A race, the nearest
race by A → B → C priority. A second A race within 3 weeks extends the block (prolonged
Peak, both races named in their week); a second A race 4 or more weeks out stays out of
the detailed block.

**Monday grid aligned to the hub.** Week starts follow the cadence of
`training_rollup[].week_start` (Monday in practice), with a Monday fallback when the
rollup is empty, so #66 joins plan rows and executed weeks on the same buckets.

**Compression cascade 24 → 12 → 8 → 4.** Below 24 available training weeks the Prep is
dropped; the Base shrinks toward its floor of 8; then the Build toward its floor of 4
(protecting the Base floor); then the Peak toward 1. Below the floors the Base turns
hybrid and shrinks, and the Build follows, keeping one Peak week for as long as a single
training week remains. The hybrid instruction travels in the phase role, not in prose the
model must invent.

**Weekly bands read from the hub.** Maintenance is `7 × CTL` from the latest rollup
fitness; the ramp target is added as `42 × ramp` (6 × 7: weekly TSS per CTL point per
week under the 42-day CTL constant): Prep and Build +0..+4, Base +5..+8, recovery tails
−5..−2, Peak −4..0 (the last Peak week −6..−3), race week floored at the load that keeps
the CTL loss within 10% under the same constant (about 2.8 × CTL), capped at
maintenance. The crash week (+8..+10) exists only for an athlete who declares an elite
profile, and at most once; an absolute CTL threshold was rejected because Friel
explicitly declines one-size CTL targets ("it depends" on the sport and the individual),
so no metric opens that gate by itself. Bands are rounded to 5 TSS. The metric is load,
not hours. The declared state (elite, illness) is resolved with the model from the
athlete's active facts and announced in the fetch line (`state: illness, elite`).

**Recovery tails are mesocycle weeks.** Every fourth week of Base and Build is a
recovery week, and a phase whose length is not a multiple of four also closes its last
week as a recovery week when that closing mesocycle has at least two weeks — a one-week
remainder joins the previous mesocycle, so recovery weeks are never back to back. The
tail's role carries the day-level content; recovery is never encoded as a fixed
percentage.

**Nine week statuses.** `done`, `under`, `over` (executed load against the band),
`current` and `upcoming` (position), and `rest`, `repeat`, `step_back`, `omitted`
(interruption adjustments). Races are never re-classified: when a goal is compromised the
coach says so, and no race mutation follows.

**Contingency adjusts statuses, roles and bands — not the layout.** The race-anchored
layout never moves; an interruption marks the stopped weeks `rest`, the omitted weeks
(zero band) at their priority, the resumption week `step_back` with its re-entry role
(Base 3 from Build, Base 1 from Base, Prep when illness or after more than two weeks), and
on an 8-14 day interruption the following week `repeat`. In the Peak the stopped week
itself absorbs the reduction (Peak 2 → 1) and the race-eve taper week is never omitted.
The interruption length is estimated from the weekly rollup: elapsed days in the current
week plus 7 per fully empty week before it (a week with sessions but no load counts as
trained), so a full week off is the rest week and a resumed week counts the weeks it
skipped. Interruption statuses override the execution statuses.

**In-block races are week roles.** A B race marks its week (mini-taper 2-4 days), a C race
is trained through, and both stay judged by execution once the week has passed.

**PLAN_HORIZON_DAYS = 224.** One constant drives the race fetch and the weekly rollup
fetch (and the injected rollup keeps the full window; context eviction bounds the
payload), so the block and the interruption detection see the whole horizon instead of a
90-day slice.

**The section is a compact Markdown table.** One `<block_plan>` block, one row per week
(Week | Phase | Role | Target | Status), cells single-lined and pipe-escaped so untrusted
race names cannot break or forge rows; no prose around it.

## Sources

1. Joe Friel, _K.I.S.S. Periodization_, joefrieltraining.com — period lengths.
2. Joe Friel, _Recovery Week Design_ and _Aging: Matching the Mesocycle to Your
   Recovery_, joefrieltraining.com — 4-week mesocycles with a recovery tail, day-level
   recovery content.
3. Joe Friel, _The CTL Ramp Rate_, _Case study: periodization for a first A race_ and
   _Lessons from the 2010 season_, joefrieltraining.com — 5-8 CTL/week sustained, >10
   crash, race week ≤ 10% CTL loss, positive race-day TSB, return to Base after an A race.
4. Joe Friel, _Peak Block Recovery Workouts_, joefrieltraining.com — Peak of two weeks,
   race-like every 72 h, Z1-only recovery.
5. Joe Friel, _Missed Workouts_, joefrieltraining.com — the interruption brackets and
   omissions.
6. Joe Friel, _The Cyclist's Training Bible_ (Late Start / Time-Crunched) — Base floor of
   8 weeks, Build floor of 4 weeks, hybrid Base with early Z3/sweet spot, B race mini-taper,
   C race trained through.
7. Joe Friel, _Your Next A-Priority Race_, joefrieltraining.com — no formula for closely
   spaced A races; a long race is more than about four hours.
