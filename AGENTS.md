# AGENTS.md

## 1. Project Overview & Goal

- **Project Name:** Open Endurance Coach
- **Goal:** A personalized AI endurance coach. It ingests physiological and telemetry data from Intervals.icu and analyzes execution against planned targets via an LLM. Manual-first: the `coach` chat is the only interface; every calendar change is a draft that needs a literal `yes` before it is written.
- **Data Hub:** Intervals.icu is the exclusive source of truth for all reads and writes.

## 2. Core Functional Principles

- **Event-Driven & On-Demand:** Manual-first: the terminal/chat is the daily interface; the activity webhook adapter is on the roadmap. There is no wellness webhook — readiness is pulled inside each analysis. Deep historical queries are on-demand (the referenced window is pinned so it survives the token budget).
- **LLM-Driven Logic:** The LLM parses data, enforces coaching methodologies (Friel periodization, Coggan power analytics), and structures calendar outputs. The LLM layer is provider-agnostic: OVHcloud's anonymous free tier is the default, DeepSeek optional.
- **Approval-Gated Writes:** The system never writes autonomously. Analysis produces a validated draft; calendar changes (workouts and races) are applied only after an explicit operator approval.
- **Architecture Agnostic:** The codebase must remain modular. Core functions (API extraction, LLM prompt building, output parsing) must be decoupled so the interface can easily evolve from a terminal CLI to a web UI in the future.

## 3. Definition of Done (DoD)

A feature is considered complete only when:

1. The AI successfully extracts the necessary data scope from Intervals.icu without exhausting token limits.
2. The LLM output successfully parses into a strict, validated schema (e.g., JSON) before any write operation is attempted.
3. Automated webhook responses process within required timeout limits.
4. User context (if provided) is verifiably injected into the LLM decision loop before the calendar is updated.
5. All new data flows and API endpoint usage are documented.
