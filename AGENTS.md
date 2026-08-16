# AGENTS.md

## 1. Project Overview & Goal

- **Project Name:** Open Endurance Coach
- **Goal:** An automated, event-driven backend system that functions as a personalized AI endurance coach. It ingests physiological and telemetry data from Intervals.icu, analyzes execution against planned targets via an LLM, and dynamically modifies future training calendars.
- **Data Hub:** Intervals.icu is the exclusive source of truth for all reads and writes.

## 2. Core Functional Principles

- **Event-Driven & On-Demand:** The system operates autonomously via webhooks for daily training/wellness updates but must also support manual, ad-hoc terminal queries for deep historical analysis.
- **LLM-Driven Logic:** The system relies on the DeepSeek API to parse data, enforce coaching methodologies, and structure the calendar outputs.
- **Architecture Agnostic:** The codebase must remain modular. Core functions (API extraction, LLM prompt building, output parsing) must be decoupled so the interface can easily evolve from a terminal CLI to a web UI in the future.

## 3. Definition of Done (DoD)

A feature is considered complete only when:

1. The AI successfully extracts the necessary data scope from Intervals.icu without exhausting token limits.
2. The LLM output successfully parses into a strict, validated schema (e.g., JSON) before any write operation is attempted.
3. Automated webhook responses process within required timeout limits.
4. User context (if provided) is verifiably injected into the LLM decision loop before the calendar is updated.
5. All new data flows and API endpoint usage are documented.
