# Open Endurance Coach

A self-hosted, AI-driven endurance coaching system.

Open Endurance Coach integrates multi-sport telemetry with Large Language Models to dynamically evaluate training execution and adjust future workouts. It operates as an automated loop via webhooks while maintaining a conversational terminal interface for deep historical analysis and subjective feedback.

## Core Features

- **Automated & On-Demand Analysis:** Triggers instantly upon activity completion or daily wellness updates via Intervals.icu webhooks, while also supporting manual terminal queries for historical trend extraction.
- **Single Source of Truth:** Centralizes all multi-sport data (Garmin Connect cycling, Health Sync running/swimming/wellness) strictly through the Intervals.icu API.
- **Interactive AI Loop:** Utilizes the DeepSeek API to validate executed targets against planned targets, pause for qualitative user feedback (e.g., fatigue, RPE, constraints), and calculate subsequent training loads.
- **Calendar Automation:** Autonomously executes structural modifications to future training blocks, pushing updated workouts directly to the Intervals.icu calendar.

## Coaching Methodology

The system is engineered to act as an elite endurance coach enforcing Joe Friel’s periodization principles and Dr. Andrew Coggan’s power analytics. It prioritizes objective execution validation and autonomous fatigue modulation over generic encouragement.

## Initial Setup Requirements

To operate this system, the following external credentials are required:

1. **Intervals.icu API Access:** An Athlete ID and Developer API Key (requires HTTP Basic Authentication).
2. **DeepSeek API Key:** For LLM inference and JSON schema generation.
