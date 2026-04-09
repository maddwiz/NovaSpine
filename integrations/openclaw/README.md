# NovaSpine OpenClaw Integration Layer

This directory carries the reusable OpenClaw-facing pieces that belong in the
promoted `NovaSpine` repo without dragging in one machine's live runtime.

Included here:

- `../../packages/openclaw-memory-plugin/`
- `../../packages/openclaw-context-engine/`
- `scripts/novaspine-sync-action-memory.py`
- `scripts/novaspine-prune-noise.py`
- `scripts/novaspine-promote-fixes.py`
- `scripts/novaspine-writeback-lessons.py`
- `scripts/novaspine-dream-report.py`
- `scripts/run-memory-maintenance.sh`
- `scripts/run-consciousness-suite.sh`
- `scripts/install.sh`
- `patch-openclaw-config.py`
- `../../packages/openclaw-consciousness/`

Deliberately excluded:

- The Lab room bridges and Discord relay topology
- systemd or launchd unit files tied to one host
- tokens, profile roots, and live session stores
- the old Nemo-sidecar-specific wrapper/bootstrap layer

## Configuration Model

The scripts here are parameterized through environment variables instead of
hard-coded profile paths. The main knobs are:

- `NOVASPINE_REPO`
- `NOVASPINE_PYTHON`
- `C3AE_DATA_DIR`
- `NOVASPINE_OPENCLAW_DIR`
- `NOVASPINE_EXTRA_WATCH_DIRS`
- `NOVASPINE_PROFILE`
- `NOVASPINE_AGENT_NAME`
- `NOVASPINE_AGENT_HANDLES`
- `NOVASPINE_REPLY_MARKERS`

Optional hooks:

- `NOVASPINE_SEED_SCRIPT`
- `NOVASPINE_GUARDRAILS_SCRIPT`
- `NOVASPINE_CONSCIOUSNESS_REPO`
- `NOVASPINE_CONSCIOUSNESS_PORT`

## What This Means

`NovaSpine` is now the umbrella repo for the reusable memory and cognition
platform, while host-specific runtime wiring stays outside the repo.

## Turnkey Install

From the repo root:

```bash
./integrations/openclaw/install.sh
```

That will:

- copy the reusable NovaSpine OpenClaw assets into a stable install root
- install the three OpenClaw plugins when `openclaw` is available
- patch `~/.openclaw/openclaw.json` safely and idempotently when it exists
- write an example config snippet under the install root
