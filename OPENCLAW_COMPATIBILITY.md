# NovaSpine x OpenClaw Compatibility Matrix

This file exists to answer one practical question:

If I install NovaSpine into OpenClaw, what still works when OpenClaw updates, and what may need a NovaSpine-side update?

## What A Compatibility Matrix Means

A compatibility matrix is a small support table that tells users:

- which OpenClaw versions NovaSpine has been tested with
- whether the install/repair flow is known to work
- which upstream memory features work automatically
- which upstream memory features are `memory-core`-specific and therefore need a NovaSpine port or adapter first

Without a matrix, users often assume one of two wrong things:

- "OpenClaw upgrades will break everything"
- "Every new OpenClaw memory feature will automatically appear inside NovaSpine"

The truth is between those two.

## Tested OpenClaw Versions

| OpenClaw | NovaSpine status | Notes |
|---|---|---|
| `2026.4.5` | Supported | Verified installer/doctor flow |
| `2026.4.7` | Supported | Verified installer/doctor flow |
| `2026.4.9` | Supported | Verified in live NovaSpine-backed OpenClaw deployments |
| `2026.4.10` | Supported | Verified in live NovaSpine-backed OpenClaw deployments; NovaSpine plugin carries Active Memory support |
| `2026.4.11` | Supported | OpenClaw stable on 2026-04-12; stock `memory-core` Dreaming import/UI additions are not auto-ported when NovaSpine owns the memory slot |
| `2026.4.12` | Supported | OpenClaw stable on 2026-04-13; NovaSpine carries the directly relevant memory-side fixes for unicode wiki slugs and nested daily-note recall |

## Upgrade Behavior Matrix

| OpenClaw surface | After an OpenClaw upgrade | What it means for NovaSpine users |
|---|---|---|
| Gateway/runtime/chat/tool plumbing | Usually still benefits from upstream updates | NovaSpine does not block normal OpenClaw runtime improvements |
| Generic plugin/config handling | Usually still benefits | NovaSpine depends on these public seams staying compatible |
| Selected memory-slot behavior | Can benefit if OpenClaw honors the active memory slot | This is the best case for NovaSpine-backed installs |
| OpenClaw Active Memory | Can benefit when ported through `novaspine-memory` | Stock `active-memory` can stay disabled while NovaSpine remains the source of truth |
| NovaSpine-native dream/wiki adapters | Supported when shipped in the NovaSpine integration layer | These let NovaSpine users keep dream diary/status and compiled wiki views without switching back to `memory-core` |
| `memory-core` internal UI/workflows | Not automatic | These are not guaranteed just because OpenClaw added them upstream |
| `memory-core`-specific dream/wiki workflows | Need NovaSpine porting or an adapter | NovaSpine is the active memory system, so those features do not appear by magic |
| NovaSpine memory recall/capture/context | Should continue working if wiring is intact | If an upgrade changes plugin/config wiring, rerun the installer and doctor flow |

## Practical Rule

When NovaSpine is the active memory slot, OpenClaw still upgrades normally, but NovaSpine becomes the source of truth for memory behavior.

That means:

- OpenClaw core improvements still matter
- OpenClaw memory-slot-aware improvements can still help
- `memory-core`-specific features may require NovaSpine work before they are available

## What Users Should Expect

### These usually continue to work after an OpenClaw upgrade

- `novaspine-memory` as the active memory slot
- `novaspine-context` as the active context engine
- NovaSpine capture / recall / augment
- NovaSpine-routed Active Memory summaries and session toggles
- NovaSpine-native dream diary / dream status surfaces
- NovaSpine-native wiki status / search / page / lint surfaces
- NovaSpine maintenance and consolidation
- `novaspine doctor`
- normal OpenClaw runtime improvements outside `memory-core` internals

### These may need NovaSpine-side work

- new `memory-core` dream UX
- new `memory-core` wiki UX
- `memory-core`-specific dashboards
- memory features that assume `memory-core` owns the durable knowledge layer

## Supported Repair Flow

If OpenClaw updates and the memory wiring drifts, the supported repair path is:

```bash
./scripts/install-openclaw.sh
novaspine doctor
openclaw config validate
```

## Why This Helps

This matrix helps users decide:

- whether a version is safe to try
- whether a missing upstream feature is a bug or just not ported yet
- whether they need a repair pass or a NovaSpine update

It also makes the NovaSpine/OpenClaw relationship explicit:

- OpenClaw remains the runtime shell
- NovaSpine remains the memory source of truth
- some upstream memory features must be adapted rather than assumed
