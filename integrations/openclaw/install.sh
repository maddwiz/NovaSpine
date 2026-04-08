#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL_ROOT="${NOVASPINE_OPENCLAW_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/novaspine/openclaw}"
OPENCLAW_CONFIG="${OPENCLAW_CONFIG_PATH:-$HOME/.openclaw/openclaw.json}"
BASE_URL="${NOVASPINE_BASE_URL:-http://127.0.0.1:8420}"
CONSCIOUSNESS_BASE_URL="${NOVASPINE_CONSCIOUSNESS_BASE_URL:-http://127.0.0.1:4111}"
SKIP_PLUGIN_INSTALL=0
SKIP_CONFIG_PATCH=0
FORCE_SLOTS=0

usage() {
  cat <<'EOF'
Usage: integrations/openclaw/install.sh [options]

Options:
  --install-root PATH            Stable install root for NovaSpine OpenClaw assets
  --config PATH                  OpenClaw config file to patch
  --base-url URL                 NovaSpine API base URL
  --consciousness-base-url URL   Consciousness suite base URL
  --skip-plugin-install          Do not run `openclaw plugins install`
  --skip-config-patch            Copy assets only; do not patch openclaw.json
  --force-slots                  Replace existing memory/contextEngine slot bindings
  --help                         Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-root)
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --config)
      OPENCLAW_CONFIG="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --consciousness-base-url)
      CONSCIOUSNESS_BASE_URL="$2"
      shift 2
      ;;
    --skip-plugin-install)
      SKIP_PLUGIN_INSTALL=1
      shift
      ;;
    --skip-config-patch)
      SKIP_CONFIG_PATCH=1
      shift
      ;;
    --force-slots)
      FORCE_SLOTS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$INSTALL_ROOT/packages" "$INSTALL_ROOT/scripts"

rsync -a "$REPO_ROOT/packages/openclaw-memory-plugin/" "$INSTALL_ROOT/packages/openclaw-memory-plugin/"
rsync -a "$REPO_ROOT/packages/openclaw-context-engine/" "$INSTALL_ROOT/packages/openclaw-context-engine/"
rsync -a "$REPO_ROOT/packages/openclaw-consciousness/" "$INSTALL_ROOT/packages/openclaw-consciousness/"
rsync -a "$REPO_ROOT/integrations/openclaw/scripts/" "$INSTALL_ROOT/scripts/"

cat >"$INSTALL_ROOT/openclaw.plugins.example.json" <<EOF
{
  "plugins": {
    "allow": [
      "novaspine-memory",
      "novaspine-context",
      "nova-consciousness"
    ],
    "load": {
      "paths": [
        "$INSTALL_ROOT/packages/openclaw-memory-plugin",
        "$INSTALL_ROOT/packages/openclaw-context-engine",
        "$INSTALL_ROOT/packages/openclaw-consciousness"
      ]
    },
    "slots": {
      "memory": "novaspine-memory",
      "contextEngine": "novaspine-context"
    },
    "entries": {
      "novaspine-memory": {
        "enabled": true,
        "config": {
          "baseUrl": "$BASE_URL"
        }
      },
      "novaspine-context": {
        "enabled": true,
        "config": {
          "baseUrl": "$BASE_URL"
        }
      },
      "nova-consciousness": {
        "enabled": true,
        "config": {
          "baseUrl": "$CONSCIOUSNESS_BASE_URL"
        }
      }
    }
  }
}
EOF

if [[ "$SKIP_PLUGIN_INSTALL" -eq 0 ]] && command -v openclaw >/dev/null 2>&1; then
  for plugin_dir in \
    "$INSTALL_ROOT/packages/openclaw-memory-plugin" \
    "$INSTALL_ROOT/packages/openclaw-context-engine" \
    "$INSTALL_ROOT/packages/openclaw-consciousness"
  do
    if ! openclaw plugins install "$plugin_dir" >/dev/null 2>&1; then
      echo "warning: openclaw plugin install did not complete cleanly for $plugin_dir" >&2
    fi
  done
fi

if [[ "$SKIP_CONFIG_PATCH" -eq 0 ]] && [[ -f "$OPENCLAW_CONFIG" ]]; then
  PATCH_ARGS=(
    "$REPO_ROOT/integrations/openclaw/patch-openclaw-config.py"
    --config "$OPENCLAW_CONFIG"
    --install-root "$INSTALL_ROOT"
    --base-url "$BASE_URL"
    --consciousness-base-url "$CONSCIOUSNESS_BASE_URL"
  )
  if [[ "$FORCE_SLOTS" -eq 1 ]]; then
    PATCH_ARGS+=(--force-slots)
  fi
  python3 "${PATCH_ARGS[@]}"
fi

cat <<EOF
NovaSpine OpenClaw integration installed.

Install root: $INSTALL_ROOT
Example config: $INSTALL_ROOT/openclaw.plugins.example.json
OpenClaw config: $OPENCLAW_CONFIG

Next:
  1. Ensure the NovaSpine API is running at $BASE_URL
  2. Ensure the consciousness suite is running at $CONSCIOUSNESS_BASE_URL if you want continuity features
  3. Run \`openclaw config validate\`
EOF
