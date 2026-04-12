import importlib.util
import json
from pathlib import Path


def _load_patcher_module():
    patcher_path = (
        Path(__file__).resolve().parents[1]
        / "integrations"
        / "openclaw"
        / "patch-openclaw-config.py"
    )
    spec = importlib.util.spec_from_file_location("patch_openclaw_config", patcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_openclaw_config_backup_preserves_pre_patch_state(tmp_path, monkeypatch):
    module = _load_patcher_module()
    config_path = tmp_path / "openclaw.json"
    install_root = tmp_path / "install-root"
    install_root.mkdir()

    original_text = '{\n  "plugins": {\n    "allow": []\n  }\n}\n'
    config_path.write_text(original_text)

    monkeypatch.setattr(
        "sys.argv",
        [
            "patch-openclaw-config.py",
            "--config",
            str(config_path),
            "--install-root",
            str(install_root),
        ],
    )

    assert module.main() == 0

    backups = sorted(config_path.parent.glob("openclaw.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original_text

    live_data = json.loads(config_path.read_text())
    assert "novaspine-memory" in live_data["plugins"]["allow"]
    active_memory = live_data["plugins"]["entries"]["novaspine-memory"]["config"]["activeMemory"]
    assert active_memory["enabled"] is True
    assert active_memory["agents"] == ["main"]
    assert active_memory["allowedChatTypes"] == ["direct", "group"]
    assert active_memory["queryMode"] == "recent"
    assert active_memory["promptStyle"] == "balanced"
    assert config_path.read_text() != original_text


def test_patch_openclaw_config_prefers_common_conversational_agents(tmp_path, monkeypatch):
    module = _load_patcher_module()
    config_path = tmp_path / "openclaw.json"
    install_root = tmp_path / "install-root"
    install_root.mkdir()

    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "list": [
                        {"id": "main"},
                        {"id": "main-lab"},
                        {"id": "discord-private"},
                        {"id": "memory-deep"},
                    ]
                },
                "plugins": {"allow": []},
            },
            indent=2,
        )
        + "\n"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "patch-openclaw-config.py",
            "--config",
            str(config_path),
            "--install-root",
            str(install_root),
        ],
    )

    assert module.main() == 0

    live_data = json.loads(config_path.read_text())
    active_memory = live_data["plugins"]["entries"]["novaspine-memory"]["config"]["activeMemory"]
    assert active_memory["agents"] == ["main", "main-lab", "discord-private"]
    assert active_memory["allowedChatTypes"] == ["direct", "group"]


def test_patch_openclaw_config_noop_does_not_create_extra_backup(tmp_path, monkeypatch, capsys):
    module = _load_patcher_module()
    config_path = tmp_path / "openclaw.json"
    install_root = tmp_path / "install-root"
    install_root.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "patch-openclaw-config.py",
            "--config",
            str(config_path),
            "--install-root",
            str(install_root),
        ],
    )

    config_path.write_text('{\n  "plugins": {\n    "allow": []\n  }\n}\n')
    assert module.main() == 0
    first_text = config_path.read_text()
    first_backups = sorted(config_path.parent.glob("openclaw.json.bak.*"))
    assert len(first_backups) == 1

    assert module.main() == 0
    second_text = config_path.read_text()
    second_backups = sorted(config_path.parent.glob("openclaw.json.bak.*"))
    assert second_text == first_text
    assert len(second_backups) == 1

    captured = capsys.readouterr()
    assert '"status": "unchanged"' in captured.out
