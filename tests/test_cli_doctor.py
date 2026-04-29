import json
from pathlib import Path

from click.testing import CliRunner

from c3ae.cli import main


def test_doctor_reports_openclaw_installation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    xdg_data_home = tmp_path / "xdg-data"
    data_dir = tmp_path / "novaspine-data"
    install_root = xdg_data_home / "novaspine" / "openclaw"
    config_path = home / ".openclaw" / "openclaw.json"

    for path in [
        install_root / "packages" / "openclaw-memory-plugin",
        install_root / "packages" / "openclaw-context-engine",
        install_root / "packages" / "openclaw-consciousness",
        install_root / "scripts",
        config_path.parent,
        data_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (install_root / "scripts" / "run-memory-maintenance.sh").write_text("#!/usr/bin/env bash\n")
    (install_root / "scripts" / "run-consciousness-suite.sh").write_text("#!/usr/bin/env bash\n")
    for package_name, plugin_id in (
        ("openclaw-memory-plugin", "novaspine-memory"),
        ("openclaw-context-engine", "novaspine-context"),
        ("openclaw-consciousness", "nova-consciousness"),
    ):
        (install_root / "packages" / package_name / "openclaw.plugin.json").write_text(
            json.dumps({"id": plugin_id})
        )

    config_path.write_text(json.dumps({
        "plugins": {
            "allow": [
                "novaspine-memory",
                "novaspine-context",
                "nova-consciousness",
            ],
            "load": {
                "paths": [
                    str(install_root / "packages" / "openclaw-memory-plugin"),
                    str(install_root / "packages" / "openclaw-context-engine"),
                    str(install_root / "packages" / "openclaw-consciousness"),
                ]
            },
            "slots": {
                "memory": "novaspine-memory",
                "contextEngine": "novaspine-context",
            },
            "entries": {
                "novaspine-memory": {
                    "enabled": True,
                    "config": {"baseUrl": "http://127.0.0.1:8420"},
                },
                "novaspine-context": {
                    "enabled": True,
                    "config": {"baseUrl": "http://127.0.0.1:8420"},
                },
                "nova-consciousness": {
                    "enabled": True,
                    "config": {"baseUrl": "http://127.0.0.1:4111"},
                },
            },
        }
    }, indent=2))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--data-dir", str(data_dir), "doctor", "--skip-api-check", "--json-output"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["ok"] >= 3
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["openclaw-config"]["level"] == "ok"
    assert checks["openclaw-install-root"]["level"] == "ok"


def test_doctor_handles_empty_openclaw_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    xdg_data_home = tmp_path / "xdg-data"
    data_dir = tmp_path / "novaspine-data"
    install_root = xdg_data_home / "novaspine" / "openclaw"
    config_path = home / ".openclaw" / "openclaw.json"

    for path in [
        install_root / "packages" / "openclaw-memory-plugin",
        install_root / "packages" / "openclaw-context-engine",
        install_root / "packages" / "openclaw-consciousness",
        install_root / "scripts",
        config_path.parent,
        data_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (install_root / "scripts" / "run-memory-maintenance.sh").write_text("#!/usr/bin/env bash\n")
    (install_root / "scripts" / "run-consciousness-suite.sh").write_text("#!/usr/bin/env bash\n")
    for package_name, plugin_id in (
        ("openclaw-memory-plugin", "novaspine-memory"),
        ("openclaw-context-engine", "novaspine-context"),
        ("openclaw-consciousness", "nova-consciousness"),
    ):
        (install_root / "packages" / package_name / "openclaw.plugin.json").write_text(
            json.dumps({"id": plugin_id})
        )
    config_path.write_text("{}")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--data-dir", str(data_dir), "doctor", "--skip-api-check", "--json-output"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["openclaw-config"]["level"] == "fail"
    assert "allow missing" in checks["openclaw-config"]["detail"]


def test_doctor_accepts_existing_plugin_paths_outside_install_root(tmp_path, monkeypatch):
    home = tmp_path / "home"
    xdg_data_home = tmp_path / "xdg-data"
    data_dir = tmp_path / "novaspine-data"
    install_root = xdg_data_home / "novaspine" / "openclaw"
    config_path = home / ".openclaw" / "openclaw.json"
    actual_root = tmp_path / "repo"
    for path in [
        install_root / "packages" / "openclaw-memory-plugin",
        install_root / "packages" / "openclaw-context-engine",
        install_root / "packages" / "openclaw-consciousness",
        install_root / "scripts",
        actual_root / "packages" / "openclaw-memory-plugin",
        actual_root / "packages" / "openclaw-context-engine",
        actual_root / "packages" / "openclaw-consciousness",
        config_path.parent,
        data_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (install_root / "scripts" / "run-memory-maintenance.sh").write_text("#!/usr/bin/env bash\n")
    (install_root / "scripts" / "run-consciousness-suite.sh").write_text("#!/usr/bin/env bash\n")
    for package_name, plugin_id in (
        ("openclaw-memory-plugin", "novaspine-memory"),
        ("openclaw-context-engine", "novaspine-context"),
        ("openclaw-consciousness", "nova-consciousness"),
    ):
        (actual_root / "packages" / package_name / "openclaw.plugin.json").write_text(
            json.dumps({"id": plugin_id})
        )

    config_path.write_text(json.dumps({
        "plugins": {
            "allow": ["novaspine-memory", "novaspine-context", "nova-consciousness"],
            "load": {
                "paths": [
                    str(actual_root / "packages" / "openclaw-memory-plugin"),
                    str(actual_root / "packages" / "openclaw-context-engine"),
                    str(actual_root / "packages" / "openclaw-consciousness"),
                ]
            },
            "slots": {"memory": "novaspine-memory", "contextEngine": "novaspine-context"},
            "entries": {
                "novaspine-memory": {"enabled": True, "config": {"baseUrl": "http://127.0.0.1:8420"}},
                "novaspine-context": {"enabled": True, "config": {"baseUrl": "http://127.0.0.1:8420"}},
                "nova-consciousness": {"enabled": True, "config": {"baseUrl": "http://127.0.0.1:4111"}},
            },
        }
    }))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))

    result = CliRunner().invoke(
        main,
        ["--data-dir", str(data_dir), "doctor", "--skip-api-check", "--json-output"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["openclaw-config"]["level"] == "ok"
