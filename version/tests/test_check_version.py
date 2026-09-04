import io
import json
from pathlib import Path
import shutil
import subprocess
import sys


VERSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_DIR))

import check_version


def state(mode="mutable", release_tag=None):
    return {
        "schema_version": 1,
        "mode": mode,
        "versions": [{
            "version": "0.1.0",
            "release_tag": release_tag,
            "based_on": None,
            "changes": ["Add version checks."],
            "known_limits": [],
        }],
    }


def write_state(root, document):
    directory = root / "version"
    directory.mkdir(exist_ok=True)
    (directory / "state.json").write_text(json.dumps(document))


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def frozen_repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Version Test")
    git(tmp_path, "config", "user.email", "version@example.invalid")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "engine.py").write_text("VALUE = 1\n")
    (tmp_path / "version").mkdir()
    shutil.copy(VERSION_DIR / "check_version.py", tmp_path / "version" / "check_version.py")
    shutil.copy(VERSION_DIR / "README.md", tmp_path / "version" / "README.md")
    write_state(tmp_path, state())
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Version candidate")
    git(tmp_path, "tag", "v0.1.0-rc1")
    write_state(tmp_path, state("frozen", "v0.1.0-rc1"))
    git(tmp_path, "add", "version/state.json")
    git(tmp_path, "commit", "-m", "Activate frozen version")


def test_mutable_version_does_not_require_a_tag(tmp_path):
    write_state(tmp_path, state())

    context, errors, warnings = check_version.verify(tmp_path)

    assert errors == []
    assert warnings
    assert context == {"version": "0.1.0", "mode": "mutable", "release_tag": None}


def test_state_and_version_history_are_validated(tmp_path):
    write_state(tmp_path, [])
    assert check_version.verify(tmp_path)[1] == ["version/state.json must be a JSON object"]

    document = state()
    document["versions"].insert(0, {
        "version": "0.1.1",
        "release_tag": "v0.1.1-rc1",
        "based_on": "v0.0.9",
        "changes": ["Second version."],
        "known_limits": [],
    })
    write_state(tmp_path, document)

    errors = "; ".join(check_version.verify(tmp_path)[1])
    assert "based_on must name the next version tag" in errors
    assert "versions[1] requires a release_tag" in errors


def test_frozen_version_matches_its_release_tag(tmp_path):
    frozen_repo(tmp_path)

    context, errors, warnings = check_version.verify(tmp_path)

    assert errors == []
    assert warnings == []
    assert context["mode"] == "frozen"


def test_opening_a_frozen_version_keeps_its_tag_as_the_mutable_baseline(tmp_path):
    frozen_repo(tmp_path)
    write_state(tmp_path, state("mutable", "v0.1.0-rc1"))
    (tmp_path / "tools" / "engine.py").write_text("VALUE = 2\n")

    context, errors, warnings = check_version.verify(tmp_path)

    assert errors == []
    assert warnings
    assert context == {
        "version": "0.1.0",
        "mode": "mutable",
        "release_tag": "v0.1.0-rc1",
    }


def test_frozen_version_detects_tracked_drift(tmp_path):
    frozen_repo(tmp_path)
    (tmp_path / "tools" / "engine.py").write_text("VALUE = 2\n")

    errors = check_version.verify(tmp_path)[1]

    assert "protected files differ from the release tag" in errors


def test_frozen_version_detects_untracked_protected_files(tmp_path):
    frozen_repo(tmp_path)
    (tmp_path / "tools" / "new_engine.py").write_text("VALUE = 2\n")

    errors = "; ".join(check_version.verify(tmp_path)[1])

    assert "untracked protected files are present" in errors
    assert "tools/new_engine.py" in errors


def test_gate_allows_only_explicit_rehearsals_while_mutable(tmp_path, monkeypatch):
    write_state(tmp_path, state())
    production = {
        "tool_name": "spawn_agent",
        "tool_input": {"agent_type": "mar-analyst", "message": "Run the case"},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(production)))
    assert check_version.gate(tmp_path) == 2

    production["tool_input"]["message"] += "\nVERSION_MODE: rehearsal"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(production)))
    assert check_version.gate(tmp_path) == 0


def test_gate_ignores_unrelated_agents(tmp_path, monkeypatch):
    write_state(tmp_path, state())
    hook = {
        "tool_name": "spawn_agent",
        "tool_input": {"task_name": "general-purpose", "message": "Inspect docs"},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook)))

    assert check_version.gate(tmp_path) == 0
