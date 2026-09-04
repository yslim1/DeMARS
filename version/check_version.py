#!/usr/bin/env python3
"""Verify the active DeMARS version without modifying the repository."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROTECTED_PATHS = (
    "demars-core",
    "tools",
    "episodes",
    ".codex/agents",
    ".agents/skills",
    ".codex/hooks.json",
    "AGENTS.md",
    "assets/demars.yaml.example",
    "assets/setup",
    "version/check_version.py",
    "version/README.md",
    "version/tests",
)
STRICT_UNTRACKED_PATHS = (".codex/agents", ".agents/skills")
CAMPAIGN_AGENTS = {"mar-analyst", "mar-reviewer"}
AGENT_TOOL_NAMES = ("Task", "Agent", "spawn_agent")
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _git(root, *args):
    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )


def _validate_versions(versions):
    errors = []
    if not isinstance(versions, list) or not versions:
        return ["versions must be a non-empty list"]

    numbers = []
    for index, item in enumerate(versions):
        if not isinstance(item, dict):
            errors.append(f"versions[{index}] must be a JSON object")
            continue

        version = item.get("version")
        if not isinstance(version, str) or not SEMVER.fullmatch(version):
            errors.append(f"versions[{index}].version must be semantic x.y.z")
            version = ""
        else:
            numbers.append(tuple(map(int, version.split("."))))

        tag = item.get("release_tag")
        if tag is not None and (
            not isinstance(tag, str)
            or not TAG.fullmatch(tag)
            or ".." in tag
            or not re.fullmatch(rf"v{re.escape(version)}(?:[-+].+)?", tag)
        ):
            errors.append(f"versions[{index}].release_tag is invalid")
        if index and tag is None:
            errors.append(f"versions[{index}] requires a release_tag")

        changes = item.get("changes")
        if not isinstance(changes, list) or not all(
            isinstance(change, str) and change for change in changes
        ):
            errors.append(f"versions[{index}].changes must be a list of non-empty strings")
        limits = item.get("known_limits")
        if not isinstance(limits, list) or not all(isinstance(limit, str) for limit in limits):
            errors.append(f"versions[{index}].known_limits must be a string list")

    if len(numbers) == len(versions):
        for current, previous in zip(numbers, numbers[1:]):
            if current <= previous:
                errors.append("versions must be unique and newest-first")
                break
    for index, item in enumerate(versions[:-1]):
        if isinstance(item, dict) and isinstance(versions[index + 1], dict):
            if item.get("based_on") != versions[index + 1].get("release_tag"):
                errors.append(f"versions[{index}].based_on must name the next version tag")
    if isinstance(versions[-1], dict) and versions[-1].get("based_on") is not None:
        errors.append("the oldest version must have based_on: null")
    return errors


def _verify_frozen(root, current):
    tag = current.get("release_tag")
    if not isinstance(tag, str) or not TAG.fullmatch(tag) or ".." in tag:
        return ["frozen mode requires a valid release_tag"]

    resolved = _git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if resolved.returncode:
        return [f"release tag does not exist: {tag}"]

    errors = []
    drift = _git(root, "diff", "--quiet", resolved.stdout.strip(), "--", *PROTECTED_PATHS)
    if drift.returncode == 1:
        errors.append("protected files differ from the release tag")
    elif drift.returncode:
        errors.append(f"Git could not compare protected files: {drift.stderr.strip()}")

    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *PROTECTED_PATHS)
    if untracked.returncode:
        errors.append("Git could not inspect untracked protected files")
    elif untracked.stdout.strip():
        errors.append("untracked protected files are present: " + ", ".join(untracked.stdout.split()))

    strict = _git(root, "ls-files", "--others", "--", *STRICT_UNTRACKED_PATHS)
    if strict.returncode:
        errors.append("Git could not inspect ignored agent or skill files")
    elif strict.stdout.strip():
        errors.append("untracked agent or skill files are present: " + ", ".join(strict.stdout.split()))
    return errors


def verify(root):
    state_path = root / "version" / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, RecursionError) as exc:
        return {}, [f"cannot load version/state.json: {exc}"], []
    if not isinstance(state, dict):
        return {}, ["version/state.json must be a JSON object"], []

    errors = []
    if type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        errors.append("unsupported version state schema_version")
    versions = state.get("versions")
    errors.extend(_validate_versions(versions))
    current = versions[0] if isinstance(versions, list) and versions and isinstance(versions[0], dict) else {}

    warnings = []
    mode = state.get("mode")
    if mode not in ("mutable", "frozen"):
        errors.append(f"invalid version mode: {mode!r}")
    elif mode == "frozen":
        errors.extend(_verify_frozen(root, current))
    else:
        warnings.append("version is mutable; production execution is not frozen")

    return {
        "version": current.get("version"),
        "mode": mode,
        "release_tag": current.get("release_tag"),
    }, errors, warnings


def _blocked(reason):
    print(f"DeMARS version gate blocked: {reason}", file=sys.stderr)
    print("Read version/README.md and follow the version workflow.", file=sys.stderr)
    return 2


def gate(root):
    try:
        hook = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError, RecursionError) as exc:
        return _blocked(f"malformed hook JSON: {exc}")
    if not isinstance(hook, dict) or hook.get("tool_name") not in AGENT_TOOL_NAMES:
        return 0
    tool_input = hook.get("tool_input")
    if not isinstance(tool_input, dict):
        return _blocked("tool_input must be a JSON object")
    agent = next(
        (tool_input.get(key) for key in ("subagent_type", "agent_type", "task_name")
         if tool_input.get(key) is not None),
        None,
    )
    if agent is not None and not isinstance(agent, str):
        return _blocked("agent identifier must be a string")
    if agent not in CAMPAIGN_AGENTS:
        return 0

    try:
        mode = json.loads((root / "version" / "state.json").read_text(encoding="utf-8"))["mode"]
    except (OSError, json.JSONDecodeError, RecursionError, KeyError, TypeError) as exc:
        return _blocked(f"cannot read version state: {exc}")

    prompt = tool_input.get("prompt", tool_input.get("message"))
    rehearsal = isinstance(prompt, str) and re.search(r"(?m)^VERSION_MODE: rehearsal$", prompt)
    if mode == "frozen" and not rehearsal:
        _, errors, _ = verify(root)
        return _blocked("frozen state check failed: " + "; ".join(errors[:3])) if errors else 0
    if mode == "mutable" and rehearsal:
        return 0
    if mode == "mutable":
        return _blocked(
            "production execution requires mode 'frozen'; a rehearsal requires the exact line "
            "VERSION_MODE: rehearsal"
        )
    if mode == "frozen":
        return _blocked("rehearsal mode is allowed only while the version is mutable")
    return _blocked(f"invalid version mode: {mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("session-start", "verify", "stop", "gate"), nargs="?", default="verify")
    args = parser.parse_args()
    root = Path(os.environ.get("CODEX_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

    if args.mode == "gate":
        try:
            return gate(root)
        except Exception as exc:
            return _blocked(f"gate failed unexpectedly: {type(exc).__name__}: {exc}")
    if args.mode == "stop":
        try:
            if json.load(sys.stdin).get("stop_hook_active"):
                return 0
        except (OSError, ValueError, AttributeError):
            pass

    try:
        context, errors, warnings = verify(root)
    except Exception as exc:
        context, warnings = {}, []
        errors = [f"version verification failed unexpectedly: {type(exc).__name__}: {exc}"]
    ok = not errors
    lines = [f"DeMARS version check: {'PASS' if ok else 'FAIL'}"]
    if context:
        lines.extend((
            f"Version: {context.get('version')}",
            f"Mode: {context.get('mode')}",
            f"Release tag: {context.get('release_tag') or 'none'}",
        ))
    lines.extend(f"Warning: {warning}" for warning in warnings)
    lines.extend(f"Error: {error}" for error in errors)

    if args.mode == "stop":
        if context.get("mode") == "mutable" or ok:
            return 0
        print("DeMARS frozen-state check: FAIL", file=sys.stderr)
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        print("Do not reset, restore, or auto-repair files. Report the violation.", file=sys.stderr)
        return 2
    if args.mode == "session-start":
        if ok:
            print(json.dumps({"systemMessage": (
                f"[DeMARS] {context['version']} | {context['mode'].upper()} | verification PASS"
            )}))
        else:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n".join(lines),
                },
                "systemMessage": "\n".join([lines[0]] + [line for line in lines if line.startswith("Error:")]),
            }))
        return 0
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
