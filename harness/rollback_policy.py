#!/usr/bin/env python3
"""Decision helpers for the variant runner: exact service restoration and Xid cursors.

Pure functions (unit-tested in test_harness.py) plus a small CLI used by run_variant.sh:

  snapshot     --services a,b --out state.json   record what was running + journal cursor
  restore-list --state state.json                services that must be running again
  xid-new      --state state.json                count of Xid events since the snapshot
  xid-lines    --state state.json                those events, one per line
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

XID_RE = re.compile(r"NVRM.*[Xx]id")
JOURNAL_CMD = ["sudo", "-n", "journalctl", "-k", "-b", "--no-pager"]


def services_to_restore(initial_active, candidates):
    """Services that were running before the variant must be running after it."""
    return [s for s in candidates if initial_active.get(s, False)]


def journal_line_count(journal_text):
    """Line-based cursor into an append-only boot journal."""
    return len(journal_text.splitlines())


def new_xid_count(cursor_lines, journal_text):
    """Xid lines strictly after the cursor; earlier events are history, not this run's failure."""
    lines = journal_text.splitlines()
    if cursor_lines < 0:
        cursor_lines = 0
    return sum(1 for line in lines[cursor_lines:] if XID_RE.search(line))


def new_xid_lines(cursor_lines, journal_text):
    lines = journal_text.splitlines()
    if cursor_lines < 0:
        cursor_lines = 0
    return [line for line in lines[cursor_lines:] if XID_RE.search(line)]


def parse_state(path):
    return json.loads(pathlib.Path(path).read_text())


def build_state(active_map, journal_text):
    return {"services": dict(active_map), "journal_lines": journal_line_count(journal_text)}


def _is_active(unit):
    try:
        out = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=15)
        return out.stdout.strip() == "active"
    except Exception:
        return False


def _journal():
    out = subprocess.run(JOURNAL_CMD, capture_output=True, text=True, timeout=120)
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("snapshot")
    p.add_argument("--services", required=True, help="comma-separated unit names")
    p.add_argument("--out", required=True)

    for name in ("restore-list", "xid-new", "xid-lines"):
        q = sub.add_parser(name)
        q.add_argument("--state", required=True)

    a = ap.parse_args()

    if a.cmd == "snapshot":
        services = [s for s in a.services.split(",") if s]
        active = {s: _is_active(s) for s in services}
        state = build_state(active, _journal())
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(state, ensure_ascii=False))
        return 0

    state = parse_state(a.state)
    if a.cmd == "restore-list":
        for s in services_to_restore(state.get("services", {}), sorted(state.get("services", {}))):
            print(s)
        return 0

    cursor = int(state.get("journal_lines", 0))
    journal = _journal()
    if a.cmd == "xid-new":
        print(new_xid_count(cursor, journal))
        return 0

    for line in new_xid_lines(cursor, journal):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
