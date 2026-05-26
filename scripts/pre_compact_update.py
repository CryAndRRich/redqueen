#!/usr/bin/env python3
"""
Pre-compact hook: appends a concise decision summary to CLAUDE.md Section 14
before Claude Code compacts the conversation context.

Invoked by .claude/settings.json PreCompact hook.
Receives a JSON payload on stdin from Claude Code.

Output format (compact, no JSON blobs):
  ### YYYY-MM-DD HH:MM — <topic>
  - key decision 1
  - key decision 2
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD    = PROJECT_ROOT / "CLAUDE.md"
BACKUP_PATH  = PROJECT_ROOT / "CLAUDE.md.bak"

SESSION_LOG_HEADER = "## 14. Session Update Log"
MAX_ENTRY_LINES    = 20   # cap per compact entry to keep Section 14 lean


def read_stdin_payload() -> dict:
    try:
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, Exception):
        return {}


def extract_summary(payload: dict) -> str:
    for key in ("summary", "content", "text", "message"):
        val = payload.get(key, "")
        if val:
            return str(val).strip()
    return json.dumps(payload, ensure_ascii=False) if payload else ""


def _extract_topic(summary: str) -> str:
    """Infer a short topic label from the summary text."""
    lower = summary.lower()
    topics = [
        ("reward",       "reward function"),
        ("entropy",      "entropy / ent_coef"),
        ("curriculum",   "curriculum training"),
        ("bc ",          "behavioral cloning"),
        ("self-play",    "self-play"),
        ("export",       "ONNX export"),
        ("submission",   "submission packaging"),
        ("mask",         "action masking"),
        ("bug",          "bug fix"),
        ("notebook",     "notebook update"),
        ("feature",      "feature engineering"),
    ]
    for key, label in topics:
        if key in lower:
            return label
    return "session update"


def _bullet_lines(summary: str) -> list[str]:
    """Convert raw summary text into concise bullet points."""
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]

    # If summary is a JSON blob (no real content), skip it
    if lines and lines[0].startswith("{"):
        return []

    bullets: list[str] = []
    for ln in lines:
        # Skip lines that look like metadata / paths
        if any(x in ln for x in ("session_id", "transcript_path", "hook_event", "cwd", "trigger")):
            continue
        # Skip lines that are pure punctuation or headers we don't need
        if ln in ("{", "}", "---") or ln.startswith("##"):
            continue
        # Normalise: strip leading bullet chars if already present
        clean = re.sub(r"^[-•*]\s*", "", ln)
        if clean:
            bullets.append(f"- {clean}")

    return bullets[:MAX_ENTRY_LINES]


def format_log_entry(summary: str) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    topic = _extract_topic(summary)
    lines = _bullet_lines(summary)

    if not lines:
        return ""   # nothing worth recording

    body = "\n".join(f"  {ln}" for ln in lines)
    return f"\n### {ts} — {topic}\n{body}\n"


def update_claude_md(entry: str) -> bool:
    if not entry.strip():
        print("[pre_compact] Nothing to record — skipping.", file=sys.stderr)
        return True

    if not CLAUDE_MD.exists():
        print(f"[pre_compact] CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return False

    content = CLAUDE_MD.read_text(encoding="utf-8")
    BACKUP_PATH.write_text(content, encoding="utf-8")

    if SESSION_LOG_HEADER in content:
        updated = content.rstrip("\n") + "\n" + entry
    else:
        updated = (
            content.rstrip("\n")
            + f"\n\n---\n\n{SESSION_LOG_HEADER}\n"
            + "*(Auto-appended by `scripts/pre_compact_update.py`)*\n"
            + entry
        )

    CLAUDE_MD.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    payload = read_stdin_payload()
    summary = extract_summary(payload)

    if not summary:
        print("[pre_compact] No summary — skipping.", file=sys.stderr)
        sys.exit(0)

    entry = format_log_entry(summary)

    if update_claude_md(entry):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[pre_compact] CLAUDE.md updated at {ts}.", file=sys.stderr)
    else:
        print("[pre_compact] CLAUDE.md update failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
