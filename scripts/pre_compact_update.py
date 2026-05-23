#!/usr/bin/env python3
"""
Pre-compact hook: automatically updates CLAUDE.md Session Update Log
before Claude Code compacts the conversation context.

Invoked by .claude/settings.json PreCompact hook.
Receives a JSON payload on stdin from Claude Code.
"""

import json
import sys
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
BACKUP_PATH = PROJECT_ROOT / "CLAUDE.md.bak"

SESSION_LOG_HEADER = "## 14. Session Update Log"


def read_stdin_payload() -> dict:
    """Read and parse the JSON payload from Claude Code's PreCompact hook."""
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except json.JSONDecodeError:
        # Not JSON — treat raw text as summary directly
        return {"summary": raw}
    except Exception as e:
        print(f"[pre_compact] stdin read error: {e}", file=sys.stderr)
        return {}


def extract_summary(payload: dict) -> str:
    """Extract the conversation summary from the hook payload."""
    # Claude Code PreCompact hook sends summary in one of these keys
    for key in ("summary", "content", "text", "message"):
        if key in payload and payload[key]:
            return str(payload[key]).strip()
    # Fallback: serialize the whole payload
    if payload:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return ""


def compress_summary(summary: str, max_lines: int = 30) -> str:
    """Keep the summary concise for the log entry."""
    lines = [ln for ln in summary.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Keep first 20 + last 5 lines with ellipsis
    kept = lines[:20] + ["...(truncated)..."] + lines[-5:]
    return "\n".join(kept)


def format_log_entry(summary: str) -> str:
    """Format a dated log entry for the Session Update Log section."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    compressed = compress_summary(summary)
    # Indent each line with 2 spaces so it reads cleanly under the heading
    indented = "\n".join(f"  {ln}" if ln.strip() else "" for ln in compressed.splitlines())
    return f"\n### {ts} — Auto-compact snapshot\n{indented}\n"


def update_claude_md(entry: str) -> bool:
    """Append the log entry to the Session Update Log section of CLAUDE.md."""
    if not CLAUDE_MD.exists():
        print(f"[pre_compact] CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return False

    content = CLAUDE_MD.read_text(encoding="utf-8")

    # Create a backup before modifying
    BACKUP_PATH.write_text(content, encoding="utf-8")

    if SESSION_LOG_HEADER in content:
        # Append before the final newline of the file
        updated = content.rstrip("\n") + "\n" + entry
    else:
        # Section missing — add it at the end
        updated = content.rstrip("\n") + f"\n\n---\n\n{SESSION_LOG_HEADER}\n*(Auto-appended by `scripts/pre_compact_update.py` before each compact)*\n{entry}"

    CLAUDE_MD.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    payload = read_stdin_payload()
    summary = extract_summary(payload)

    if not summary:
        print("[pre_compact] No summary content received — skipping CLAUDE.md update.", file=sys.stderr)
        sys.exit(0)

    entry = format_log_entry(summary)

    if update_claude_md(entry):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[pre_compact] CLAUDE.md Session Update Log appended at {ts}.", file=sys.stderr)
    else:
        print("[pre_compact] CLAUDE.md update failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
