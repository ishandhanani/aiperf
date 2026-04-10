#!/usr/bin/env python3
"""Export OpenCode sessions to agentic_trace JSONL for aiperf.

Reads the OpenCode SQLite database and exports per-step token counts and
timing for agentic coding sessions. Each JSONL line represents one inference
step (assistant message) with input/output token counts, inter-step delay
(tool execution time), and compaction flags.

This is the token-count-based counterpart to opencode_to_trace.py (which
exports raw messages for mooncake_trace format).

Usage:
    # List sessions
    python scripts/opencode_export.py --list

    # Export a specific session by title substring
    python scripts/opencode_export.py --match "inference engine" -o engine.jsonl

    # Export by session ID
    python scripts/opencode_export.py --session-id ses_abc123 -o trace.jsonl

    # Export all main sessions
    python scripts/opencode_export.py --all -o traces/

    # Include subagent sessions as separate conversations
    python scripts/opencode_export.py --match "inference" --include-subagents -o trace.jsonl

    # Create replicas for multi-user simulation
    python scripts/opencode_export.py --match "inference" --replicas 4 -o trace.jsonl
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB_PATHS = [
    Path.home() / ".local/share/opencode/opencode.db",
    Path.home() / ".local/share/opencode/opencode-local.db",
]


def find_db() -> Path:
    for p in DEFAULT_DB_PATHS:
        if p.exists():
            return p
    print("No OpenCode database found. Try --db <path>", file=sys.stderr)
    sys.exit(1)


def list_sessions(db: sqlite3.Connection) -> None:
    cur = db.execute(
        "SELECT id, parent_id, title, time_created FROM session ORDER BY time_created DESC"
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    sessions = [dict(zip(cols, r)) for r in rows]

    parents = [s for s in sessions if not s["parent_id"]]
    children_map: dict[str, list[dict]] = {}
    for s in sessions:
        if s["parent_id"]:
            children_map.setdefault(s["parent_id"], []).append(s)

    for p in parents:
        msg_count = db.execute(
            "SELECT count(*) FROM message WHERE session_id = ?", (p["id"],)
        ).fetchone()[0]
        assistant_count = _count_assistant_messages(db, p["id"])
        children = children_map.get(p["id"], [])

        print(f"\n{p['title']}")
        print(f"  id: {p['id']}")
        print(f"  messages: {msg_count} total, {assistant_count} inference steps")
        if children:
            print(f"  subagents: {len(children)}")
            for c in children:
                c_steps = _count_assistant_messages(db, c["id"])
                print(f"    - {(c['title'] or '?')[:60]} ({c_steps} steps)")


def _count_assistant_messages(db: sqlite3.Connection, session_id: str) -> int:
    cur = db.execute(
        "SELECT count(*) FROM message WHERE session_id = ? "
        "AND json_extract(data, '$.role') = 'assistant'",
        (session_id,),
    )
    return cur.fetchone()[0]


def extract_steps(db: sqlite3.Connection, session_id: str) -> list[dict]:
    """Extract per-step token counts and timing from a session."""
    cur = db.execute(
        "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created",
        (session_id,),
    )
    messages = [(r[0], json.loads(r[1])) for r in cur.fetchall()]

    # Prefetch all parts grouped by message
    cur = db.execute(
        "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    parts_by_msg: dict[str, list[dict]] = {}
    for r in cur.fetchall():
        parts_by_msg.setdefault(r[0], []).append(json.loads(r[1]))

    steps = []
    prev_completed: int | None = None

    for msg_id, data in messages:
        if data.get("role") != "assistant":
            continue

        tokens = data.get("tokens", {})
        if not tokens:
            continue

        time_info = data.get("time", {})
        created = time_info.get("created")
        completed = time_info.get("completed")

        # Count tool calls from parts
        parts = parts_by_msg.get(msg_id, [])
        tool_count = sum(1 for p in parts if p.get("type") == "tool")

        # Compute inter-step delay
        delay = None
        if prev_completed is not None and created is not None:
            delay = max(0, created - prev_completed)

        is_compaction = data.get("mode") == "compaction" or data.get("agent") == "compaction"

        cache = tokens.get("cache", {})

        step = {
            "session_id": session_id,
            "step_index": len(steps),
            "input_length": tokens.get("input", 0),
            "output_length": tokens.get("output", 0),
            "reasoning_length": tokens.get("reasoning", 0),
            "cache_read": cache.get("read", 0),
            "cache_write": cache.get("write", 0),
            "delay": delay,
            "timestamp": created,
            "is_compaction": is_compaction,
            "finish_reason": data.get("finish"),
            "tool_call_count": tool_count,
            "model": data.get("modelID"),
        }
        steps.append(step)

        if completed is not None:
            prev_completed = completed

    return steps


def normalize_timestamps(entries: list[dict]) -> None:
    """Normalize timestamps relative to the earliest entry."""
    timestamps = [e["timestamp"] for e in entries if e["timestamp"] is not None]
    if not timestamps:
        return
    base_time = min(timestamps)
    for e in entries:
        if e["timestamp"] is not None:
            e["timestamp"] = e["timestamp"] - base_time


def replicate_entries(
    entries: list[dict], num_replicas: int, offset_ms: int
) -> list[dict]:
    """Create replicated traces with unique session IDs and staggered timestamps."""
    if num_replicas <= 1:
        return entries

    all_entries = []
    for replica in range(num_replicas):
        time_offset = replica * offset_ms
        for entry in entries:
            new_entry = {
                **entry,
                "session_id": f"r{replica}-{entry['session_id']}",
            }
            if new_entry["timestamp"] is not None:
                new_entry["timestamp"] = new_entry["timestamp"] + time_offset
            all_entries.append(new_entry)

    all_entries.sort(key=lambda e: e.get("timestamp") or 0)
    return all_entries


def write_jsonl(entries: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(entries)} steps to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenCode sessions to agentic_trace JSONL for aiperf"
    )
    parser.add_argument("--db", type=Path, help="Path to OpenCode SQLite database")
    parser.add_argument("--list", action="store_true", help="List sessions")
    parser.add_argument("--match", type=str, help="Export session matching title substring")
    parser.add_argument("--session-id", type=str, help="Export specific session by ID")
    parser.add_argument("--all", action="store_true", help="Export all main sessions")
    parser.add_argument(
        "--include-subagents",
        action="store_true",
        help="Include subagent sessions as separate conversations",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="trace.jsonl", help="Output file or directory (with --all)"
    )
    parser.add_argument("--replicas", type=int, default=1, help="Number of trace replicas")
    parser.add_argument(
        "--replica-offset-ms", type=int, default=5000, help="Milliseconds between replica starts"
    )
    args = parser.parse_args()

    db_path = args.db or find_db()
    db = sqlite3.connect(str(db_path))

    if args.list:
        list_sessions(db)
        db.close()
        return

    # Find target session IDs
    cur = db.execute(
        "SELECT id, parent_id, title FROM session ORDER BY time_created DESC"
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    all_sessions = [dict(zip(cols, r)) for r in rows]
    parents = [s for s in all_sessions if not s["parent_id"]]

    targets: list[str] = []
    if args.session_id:
        targets = [args.session_id]
    elif args.match:
        for s in parents:
            if args.match.lower() in (s["title"] or "").lower():
                targets.append(s["id"])
        if not targets:
            print(f"No session matching '{args.match}'. Use --list.", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        targets = [s["id"] for s in parents]
    else:
        parser.print_help()
        sys.exit(1)

    # Export
    if args.all and len(targets) > 1:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid in targets:
            session = next((s for s in parents if s["id"] == pid), None)
            title = (session["title"] or "unknown")[:40].replace(" ", "_").replace("/", "_")

            entries = _collect_entries(db, pid, all_sessions, args.include_subagents)
            normalize_timestamps(entries)
            entries = replicate_entries(entries, args.replicas, args.replica_offset_ms)

            out_file = out_dir / f"{title}.jsonl"
            write_jsonl(entries, str(out_file))
    else:
        all_entries: list[dict] = []
        for pid in targets:
            entries = _collect_entries(db, pid, all_sessions, args.include_subagents)
            all_entries.extend(entries)

        normalize_timestamps(all_entries)
        all_entries = replicate_entries(all_entries, args.replicas, args.replica_offset_ms)
        write_jsonl(all_entries, args.output)

    db.close()


def _collect_entries(
    db: sqlite3.Connection,
    parent_id: str,
    all_sessions: list[dict],
    include_subagents: bool,
) -> list[dict]:
    """Collect steps from a parent session and optionally its subagents."""
    entries = extract_steps(db, parent_id)

    if include_subagents:
        children = [s for s in all_sessions if s["parent_id"] == parent_id]
        for child in children:
            child_steps = extract_steps(db, child["id"])
            entries.extend(child_steps)

    entries.sort(key=lambda e: e.get("timestamp") or 0)
    return entries


if __name__ == "__main__":
    main()
