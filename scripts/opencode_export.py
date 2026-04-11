#!/usr/bin/env python3
"""Export OpenCode sessions to agentic_trace JSONL for aiperf.

Reads the OpenCode SQLite database and exports per-step deltas for agentic
coding sessions. Each JSONL line represents the delta content for one
inference step -- either the initial user message or tool results from the
previous step's tool calls.

Uses DELTAS_WITHOUT_RESPONSES mode: aiperf accumulates deltas + live model
responses. The server sees identical token prefixes and reuses KV cache.

Usage:
    python scripts/opencode_export.py --list
    python scripts/opencode_export.py --match "inference engine" -o engine.jsonl
    python scripts/opencode_export.py --session-id ses_abc123 -o trace.jsonl
    python scripts/opencode_export.py --all -o traces/
    python scripts/opencode_export.py --match "inference" --include-subagents -o trace.jsonl
"""

import argparse
import json
import random
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
        children = children_map.get(p["id"], [])

        print(f"\n{p['title']}")
        print(f"  id: {p['id']}")
        print(f"  messages: {msg_count}")
        if children:
            print(f"  subagents: {len(children)}")
            for c in children:
                cm = db.execute(
                    "SELECT count(*) FROM message WHERE session_id = ?", (c["id"],)
                ).fetchone()[0]
                print(f"    - {(c['title'] or '?')[:60]} ({cm} msgs)")


def _get_parts(db: sqlite3.Connection, session_id: str) -> dict[str, list[dict]]:
    """Prefetch all parts grouped by message_id."""
    cur = db.execute(
        "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    parts_by_msg: dict[str, list[dict]] = {}
    for r in cur.fetchall():
        parts_by_msg.setdefault(r[0], []).append(json.loads(r[1]))
    return parts_by_msg


def _extract_tool_results(parts: list[dict]) -> str:
    """Extract tool result text from an assistant message's parts.

    Each tool part has state.output containing the tool's output text.
    This is the content that gets fed back as the delta for the next step.
    """
    results = []
    for part in parts:
        if part.get("type") != "tool":
            continue
        state = part.get("state", {})
        tool_name = part.get("tool", "unknown")
        output = state.get("output", "")
        if output:
            results.append(f"[Tool: {tool_name}]\n{output}")
    return "\n\n".join(results)


def _extract_text_content(parts: list[dict]) -> str:
    """Extract text content from message parts."""
    texts = []
    for part in parts:
        if part.get("type") == "text" and part.get("text", "").strip():
            texts.append(part["text"])
    return "\n".join(texts)


def extract_steps(db: sqlite3.Connection, session_id: str) -> list[dict]:
    """Extract per-step deltas from a session.

    For DELTAS_WITHOUT_RESPONSES mode:
    - Step 0 (first user message): the user's prompt text
    - Step N (after assistant response with tool calls): tool results text
    - Compaction step: the compaction summary text
    """
    cur = db.execute(
        "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created",
        (session_id,),
    )
    messages = [(r[0], json.loads(r[1])) for r in cur.fetchall()]
    parts_by_msg = _get_parts(db, session_id)

    steps = []
    prev_completed: int | None = None
    # Track tool results from the previous assistant step
    pending_tool_results: str | None = None

    for msg_id, data in messages:
        parts = parts_by_msg.get(msg_id, [])
        role = data.get("role")

        if role == "user":
            # User messages: extract text content as a delta
            text = _extract_text_content(parts)
            if not text.strip():
                # Compaction trigger or empty user message -- skip
                continue

            tokens = data.get("tokens", {})
            time_info = data.get("time", {})
            created = time_info.get("created")

            delay = None
            if prev_completed is not None and created is not None:
                delay = max(0, created - prev_completed)

            steps.append({
                "session_id": session_id,
                "step_index": len(steps),
                "text_input": text,
                "output_length": 4096,  # placeholder, actual output determined by model
                "delay": delay,
                "timestamp": created,
                "is_compaction": False,
                "finish_reason": None,
                "tool_call_count": 0,
                "model": None,
            })

        elif role == "assistant":
            tokens = data.get("tokens", {})
            time_info = data.get("time", {})
            created = time_info.get("created")
            completed = time_info.get("completed")
            is_compaction = data.get("mode") == "compaction" or data.get("agent") == "compaction"

            if is_compaction:
                # Compaction: the summary text IS the delta (replaces all prior context)
                summary_text = _extract_text_content(parts)
                if summary_text.strip():
                    delay = None
                    if prev_completed is not None and created is not None:
                        delay = max(0, created - prev_completed)

                    steps.append({
                        "session_id": session_id,
                        "step_index": len(steps),
                        "text_input": summary_text,
                        "output_length": tokens.get("output", 1024),
                        "reasoning_length": tokens.get("reasoning", 0),
                        "delay": delay,
                        "timestamp": created,
                        "is_compaction": True,
                        "finish_reason": data.get("finish"),
                        "tool_call_count": 0,
                        "model": data.get("modelID"),
                    })
            else:
                # Normal assistant step
                # If the previous step produced tool results, emit them as a delta first
                tool_results = _extract_tool_results(parts)
                tool_count = sum(1 for p in parts if p.get("type") == "tool")

                if pending_tool_results:
                    # Emit pending tool results from PREVIOUS step as a delta
                    delay = None
                    if prev_completed is not None and created is not None:
                        delay = max(0, created - prev_completed)

                    steps.append({
                        "session_id": session_id,
                        "step_index": len(steps),
                        "text_input": pending_tool_results,
                        "output_length": tokens.get("output", 1024),
                        "reasoning_length": tokens.get("reasoning", 0),
                        "delay": delay,
                        "timestamp": created,
                        "is_compaction": False,
                        "finish_reason": data.get("finish"),
                        "tool_call_count": tool_count,
                        "model": data.get("modelID"),
                    })

                # Store this step's tool results for the next step's delta
                pending_tool_results = tool_results if tool_results else None

            if completed is not None:
                prev_completed = completed

    return steps


def normalize_timestamps(entries: list[dict]) -> None:
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


def advance_session(
    steps: list[dict], pct: float, jitter: float, rng: random.Random
) -> list[dict]:
    """Skip the first pct% of steps, replacing the new first step with a
    synthetic bootstrap sized to the cumulative context at that point.

    This lets users start mid-trace so the server reaches steady-state
    KV cache occupancy faster.
    """
    if pct <= 0 or len(steps) <= 2:
        return steps

    actual_pct = max(0.0, min(0.95, pct + rng.uniform(-jitter, jitter)))
    start_idx = int(len(steps) * actual_pct)
    if start_idx == 0:
        return steps

    # Estimate cumulative context at the start point (deltas + estimated responses)
    cumulative = 0
    for s in steps[:start_idx]:
        if s.get("text_input"):
            cumulative += len(s["text_input"]) // 4
        elif s.get("input_length"):
            cumulative += s["input_length"]
        cumulative += s.get("output_length", 0)

    advanced = [s.copy() for s in steps[start_idx:]]

    # Replace first step with a synthetic bootstrap
    first = advanced[0].copy()
    first["step_index"] = 0
    first["text_input"] = None
    first["input_length"] = max(1, cumulative)
    first["delay"] = None
    first["timestamp"] = steps[start_idx].get("timestamp", 0)
    first["is_compaction"] = False
    advanced[0] = first

    # Re-index
    for i, s in enumerate(advanced):
        s["step_index"] = i

    return advanced


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
    parser.add_argument(
        "--advance-pct", type=float, default=0.0,
        help="Skip first N%% of steps per session for steady-state benchmarking (0.0-0.95)",
    )
    parser.add_argument(
        "--advance-jitter", type=float, default=0.0,
        help="Random jitter around advance-pct so sessions start at different points (0.0-0.5)",
    )
    args = parser.parse_args()

    db_path = args.db or find_db()
    db = sqlite3.connect(str(db_path))

    if args.list:
        list_sessions(db)
        db.close()
        return

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

    if args.all and len(targets) > 1:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid in targets:
            session = next((s for s in parents if s["id"] == pid), None)
            title = (session["title"] or "unknown")[:40].replace(" ", "_").replace("/", "_")
            entries = _collect_entries(db, pid, all_sessions, args.include_subagents, args.advance_pct, args.advance_jitter)
            normalize_timestamps(entries)
            entries = replicate_entries(entries, args.replicas, args.replica_offset_ms)
            out_file = out_dir / f"{title}.jsonl"
            write_jsonl(entries, str(out_file))
    else:
        all_entries: list[dict] = []
        for pid in targets:
            entries = _collect_entries(db, pid, all_sessions, args.include_subagents, args.advance_pct, args.advance_jitter)
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
    advance_pct: float = 0.0,
    advance_jitter: float = 0.0,
) -> list[dict]:
    rng = random.Random(hash(parent_id))
    entries = extract_steps(db, parent_id)
    if advance_pct > 0:
        entries = advance_session(entries, advance_pct, advance_jitter, rng)
    if include_subagents:
        children = [s for s in all_sessions if s["parent_id"] == parent_id]
        for child in children:
            child_steps = extract_steps(db, child["id"])
            if advance_pct > 0:
                child_rng = random.Random(hash(child["id"]))
                child_steps = advance_session(child_steps, advance_pct, advance_jitter, child_rng)
            entries.extend(child_steps)
    entries.sort(key=lambda e: e.get("timestamp") or 0)
    return entries


if __name__ == "__main__":
    main()
