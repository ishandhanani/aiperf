#!/usr/bin/env python3
"""Extract OpenCode sessions into aiperf mooncake_trace JSONL files.

Reads the OpenCode SQLite database and converts session trees (parent + subagent
children) into mooncake_trace JSONL with nvext session_control for Dynamo.

Usage:
    # List sessions with subagents
    python scripts/opencode_to_trace.py --list

    # Extract a specific session by title substring
    python scripts/opencode_to_trace.py --match "inference engine" -o engine-trace.jsonl

    # Extract by session ID
    python scripts/opencode_to_trace.py --session-id ses_abc123 -o trace.jsonl

    # Extract all sessions with subagents
    python scripts/opencode_to_trace.py --all -o traces/

    # Use a specific DB file
    python scripts/opencode_to_trace.py --db ~/.local/share/opencode/opencode.db --list
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB_PATHS = [
    Path.home() / ".local/share/opencode/opencode.db",
    Path.home() / ".local/share/opencode/opencode-local.db",
]

_SESSION_TIMEOUT = 60


def find_db() -> Path:
    for p in DEFAULT_DB_PATHS:
        if p.exists():
            return p
    print("No OpenCode database found. Try --db <path>", file=sys.stderr)
    sys.exit(1)


def get_sessions_with_subagents(db: sqlite3.Connection) -> list[dict]:
    cur = db.execute(
        "SELECT id, parent_id, title, directory, time_created FROM session ORDER BY time_created DESC"
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    sessions = [dict(zip(cols, r)) for r in rows]

    parents = [s for s in sessions if not s["parent_id"]]
    result = []
    for p in parents:
        children = [s for s in sessions if s["parent_id"] == p["id"]]
        if children:
            result.append({"parent": p, "children": children})
    return result


def extract_session_turns(db: sqlite3.Connection, session_id: str) -> list[dict]:
    """Extract cumulative message arrays for each LLM turn in a session."""
    cur = db.execute(
        "SELECT id, data, time_created FROM message WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    messages = [{"id": r[0], "data": json.loads(r[1]), "time_created": r[2]} for r in cur.fetchall()]

    cur = db.execute(
        "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    parts_by_msg: dict[str, list[dict]] = {}
    for r in cur.fetchall():
        parts_by_msg.setdefault(r[0], []).append(json.loads(r[1]))

    entries = []
    cumulative: list[dict] = []

    for msg in messages:
        parsed = msg["data"]
        parts = parts_by_msg.get(msg["id"], [])

        if parsed.get("role") == "user":
            texts = [p["text"] for p in parts if p.get("type") == "text" and p.get("text", "").strip()]
            text = "\n".join(texts)
            if text.strip():
                cumulative.append({"role": "user", "content": text})
                entries.append({
                    "timestamp": msg["time_created"],
                    "messages": list(cumulative),
                })

        elif parsed.get("role") == "assistant":
            content_parts = []
            tool_calls = []
            tool_results = []

            for part in parts:
                if part.get("type") == "text" and part.get("text", "").strip():
                    content_parts.append(part["text"])
                if part.get("type") == "tool" and part.get("state"):
                    call_id = part.get("callID", f"tc_{os.urandom(4).hex()}")
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": part["tool"],
                            "arguments": json.dumps(part["state"].get("input", {})),
                        },
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (part["state"].get("output") or "")[:4000],
                    })

            if tool_calls:
                cumulative.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts) or None,
                    "tool_calls": tool_calls,
                })
                cumulative.extend(tool_results)
                entries.append({
                    "timestamp": msg["time_created"],
                    "messages": list(cumulative),
                })
            elif content_parts:
                cumulative.append({"role": "assistant", "content": "\n".join(content_parts)})

    return entries


def build_trace(db: sqlite3.Connection, parent_id: str, session_timeout: int = 60) -> list[dict]:
    """Build a mooncake_trace JSONL from a parent session + its children."""
    cur = db.execute(
        "SELECT id, parent_id, title FROM session WHERE id = ? OR parent_id = ? ORDER BY time_created",
        (parent_id, parent_id),
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    all_sessions = [dict(zip(cols, r)) for r in rows]

    all_entries = []

    # Parent: no nvext
    parent_turns = extract_session_turns(db, parent_id)
    for entry in parent_turns:
        all_entries.append({
            "session_id": parent_id,
            "timestamp": entry["timestamp"],
            "messages": entry["messages"],
            "output_length": 4096,
        })

    # Children: with nvext session_control
    children = [s for s in all_sessions if s["parent_id"] == parent_id]
    for child in children:
        child_turns = extract_session_turns(db, child["id"])
        dynamo_session_id = f"opencode-{child['id']}"

        for i, entry in enumerate(child_turns):
            is_first = i == 0
            is_last = i == len(child_turns) - 1

            session_control: dict = {
                "session_id": dynamo_session_id,
                "timeout": session_timeout,
            }
            if is_first:
                session_control["action"] = "open"
            elif is_last:
                session_control["action"] = "close"

            all_entries.append({
                "session_id": child["id"],
                "timestamp": entry["timestamp"],
                "messages": entry["messages"],
                "output_length": 4096,
                "nvext": {"session_control": session_control},
            })

    # Normalize timestamps and sort
    if all_entries:
        base_time = min(e["timestamp"] for e in all_entries)
        for e in all_entries:
            e["timestamp"] = e["timestamp"] - base_time
        all_entries.sort(key=lambda e: e["timestamp"])

    return all_entries


def replicate_trace(
    entries: list[dict], num_replicas: int, offset_ms: int
) -> list[dict]:
    """Create multiple replicas of a trace with unique system prompts and session IDs.

    Each replica gets:
    - A unique system prompt prefix (diverges KV from token 1)
    - Unique session IDs (separate Dynamo sessions)
    - Staggered start timestamps
    """
    if num_replicas <= 1:
        return entries

    all_entries = []
    for replica in range(num_replicas):
        time_offset = replica * offset_ms
        system_prefix = (
            f"You are user {replica + 1} of {num_replicas} in a shared workspace. "
            f"Your unique workspace identifier is replica-{replica:04d}. "
            f"Prioritize changes in your designated workspace area."
        )

        for entry in entries:
            new_entry = {
                **entry,
                "session_id": f"r{replica}-{entry['session_id']}",
                "timestamp": entry["timestamp"] + time_offset,
            }
            # Prepend system message to diverge prefix
            msgs = list(new_entry["messages"])
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {**msgs[0], "content": system_prefix + "\n\n" + msgs[0]["content"]}
            else:
                msgs.insert(0, {"role": "system", "content": system_prefix})
            new_entry["messages"] = msgs

            # Update nvext session IDs
            if "nvext" in new_entry:
                nvext = json.loads(json.dumps(new_entry["nvext"]))
                if "session_control" in nvext:
                    sc = nvext["session_control"]
                    sc["session_id"] = f"r{replica}-{sc['session_id']}"
                new_entry["nvext"] = nvext

            all_entries.append(new_entry)

    all_entries.sort(key=lambda e: e["timestamp"])
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Extract OpenCode sessions to aiperf mooncake_trace JSONL")
    parser.add_argument("--db", type=Path, help="Path to OpenCode SQLite database")
    parser.add_argument("--list", action="store_true", help="List sessions with subagents")
    parser.add_argument("--match", type=str, help="Extract session matching title substring")
    parser.add_argument("--session-id", type=str, help="Extract specific session by ID")
    parser.add_argument("--all", action="store_true", help="Extract all sessions with subagents")
    parser.add_argument("-o", "--output", type=str, default="trace.jsonl", help="Output file or directory (with --all)")
    parser.add_argument("--timeout", type=int, default=_SESSION_TIMEOUT, help="Dynamo session timeout in seconds")
    parser.add_argument("--replicas", type=int, default=1,
                        help="Number of replicas per trace. Each gets a unique system prompt "
                             "prefix and session IDs to simulate distinct concurrent users.")
    parser.add_argument("--replica-offset-ms", type=int, default=5000,
                        help="Milliseconds between replica start times (default: 5000)")
    args = parser.parse_args()

    db_path = args.db or find_db()
    db = sqlite3.connect(str(db_path))

    trees = get_sessions_with_subagents(db)

    if args.list:
        for tree in trees:
            p = tree["parent"]
            children = tree["children"]
            total_parts = sum(
                db.execute("SELECT count(*) FROM part WHERE session_id = ?", (c["id"],)).fetchone()[0]
                for c in children
            )
            parent_parts = db.execute("SELECT count(*) FROM part WHERE session_id = ?", (p["id"],)).fetchone()[0]
            print(f"\n{p['title']}")
            print(f"  id: {p['id']}")
            print(f"  dir: {p.get('directory', '?')}")
            print(f"  parts: {parent_parts} parent + {total_parts} child | {len(children)} subagents")
            for c in children:
                cp = db.execute("SELECT count(*) FROM part WHERE session_id = ?", (c["id"],)).fetchone()[0]
                print(f"    - {c['title'][:60]} ({cp} parts)")
        return

    targets = []
    if args.session_id:
        targets = [args.session_id]
    elif args.match:
        for tree in trees:
            if args.match.lower() in (tree["parent"]["title"] or "").lower():
                targets.append(tree["parent"]["id"])
        if not targets:
            print(f"No session matching '{args.match}'. Use --list to see available sessions.", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        targets = [tree["parent"]["id"] for tree in trees]
    else:
        parser.print_help()
        sys.exit(1)

    if args.all and len(targets) > 1:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid in targets:
            tree = next((t for t in trees if t["parent"]["id"] == pid), None)
            title = (tree["parent"]["title"] or "unknown")[:40].replace(" ", "_").replace("/", "_")
            entries = build_trace(db, pid, args.timeout)
            entries = replicate_trace(entries, args.replicas, args.replica_offset_ms)
            out_file = out_dir / f"{title}.jsonl"
            with open(out_file, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            print(f"Wrote {len(entries)} entries to {out_file}")
    else:
        for pid in targets:
            entries = build_trace(db, pid, args.timeout)
            entries = replicate_trace(entries, args.replicas, args.replica_offset_ms)
            with open(args.output, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            parent_count = sum(1 for e in entries if "nvext" not in e)
            child_count = sum(1 for e in entries if "nvext" in e)
            print(f"Wrote {len(entries)} entries ({parent_count} parent, {child_count} subagent) to {args.output}")

    db.close()


if __name__ == "__main__":
    main()
