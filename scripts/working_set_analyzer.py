#!/usr/bin/env python3
"""Analyze KV cache working set from agentic_trace JSONL.

Reads a trace JSONL and computes per-session context growth and aggregate
working set over time. Useful for understanding KV cache memory pressure,
compaction effectiveness, and capacity planning.

Usage:
    python scripts/working_set_analyzer.py trace.jsonl
    python scripts/working_set_analyzer.py trace.jsonl --plot working_set.png
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SessionState:
    context_tokens: int = 0
    steps: int = 0
    peak_tokens: int = 0
    compactions: int = 0


@dataclass
class Snapshot:
    timestamp_ms: float
    session_id: str
    context_tokens: int
    delta_tokens: int
    is_compaction: bool
    aggregate_tokens: int = 0  # filled in post-processing


def analyze(jsonl_path: str) -> tuple[list[Snapshot], dict[str, SessionState]]:
    """Parse JSONL and produce a timeline of working set snapshots."""
    with open(jsonl_path) as f:
        steps = [json.loads(line) for line in f if line.strip()]

    steps.sort(key=lambda s: s.get("timestamp") or 0)

    sessions: dict[str, SessionState] = defaultdict(SessionState)
    snapshots: list[Snapshot] = []

    for step in steps:
        sid = step["session_id"]
        session = sessions[sid]

        # Estimate delta tokens
        if step.get("text_input"):
            delta = len(step["text_input"]) // 4
        elif step.get("input_length"):
            delta = step["input_length"]
        else:
            delta = 0

        # Add estimated output tokens (model response adds to context)
        output_tokens = step.get("output_length", 0)

        if step.get("is_compaction"):
            # Compaction resets context to just this step's content
            session.context_tokens = delta
            session.compactions += 1
        else:
            session.context_tokens += delta + output_tokens

        session.steps += 1
        session.peak_tokens = max(session.peak_tokens, session.context_tokens)

        snapshots.append(Snapshot(
            timestamp_ms=step.get("timestamp") or 0,
            session_id=sid,
            context_tokens=session.context_tokens,
            delta_tokens=delta,
            is_compaction=step.get("is_compaction", False),
        ))

    # Compute aggregate working set at each snapshot
    current_context: dict[str, int] = {}
    for snap in snapshots:
        current_context[snap.session_id] = snap.context_tokens
        snap.aggregate_tokens = sum(current_context.values())

    return snapshots, dict(sessions)


def compute_windowed(snapshots: list[Snapshot], window_ms: float) -> list[tuple[float, int]]:
    """Compute time-windowed aggregate working set."""
    if not snapshots:
        return []

    results = []
    end_ms = snapshots[-1].timestamp_ms
    t = 0.0
    while t <= end_ms:
        # Find snapshots within [t - window_ms, t]
        window_snaps = [s for s in snapshots if t - window_ms <= s.timestamp_ms <= t]
        if window_snaps:
            results.append((t, window_snaps[-1].aggregate_tokens))
        t += window_ms / 10  # sample at 10 points per window

    return results


def print_summary(
    snapshots: list[Snapshot], sessions: dict[str, SessionState]
) -> None:
    if not snapshots:
        print("No data.")
        return

    peak_aggregate = max(s.aggregate_tokens for s in snapshots)
    avg_aggregate = sum(s.aggregate_tokens for s in snapshots) // len(snapshots)
    duration_s = (snapshots[-1].timestamp_ms - snapshots[0].timestamp_ms) / 1000

    print(f"\n{'='*60}")
    print(f"  Working Set Analysis")
    print(f"{'='*60}")
    print(f"  Sessions:           {len(sessions)}")
    print(f"  Total steps:        {sum(s.steps for s in sessions.values())}")
    print(f"  Duration:           {duration_s:.1f}s")
    print(f"  Peak aggregate:     {peak_aggregate:,} tokens ({peak_aggregate * 2 / 1024 / 1024:.1f} MB @ FP16)")
    print(f"  Avg aggregate:      {avg_aggregate:,} tokens")
    print(f"  Total compactions:  {sum(s.compactions for s in sessions.values())}")
    print()

    print(f"  Per-Session Summary:")
    print(f"  {'Session':<40s} {'Steps':>6s} {'Peak':>10s} {'Compactions':>12s}")
    print(f"  {'-'*40} {'-'*6} {'-'*10} {'-'*12}")
    for sid, state in sorted(sessions.items(), key=lambda x: -x[1].peak_tokens):
        label = sid[:38] + ".." if len(sid) > 40 else sid
        print(f"  {label:<40s} {state.steps:>6d} {state.peak_tokens:>10,d} {state.compactions:>12d}")
    print()


def plot_timeline(snapshots: list[Snapshot], output_path: str) -> None:
    """Plot working set over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.", file=sys.stderr)
        return

    timestamps = [s.timestamp_ms / 1000 for s in snapshots]
    aggregate = [s.aggregate_tokens / 1000 for s in snapshots]

    # Per-session lines
    session_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for s in snapshots:
        ts_list, tok_list = session_data[s.session_id]
        ts_list.append(s.timestamp_ms / 1000)
        tok_list.append(s.context_tokens / 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: aggregate
    ax1.fill_between(timestamps, aggregate, alpha=0.3, color="blue")
    ax1.plot(timestamps, aggregate, color="blue", linewidth=1)
    ax1.set_ylabel("Aggregate Working Set (K tokens)")
    ax1.set_title("KV Cache Working Set Over Time")
    ax1.grid(True, alpha=0.3)

    # Bottom: per-session
    for sid, (ts, toks) in session_data.items():
        ax2.plot(ts, toks, linewidth=0.8, alpha=0.7, label=sid[:20])
    ax2.set_ylabel("Per-Session Context (K tokens)")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    if len(session_data) <= 10:
        ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV cache working set from agentic_trace JSONL"
    )
    parser.add_argument("jsonl", help="Path to agentic_trace JSONL file")
    parser.add_argument("--plot", type=str, help="Output PNG path for working set plot")
    args = parser.parse_args()

    snapshots, sessions = analyze(args.jsonl)
    print_summary(snapshots, sessions)

    if args.plot:
        plot_timeline(snapshots, args.plot)


if __name__ == "__main__":
    main()
