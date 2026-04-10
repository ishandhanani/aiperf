# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic coding trace dataset loader.

Loads JSONL files where each line represents a single inference step from an
agentic coding session (e.g. OpenCode, Claude Code). Steps are grouped by
``session_id`` into multi-turn conversations. Each step carries its own
``input_length``, so context mode is always MESSAGE_ARRAY_WITH_RESPONSES
(no accumulation -- the sawtooth context pattern is encoded directly in the
per-step input_length values).
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Text, Turn
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import AgenticTrace


class AgenticTraceDatasetLoader(BaseTraceDatasetLoader[AgenticTrace]):
    """Dataset loader for agentic coding traces.

    Loads JSONL files where each line is one inference step from an agentic
    coding session. Steps are grouped by session_id into multi-turn
    conversations. Context mode is always MESSAGE_ARRAY_WITH_RESPONSES
    because each step carries its own input_length (context grows within
    a session, with compaction resets).

    Example JSONL::

        {"session_id": "s1", "step_index": 0, "input_length": 13361, "output_length": 1091}
        {"session_id": "s1", "step_index": 1, "input_length": 15000, "output_length": 800, "delay": 2000}
        {"session_id": "s1", "step_index": 2, "input_length": 160848, "output_length": 1435, "is_compaction": true}
        {"session_id": "s1", "step_index": 3, "input_length": 6000, "output_length": 500}
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        if data is None:
            return False
        try:
            AgenticTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    # ------------------------------------------------------------------
    # Template-method hooks (see BaseTraceDatasetLoader.load_dataset)
    # ------------------------------------------------------------------

    def _parse_trace(self, line: str) -> AgenticTrace:
        return AgenticTrace.model_validate_json(line)

    def _group_traces(
        self, items: list[AgenticTrace]
    ) -> dict[str, list[AgenticTrace]]:
        data: dict[str, list[AgenticTrace]] = defaultdict(list)
        for trace in items:
            data[trace.session_id].append(trace)
        for traces in data.values():
            traces.sort(key=lambda t: t.step_index)
        return dict(data)

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _infer_context_mode(
        self, traces: list[AgenticTrace]
    ) -> ConversationContextMode:
        return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

    def _build_turn(self, trace: AgenticTrace, prompt: str) -> Turn:
        return Turn(
            timestamp=trace.timestamp,
            delay=trace.delay,
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=(trace.output_length + trace.reasoning_length),
        )

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset({
            "type",
            "session_id",
            "step_index",
            "is_compaction",
            "finish_reason",
            "tool_call_count",
            "model",
        })

    def _reconstruct_traces(
        self, originals: list[AgenticTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[AgenticTrace]:
        result = []
        for i, synth_dict in enumerate(synth_dicts):
            original = originals[i] if i < len(originals) else originals[-1]
            result.append(
                AgenticTrace(
                    session_id=original.session_id,
                    step_index=original.step_index,
                    input_length=synth_dict.get("input_length", original.input_length),
                    output_length=synth_dict.get("output_length", original.output_length),
                    reasoning_length=synth_dict.get("reasoning_length", original.reasoning_length),
                    cache_read=synth_dict.get("cache_read", original.cache_read),
                    cache_write=synth_dict.get("cache_write", original.cache_write),
                    delay=synth_dict.get("delay", original.delay),
                    timestamp=synth_dict.get("timestamp", original.timestamp),
                    is_compaction=original.is_compaction,
                    finish_reason=original.finish_reason,
                    tool_call_count=original.tool_call_count,
                    model=original.model,
                )
            )
        return result
