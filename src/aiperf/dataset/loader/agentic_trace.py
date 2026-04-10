# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic coding trace dataset loader.

Loads JSONL files where each line represents a single inference step from an
agentic coding session (e.g. OpenCode, Claude Code). Steps are grouped by
``session_id`` into multi-turn conversations using DELTAS_WITHOUT_RESPONSES
mode: each turn is a delta (new content for that step), and the model's live
response is captured and accumulated by aiperf.

This ensures identical token prefixes across steps for KV cache reuse on the
inference server -- the same approach used by kv-cache-tester.

Two input modes:

- ``text_input``: Real content deltas (e.g. tool results from OpenCode DB).
- ``input_length``: Synthetic delta generation sized to token count.
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
    conversations using DELTAS_WITHOUT_RESPONSES mode.

    Each turn represents the delta content added at that step (user message
    or tool results). aiperf captures the model's live response and
    accumulates it in the conversation history. This produces identical
    token prefixes across steps, enabling KV cache reuse on the server.

    Compaction steps (is_compaction=true) reset the accumulated context.

    Example JSONL (real content)::

        {"session_id": "s1", "step_index": 0, "text_input": "Build an inference engine...", "output_length": 1091}
        {"session_id": "s1", "step_index": 1, "text_input": "Tool result: file created...", "output_length": 800, "delay": 2000}

    Example JSONL (synthetic)::

        {"session_id": "s1", "step_index": 0, "input_length": 14000, "output_length": 1091}
        {"session_id": "s1", "step_index": 1, "input_length": 2000, "output_length": 800, "delay": 2000}
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
        return ConversationContextMode.DELTAS_WITHOUT_RESPONSES

    def _get_text_input(self, trace: AgenticTrace) -> str | None:
        return trace.text_input

    def _build_turn(self, trace: AgenticTrace, prompt: str) -> Turn:
        return Turn(
            timestamp=trace.timestamp,
            delay=trace.delay,
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=(trace.output_length + trace.reasoning_length),
            compaction=trace.is_compaction,
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
            "text_input",
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
                    text_input=original.text_input,
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
