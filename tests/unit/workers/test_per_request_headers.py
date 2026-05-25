# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


def _make_endpoint() -> ChatEndpoint:
    model_endpoint = create_model_endpoint(EndpointType.CHAT)
    return create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)


def test_turn_headers_merge_into_current_request_headers() -> None:
    endpoint = _make_endpoint()
    endpoint.model_endpoint.endpoint.headers = [("x-worker-instance-id", "0")]
    turn = Turn(
        texts=[Text(contents=["hello"])],
        headers={"x-worker-instance-id": 1, "x-pin-reason": "reuse"},
    )
    request_info = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        turns=[turn],
    )

    headers = endpoint.get_endpoint_headers(request_info)

    assert headers["x-worker-instance-id"] == "1"
    assert headers["x-pin-reason"] == "reuse"


def test_api_key_overrides_turn_authorization_header() -> None:
    endpoint = _make_endpoint()
    endpoint.model_endpoint.endpoint.api_key = "real-key"
    turn = Turn(
        texts=[Text(contents=["hello"])],
        headers={"Authorization": "Bearer wrong-key"},
    )
    request_info = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        turns=[turn],
    )

    headers = endpoint.get_endpoint_headers(request_info)

    assert headers["Authorization"] == "Bearer real-key"
