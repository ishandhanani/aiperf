# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiperf.config.endpoint import EndpointConfig
from aiperf.config.loader.parsing import (
    parse_headers_as_dict,
    parse_headers_as_tuple_list,
    parse_str_or_dict_as_tuple_list,
)


def test_header_parser_preserves_numeric_values_as_strings() -> None:
    headers = parse_headers_as_tuple_list("x-worker-instance-id:123")

    assert headers == [("x-worker-instance-id", "123")]


def test_header_parser_stringifies_json_values() -> None:
    headers = parse_headers_as_dict('{"x-worker-instance-id": 123}')

    assert headers == {"x-worker-instance-id": "123"}


def test_generic_tuple_parser_still_coerces_extra_inputs() -> None:
    values = parse_str_or_dict_as_tuple_list("min_tokens:123,ignore_eos:true")

    assert values == [("min_tokens", 123), ("ignore_eos", True)]


def test_endpoint_config_stringifies_header_values() -> None:
    endpoint = EndpointConfig(
        urls=["localhost:8000"],
        headers={"x-worker-instance-id": 123},
    )

    assert endpoint.headers == {"x-worker-instance-id": "123"}


def test_header_parser_rejects_nested_values() -> None:
    with pytest.raises(ValueError, match="scalar strings"):
        parse_headers_as_dict({"x-bad": {"nested": "value"}})
