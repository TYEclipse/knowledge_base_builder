from __future__ import annotations

import pytest  # type: ignore[import-not-found]

from api_client import KimiApiClient


def test_safe_parse_json_invalid():
    with pytest.raises(Exception):
        KimiApiClient.safe_parse_json("not-json")


def test_safe_parse_json_empty_object():
    payload = KimiApiClient.safe_parse_json("{}")
    assert payload == {}
