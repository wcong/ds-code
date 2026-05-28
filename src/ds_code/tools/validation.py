from __future__ import annotations

from typing import Any, Dict


def validate_input(schema: Dict[str, Any], payload: Dict[str, Any]) -> tuple[bool, str | None]:
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    for key in required:
        if key not in payload:
            return False, f"missing required field '{key}'"
    for key, value in payload.items():
        spec = properties.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return False, f"field '{key}' must be string"
        if expected == "integer" and not isinstance(value, int):
            return False, f"field '{key}' must be integer"
        if expected == "number" and not isinstance(value, (int, float)):
            return False, f"field '{key}' must be number"
        if expected == "boolean" and not isinstance(value, bool):
            return False, f"field '{key}' must be boolean"
        if expected == "object" and not isinstance(value, dict):
            return False, f"field '{key}' must be object"
    return True, None
