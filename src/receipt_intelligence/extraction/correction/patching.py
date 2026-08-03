from __future__ import annotations

import copy
import math
from typing import Any

PATCH_FIELDS_BY_OPERATION: dict[str, set[str]] = {
    "replace_value": {"op", "reason", "path", "value"},
    "replace_array_element": {"op", "reason", "path", "index", "value"},
    "insert_array_element": {"op", "reason", "path", "index", "value"},
    "remove_array_elements": {"op", "reason", "path", "indices"},
}


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _resolve(root: Any, path: str) -> Any:
    if path == "":
        return root
    if not isinstance(path, str) or not path.startswith("/"):
        raise KeyError(f"invalid_json_pointer:{path}")
    current = root
    for raw in path[1:].split("/"):
        token = _unescape(raw)
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"missing_path:{path}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise KeyError(f"array_index_required:{path}")
            index = int(token)
            if index < 0 or index >= len(current):
                raise KeyError(f"array_index_out_of_range:{path}")
            current = current[index]
        else:
            raise KeyError(f"non_container_path:{path}")
    return current


def _replace(root: Any, path: str, value: Any) -> None:
    if not isinstance(path, str) or not path.startswith("/"):
        raise KeyError(f"invalid_json_pointer:{path}")
    tokens = path[1:].split("/")
    if not tokens or tokens == [""]:
        raise KeyError("root_replacement_not_supported")
    parent_path = "/" + "/".join(tokens[:-1]) if len(tokens) > 1 else ""
    parent = _resolve(root, parent_path)
    token = _unescape(tokens[-1])
    if isinstance(parent, dict):
        if token not in parent:
            raise KeyError(f"missing_path:{path}")
        parent[token] = copy.deepcopy(value)
    elif isinstance(parent, list):
        if not token.isdigit():
            raise KeyError(f"array_index_required:{path}")
        index = int(token)
        if index < 0 or index >= len(parent):
            raise KeyError(f"array_index_out_of_range:{path}")
        parent[index] = copy.deepcopy(value)
    else:
        raise KeyError(f"non_container_path:{path}")


def _supported_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_supported_json(child) for child in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _supported_json(child) for key, child in value.items())
    return False


def validate_patch(
    answer: Any,
    receipt: dict[str, Any],
    *,
    target: dict[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(answer, dict) or set(answer) != {"patches"}:
        return {
            "status": "invalid",
            "errors": [{"code": "PATCH_ENVELOPE_INVALID"}],
            "patch_count": 0,
        }
    patches = answer.get("patches")
    if not isinstance(patches, list):
        return {"status": "invalid", "errors": [{"code": "PATCHES_NOT_ARRAY"}], "patch_count": 0}
    max_patches = int(target.get("max_patches") or 1)
    if len(patches) > max_patches:
        errors.append({"code": "TOO_MANY_PATCHES", "value": len(patches), "maximum": max_patches})

    permitted_operations = set(target.get("permitted_operations") or [])
    permitted_value_paths = set(target.get("permitted_value_paths") or [])
    permitted_array_paths = set(target.get("permitted_array_paths") or [])
    working = copy.deepcopy(receipt)
    for index, patch in enumerate(patches):
        location = f"patches[{index}]"
        if not isinstance(patch, dict):
            errors.append({"code": "PATCH_NOT_OBJECT", "location": location})
            continue
        op = patch.get("op")
        if op not in PATCH_FIELDS_BY_OPERATION:
            errors.append(
                {"code": "UNSUPPORTED_PATCH_OPERATION", "location": f"{location}.op", "value": op}
            )
            continue
        if op not in permitted_operations:
            errors.append(
                {
                    "code": "OPERATION_NOT_PERMITTED_FOR_TARGET",
                    "location": f"{location}.op",
                    "value": op,
                }
            )
        if set(patch) != PATCH_FIELDS_BY_OPERATION[op]:
            errors.append({"code": "PATCH_FIELDS_DO_NOT_MATCH_OPERATION", "location": location})
        reason = patch.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
            errors.append({"code": "INVALID_PATCH_REASON", "location": location})
        path = patch.get("path")
        if not isinstance(path, str):
            errors.append({"code": "PATCH_PATH_REQUIRED", "location": location})
            continue
        try:
            if op == "replace_value":
                if path not in permitted_value_paths:
                    raise ValueError("path_not_permitted")
                if not _supported_json(patch.get("value")):
                    raise ValueError("unsupported_json_value")
                _replace(working, path, patch.get("value"))
            else:
                if path not in permitted_array_paths:
                    raise ValueError("array_path_not_permitted")
                array = _resolve(working, path)
                if not isinstance(array, list):
                    raise ValueError("path_is_not_array")
                if op == "replace_array_element":
                    array_index = patch.get("index")
                    if (
                        isinstance(array_index, bool)
                        or not isinstance(array_index, int)
                        or not 0 <= array_index < len(array)
                    ):
                        raise ValueError("array_index_out_of_range")
                    if not _supported_json(patch.get("value")):
                        raise ValueError("unsupported_json_value")
                    array[array_index] = copy.deepcopy(patch.get("value"))
                elif op == "insert_array_element":
                    array_index = patch.get("index")
                    if not _supported_json(patch.get("value")):
                        raise ValueError("unsupported_json_value")
                    if array_index is None:
                        array.append(copy.deepcopy(patch.get("value")))
                    elif (
                        isinstance(array_index, int)
                        and not isinstance(array_index, bool)
                        and 0 <= array_index <= len(array)
                    ):
                        array.insert(array_index, copy.deepcopy(patch.get("value")))
                    else:
                        raise ValueError("insert_index_out_of_range")
                elif op == "remove_array_elements":
                    indices = patch.get("indices")
                    if (
                        not isinstance(indices, list)
                        or not indices
                        or len(set(indices)) != len(indices)
                    ):
                        raise ValueError("indices_invalid")
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or not 0 <= value < len(array)
                        for value in indices
                    ):
                        raise ValueError("array_index_out_of_range")
                    for value in sorted(indices, reverse=True):
                        del array[value]
        except (KeyError, ValueError) as exc:
            errors.append(
                {"code": "INVALID_PATCH_OPERATION", "location": location, "message": str(exc)}
            )
    return {
        "status": "invalid" if errors else "valid",
        "errors": errors,
        "patch_count": len(patches),
    }


def apply_patch(receipt: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    corrected = copy.deepcopy(receipt)
    for patch in answer.get("patches", []):
        op = patch["op"]
        path = patch["path"]
        if op == "replace_value":
            _replace(corrected, path, patch["value"])
        else:
            array = _resolve(corrected, path)
            if op == "replace_array_element":
                array[patch["index"]] = copy.deepcopy(patch["value"])
            elif op == "insert_array_element":
                index = patch.get("index")
                if index is None:
                    array.append(copy.deepcopy(patch["value"]))
                else:
                    array.insert(index, copy.deepcopy(patch["value"]))
            elif op == "remove_array_elements":
                for index in sorted(set(patch["indices"]), reverse=True):
                    del array[index]
    return corrected


def target_for_strategy(
    strategy_id: str,
    check: dict[str, Any],
    validation: dict[str, Any],
    receipt: dict[str, Any],
    *,
    max_patches: int,
) -> dict[str, Any]:
    code = str(check.get("code") or "UNNAMED_CONSTRAINT")
    base = {
        "code": code,
        "severity": check.get("severity"),
        "message": check.get("message"),
        "constraint_values": check.get("values"),
        "constraint_details": check.get("details"),
        "validator_policy": validation.get("policy"),
        "strategy": strategy_id,
        "max_patches": max_patches,
    }
    if strategy_id == "item_sum_source_blocks_v3":
        items = receipt.get("items")
        items = items if isinstance(items, list) else []
        value_paths = [
            f"/items/{index}/{field_name}"
            for index, item in enumerate(items)
            if isinstance(item, dict)
            for field_name in ("final_price", "original_price", "discount_amount")
            if field_name in item
        ]
        operations = (["replace_value"] if value_paths else []) + (
            ["insert_array_element"] if isinstance(receipt.get("items"), list) else []
        )
        return {
            **base,
            "permitted_operations": operations,
            "permitted_value_paths": value_paths,
            "permitted_array_paths": ["/items"] if isinstance(receipt.get("items"), list) else [],
            "model_patch_supported": bool(operations),
        }
    if strategy_id == "vat_source_evidence_v9":
        paths = ["/tax/vat_lines", "/tax/vat_amount"]
        tax = receipt.get("tax")
        if (
            isinstance(tax, dict)
            and isinstance(tax.get("vat_amount"), dict)
            and "vat_amount" in tax["vat_amount"]
        ):
            paths.append("/tax/vat_amount/vat_amount")
        return {
            **base,
            "permitted_operations": ["replace_value"],
            "permitted_value_paths": paths,
            "permitted_array_paths": [],
            "model_patch_supported": True,
        }
    if strategy_id == "final_total_source_evidence_v2_4":
        paths = ["/totals/final_purchase_total"]
        totals = receipt.get("totals")
        if (
            isinstance(totals, dict)
            and isinstance(totals.get("final_purchase_total"), dict)
            and "final_purchase_total" in totals["final_purchase_total"]
        ):
            paths.append("/totals/final_purchase_total/final_purchase_total")
        return {
            **base,
            "permitted_operations": ["replace_value"],
            "permitted_value_paths": paths,
            "permitted_array_paths": [],
            "model_patch_supported": True,
        }
    return {
        **base,
        "permitted_operations": [],
        "permitted_value_paths": [],
        "permitted_array_paths": [],
        "model_patch_supported": False,
        "routing_reason": "No specialist mutation scope is registered for this strategy.",
    }
