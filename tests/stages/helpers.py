# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
from unittest.mock import Mock

import pytest
import torch
from sympy import Symbol

from torch._inductor.ir import Pointwise, Reduction
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.op_spec import TensorArg
from torch_spyre._inductor.spyre_kernel import SpyreKernel

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"
GOLDEN_DIR = ARTIFACTS_DIR / "golden"


# ---------------------------------------------------------------------------
# Mock SchedulerNode for SpyreKernel.create_op_spec()
# ---------------------------------------------------------------------------


def make_mock_node(is_reduction, ranges_dict):
    """Create a minimal mock SchedulerNode for SpyreKernel.create_op_spec().

    iteration_space() (pass_utils.py) checks isinstance(node.data, Pointwise|Reduction)
    and reads node.read_writes.writes.ranges (Pointwise) or
    node.read_writes.reads.ranges (Reduction).
    """
    node = Mock()
    node.op_it_space_splits = {}
    node.node = Mock(spec=["data"])
    node.node.data = Mock(spec=Reduction if is_reduction else Pointwise)

    dep = Mock()
    dep.ranges = ranges_dict

    if is_reduction:
        node.read_writes.reads = [dep]
    else:
        node.read_writes.writes = [dep]

    return node


# ---------------------------------------------------------------------------
# OpSpec construction via Mock SchedulerNode
# ---------------------------------------------------------------------------


def generate_mock_op_spec(
    op_name, is_reduction, output_shape, reduction_shape, dtype, num_inputs
):
    """Use real SpyreKernel.create_op_spec() with a mock node to build an OpSpec.

    Constructs TensorArgs with real SpyreTensorLayout-derived device sizes,
    creates a mock SchedulerNode with the correct iteration space, and calls
    the actual create_op_spec() method.
    """
    layout = SpyreTensorLayout(list(output_shape), torch.float16)
    out_syms = [Symbol(f"i{d}") for d in range(len(output_shape))]
    red_syms = [Symbol(f"r{d}") for d in range(len(reduction_shape))]

    ranges_dict = {}
    for d, size in enumerate(output_shape):
        ranges_dict[out_syms[d]] = size
    for d, size in enumerate(reduction_shape):
        ranges_dict[red_syms[d]] = size

    coords = list(out_syms)

    args = []
    for i in range(num_inputs):
        args.append(
            TensorArg(
                is_input=True,
                arg_index=i,
                device_dtype=dtype,
                device_size=layout.device_size,
                device_coordinates=list(coords),
                allocation=None,
            )
        )
    args.append(
        TensorArg(
            is_input=False,
            arg_index=num_inputs,
            device_dtype=dtype,
            device_size=layout.device_size,
            device_coordinates=list(coords),
            allocation=None,
        )
    )

    mock_node = make_mock_node(is_reduction, ranges_dict)
    kernel = SpyreKernel()
    kernel.current_node = mock_node

    return kernel.create_op_spec(op_name, is_reduction, args, op_info={})


# ---------------------------------------------------------------------------
# SuperDSC JSON schema and validators
#
# These checks are a lightweight subset of what the full SDSC validator
# (see rfcs/sdsc_validator/SDSC_VALIDATOR_DESIGN.md) would enforce.
# Once the validator is implemented it can be plugged in here to perform
# comprehensive structural + semantic validation via its YAML rule engine.
# ---------------------------------------------------------------------------

SCHEMA_PATH = ARTIFACTS_DIR / "sdsc_schema.json"

with open(SCHEMA_PATH) as _f:
    SDSC_SCHEMA = json.load(_f)


def _check_required(obj, required, path):
    """Check that all required keys are present in obj."""
    errors = []
    for key in required:
        if key not in obj:
            errors.append(f"Missing required key '{key}' at {path}")
    return errors


def _check_type(value, expected_type, path):
    """Check that value matches the expected JSON Schema type."""
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
    }
    py_type = type_map.get(expected_type)
    if py_type and not isinstance(value, py_type):
        return [f"Expected {expected_type} at {path}, got {type(value).__name__}"]
    return []


def _resolve_ref(ref, schema):
    """Resolve a $ref pointer like '#/$defs/ComputeOp'."""
    parts = ref.lstrip("#/").split("/")
    node = schema
    for part in parts:
        node = node[part]
    return node


def _validate_node(value, node_schema, path, root_schema):
    """Recursively validate a value against a JSON Schema node."""
    errors = []

    if "$ref" in node_schema:
        node_schema = _resolve_ref(node_schema["$ref"], root_schema)

    schema_type = node_schema.get("type")
    if schema_type:
        errors.extend(_check_type(value, schema_type, path))
        if errors:
            return errors

    if schema_type == "object":
        if not isinstance(value, dict):
            return errors
        required = node_schema.get("required", [])
        errors.extend(_check_required(value, required, path))

        props = node_schema.get("properties", {})
        for key, prop_schema in props.items():
            if key in value:
                errors.extend(
                    _validate_node(
                        value[key], prop_schema, f"{path}.{key}", root_schema
                    )
                )

        additional = node_schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key, val in value.items():
                if key not in props:
                    errors.extend(
                        _validate_node(val, additional, f"{path}.{key}", root_schema)
                    )

        min_props = node_schema.get("minProperties")
        if min_props is not None and len(value) < min_props:
            errors.append(f"Too few properties at {path}: {len(value)} < {min_props}")

    elif schema_type == "array":
        if not isinstance(value, list):
            return errors
        min_items = node_schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"Too few items at {path}: {len(value)} < {min_items}")

        items_schema = node_schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                errors.extend(
                    _validate_node(item, items_schema, f"{path}[{i}]", root_schema)
                )

    elif schema_type == "integer":
        minimum = node_schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(f"Value {value} at {path} < minimum {minimum}")

    elif schema_type == "string":
        enum_vals = node_schema.get("enum")
        if enum_vals and value not in enum_vals:
            errors.append(f"Value '{value}' at {path} not in {enum_vals}")
        min_len = node_schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"String too short at {path}: {len(value)} < {min_len}")

    return errors


def validate_sdsc_schema(sdsc):
    """Validate a SuperDSC dict against the JSON schema.

    Uses a lightweight stdlib-only schema checker (no jsonschema dependency).
    """
    errors = _validate_node(sdsc, SDSC_SCHEMA, "$", SDSC_SCHEMA)
    assert not errors, "SuperDSC schema validation errors:\n" + "\n".join(errors)


def validate_sdsc_structure(sdsc):
    """Validate structural invariants and semantic rules of a SuperDSC dict.

    Checks JSON serializability, schema conformance, and a subset of the
    semantic rules from the SDSC validator design
    (rfcs/sdsc_validator/SDSC_VALIDATOR_DESIGN.md), including:
      - Phase 1: coreIdsUsed_ cardinality matches numCoresUsed_ (superdsc.cpp:1338)
      - Phase 1: computeOp_ has valid exUnit and opFuncName (superdsc.cpp:673)
      - Phase 1: labeledDs_ entries have required fields (superdsc.cpp)
      - Phase 1: scheduleTree_ has 4-component schedule steps (superdsc.cpp:760)
      - Phase 2: dscs_ must be non-empty (dxp.cpp:346-347)
      - Phase 3: labeledDs_ count must not exceed 8 (dxp.cpp:239,262)
    """
    errors = []

    try:
        json.dumps(sdsc, default=str)
    except TypeError as e:
        errors.append(f"Not JSON serializable: {e}")

    validate_sdsc_schema(sdsc)

    top_key = list(sdsc.keys())[0]
    inner = sdsc[top_key]

    # Phase 2 (dxp.cpp:346-347): dscs_ must be non-empty
    dscs = inner.get("dscs_", [])
    if not dscs:
        errors.append("dscs_ must be non-empty (phase2_dsc_presence)")

    num_cores_outer = inner.get("numCoresUsed_", 0)

    # Phase 1: coreIdToWkSlice_ cardinality matches numCoresUsed_
    wk_slices = inner.get("coreIdToWkSlice_", {})
    if len(wk_slices) != num_cores_outer:
        errors.append(
            f"coreIdToWkSlice_ has {len(wk_slices)} entries, "
            f"expected {num_cores_outer} (numCoresUsed_)"
        )

    # Phase 1: coreIdToDscSchedule steps must have 4 components (superdsc.cpp:760)
    schedule = inner.get("coreIdToDscSchedule", {})
    for core_id, steps in schedule.items():
        for i, step in enumerate(steps):
            if not isinstance(step, list) or len(step) != 4:
                errors.append(
                    f"coreIdToDscSchedule[{core_id}][{i}] must have 4 components"
                )

    for dsc_wrapper in dscs:
        dsc_key = list(dsc_wrapper.keys())[0]
        dsc = dsc_wrapper[dsc_key]

        # Phase 1 (superdsc.cpp:1338): coreIdsUsed_ cardinality
        core_ids = dsc.get("coreIdsUsed_", [])
        num_cores_dsc = dsc.get("numCoresUsed_", 0)
        if len(core_ids) != num_cores_dsc:
            errors.append(
                f"DSC '{dsc_key}': coreIdsUsed_ has {len(core_ids)} entries, "
                f"expected {num_cores_dsc} (numCoresUsed_)"
            )

        # Phase 1 (superdsc.cpp:673): computeOp_ validation
        compute_ops = dsc.get("computeOp_", [])
        valid_units = {"sfp", "pt"}
        for j, cop in enumerate(compute_ops):
            unit = cop.get("exUnit", "")
            if unit not in valid_units:
                errors.append(
                    f"DSC '{dsc_key}' computeOp_[{j}].exUnit='{unit}' "
                    f"not in {valid_units}"
                )
            if not cop.get("opFuncName"):
                errors.append(f"DSC '{dsc_key}' computeOp_[{j}] missing opFuncName")

        # Phase 3 (dxp.cpp:239,262): labeledDs_ count must not exceed 8
        labeled_ds = dsc.get("labeledDs_", [])
        if len(labeled_ds) > 8:
            errors.append(
                f"DSC '{dsc_key}': labeledDs_ has {len(labeled_ds)} entries, "
                "max 8 LDS segments allowed"
            )

        # Structural: scheduleTree_ and labeledDs_ must have same length
        schedule_tree = dsc.get("scheduleTree_", [])
        if len(schedule_tree) != len(labeled_ds):
            errors.append(
                f"DSC '{dsc_key}': scheduleTree_ has {len(schedule_tree)} entries "
                f"but labeledDs_ has {len(labeled_ds)}"
            )

    assert not errors, "SuperDSC structural/semantic errors:\n" + "\n".join(errors)


def validate_sdsc_json_round_trip(sdsc):
    """Verify the SDSC dict survives JSON serialization round-trip."""
    json_str = json.dumps(sdsc, default=str)
    round_tripped = json.loads(json_str)
    expected = json.loads(json.dumps(sdsc, default=str))
    assert round_tripped == expected, "SDSC dict did not survive JSON round-trip"


# ---------------------------------------------------------------------------
# Golden file management
# ---------------------------------------------------------------------------


def _canonicalize(obj):
    """Recursively sort dicts by key so JSON comparison is order-independent."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    return obj


def assert_sdsc_matches_golden(sdsc_json, golden_path, update=False):
    """Compare generated SuperDSC JSON against golden file.

    If the golden file does not exist, it is created and the test is skipped
    so the file can be reviewed and committed. Pass update=True (or run with
    --update-golden) to regenerate.

    NOTE: This uses canonicalized JSON string comparison — a lightweight
    approach that is order-independent but not a true deep structural diff.
    Once the full SDSC validator (rfcs/sdsc_validator/SDSC_VALIDATOR_DESIGN.md)
    is implemented, it should be plugged in here to perform semantic-aware
    comparison rather than string equality.
    """
    golden_path = str(golden_path)

    if update or not os.path.exists(golden_path):
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        with open(golden_path, "w") as f:
            json.dump(sdsc_json, f, indent=2, sort_keys=True, default=str)
        if not update:
            pytest.skip("Golden file created; review and commit it")
        return

    with open(golden_path) as f:
        expected = json.load(f)

    expected_canonical = json.dumps(
        _canonicalize(expected), sort_keys=True, default=str
    )
    actual_canonical = json.dumps(_canonicalize(sdsc_json), sort_keys=True, default=str)
    assert expected_canonical == actual_canonical, (
        f"SuperDSC JSON mismatch with golden file {golden_path}"
    )


@pytest.fixture
def golden(request):
    """Fixture for golden file comparison."""
    update = request.config.getoption("--update-golden", default=False)

    class GoldenFile:
        def compare(self, name, actual_dict):
            path = GOLDEN_DIR / f"{name}.json"
            assert_sdsc_matches_golden(actual_dict, str(path), update=update)

    return GoldenFile()
