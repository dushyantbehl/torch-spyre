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

"""
Shared helpers for compilation stage-wise tests.
"""

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

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"


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
    node.node.data = Mock(spec=Reduction if is_reduction else Pointwise)

    # create_op_spec() checks hasattr(node, "op_it_space_splits") — on a Mock
    # hasattr always returns True, so we must set it to a real dict to avoid
    # core_division.get() returning a Mock instead of an int.
    node.op_it_space_splits = {}

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
# SuperDSC JSON validators
# ---------------------------------------------------------------------------


def validate_sdsc_structure(sdsc):
    """Validate structural invariants of a SuperDSC dict.

    Raises AssertionError with details on failure.
    """
    errors = []

    try:
        json.dumps(sdsc, default=str)
    except TypeError as e:
        errors.append(f"Not JSON serializable: {e}")

    top_key = list(sdsc.keys())[0]
    inner = sdsc[top_key]
    if "numCoresUsed_" not in inner:
        errors.append("Missing 'numCoresUsed_' in top-level")
    if "dscs_" not in inner:
        errors.append("Missing 'dscs_' in top-level")
    elif not isinstance(inner["dscs_"], list) or len(inner["dscs_"]) == 0:
        errors.append("'dscs_' must be a non-empty list")

    assert not errors, "SuperDSC structural errors:\n" + "\n".join(errors)


def validate_sdsc_json_round_trip(sdsc):
    """Verify the SDSC dict survives JSON serialization."""
    json_str = json.dumps(sdsc, default=str)
    round_tripped = json.loads(json_str)
    assert round_tripped == json.loads(json_str)


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
