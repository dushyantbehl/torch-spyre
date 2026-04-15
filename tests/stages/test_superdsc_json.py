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
Seam 8: SuperDSC JSON Generation tests.

This is the highest-value testing seam. Validates that
SpyreKernel.create_op_spec() + parse_op_spec() + generate_sdsc()
produce structurally correct SuperDSC JSON for each operation type,
and that outputs match golden reference files.
"""

from math import prod

import pytest
import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
from torch_spyre._inductor.codegen.superdsc import parse_op_spec

from tests.stages.helpers import (
    GOLDEN_DIR,
    assert_sdsc_matches_golden,
    generate_mock_op_spec,
    validate_sdsc_json_round_trip,
    validate_sdsc_structure,
)


@pytest.mark.parametrize(
    "desc,op_name,is_reduction,output_shape,reduction_shape,num_inputs,expected_unit",
    [
        (
            "pointwise add",
            "add",
            False,
            [512, 256],
            [],
            2,
            "sfp",
        ),
        (
            "matmul",
            "matmul",
            True,
            [128, 64],
            [256],
            2,
            "pt",
        ),
    ],
)
def test_generate_sdsc(
    desc,
    op_name,
    is_reduction,
    output_shape,
    reduction_shape,
    num_inputs,
    expected_unit,
):
    """Generate SuperDSC JSON for each op type and validate structure,
    SDSCSpec properties, and SuperDSC JSON contents."""
    op_spec = generate_mock_op_spec(
        op_name,
        is_reduction,
        output_shape,
        reduction_shape,
        DataFormats.SEN169_FP16,
        num_inputs,
    )
    sdsc_spec = parse_op_spec(op_spec)
    sdsc = generate_sdsc(sdsc_spec)

    # --- SDSCSpec checks ---
    assert sdsc_spec.execution_unit == expected_unit, f"wrong unit for {desc}"
    assert sdsc_spec.opfunc == op_name, f"wrong opfunc for {desc}"
    assert sdsc_spec.data_format == DataFormats.SEN169_FP16
    assert sdsc_spec.num_cores >= 1
    assert sdsc_spec.num_inputs == num_inputs

    # iteration_space keys are Symbols with positive values
    for sym, val in sdsc_spec.iteration_space.items():
        assert isinstance(sym, sympy.Symbol)
        assert int(val) > 0

    # num_cores equals product of work_slices
    assert sdsc_spec.num_cores == prod(sdsc_spec.work_slices.values())

    # layouts have required keys
    for label, layout_info in sdsc_spec.layouts.items():
        assert "dim_order" in layout_info
        assert "stick_dim_order" in layout_info
        assert "stick_size" in layout_info

    # --- SuperDSC JSON checks ---
    validate_sdsc_structure(sdsc)
    validate_sdsc_json_round_trip(sdsc)

    top_key = list(sdsc.keys())[0]
    inner = sdsc[top_key]

    # numCoresUsed_ matches sdsc_spec
    assert inner["numCoresUsed_"] == sdsc_spec.num_cores

    # coreIdToWkSlice_ has one entry per core
    assert len(inner["coreIdToWkSlice_"]) == sdsc_spec.num_cores

    # dscs_ inner structure has computeOp_ with correct unit and opfunc
    dsc_inner = inner["dscs_"][0][top_key]
    compute_op = dsc_inner["computeOp_"][0]
    assert compute_op["exUnit"] == expected_unit
    assert compute_op["opFuncName"] == op_name

    # labeledDs_ length matches total number of args
    assert len(dsc_inner["labeledDs_"]) == num_inputs + 1

    # scheduleTree_ length matches total number of args
    assert len(dsc_inner["scheduleTree_"]) == num_inputs + 1


@pytest.mark.parametrize(
    "desc,op_name,is_reduction,output_shape,reduction_shape,num_inputs,golden_name",
    [
        (
            "add 512x256",
            "add",
            False,
            [512, 256],
            [],
            2,
            "add_512x256",
        ),
        (
            "matmul 128x64x256",
            "matmul",
            True,
            [128, 64],
            [256],
            2,
            "matmul_128x64x256",
        ),
    ],
)
def test_sdsc_golden(
    desc, op_name, is_reduction, output_shape, reduction_shape, num_inputs, golden_name
):
    """Regression: generated SuperDSC JSON must match the checked-in golden file."""
    op_spec = generate_mock_op_spec(
        op_name,
        is_reduction,
        output_shape,
        reduction_shape,
        DataFormats.SEN169_FP16,
        num_inputs,
    )
    sdsc = generate_sdsc(parse_op_spec(op_spec))
    assert_sdsc_matches_golden(sdsc, GOLDEN_DIR / f"{golden_name}.json")
