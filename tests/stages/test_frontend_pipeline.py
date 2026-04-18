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
Seam 9: Full Front-End Pipeline (OpSpec -> SDSCSpec -> SuperDSC JSON) tests.

Validates the front-end codegen pipeline by using SpyreKernel.create_op_spec()
to build OpSpecs from a mock SchedulerNode, then running them through
parse_op_spec + generate_sdsc, and checking that the output is structurally
valid. No torch.compile or Spyre hardware needed.
"""

import pytest

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
from torch_spyre._inductor.codegen.superdsc import parse_op_spec

from .helpers import (
    GOLDEN_DIR,
    assert_sdsc_matches_golden,
    generate_mock_op_spec,
    validate_sdsc_structure,
)

pytestmark = pytest.mark.no_device


@pytest.mark.parametrize(
    "desc,op_name,is_reduction,output_shape,reduction_shape,num_inputs,expected_ops",
    [
        (
            "pointwise add",
            "add",
            False,
            [64, 256],
            [],
            2,
            {"add"},
        ),
        (
            "matmul",
            "matmul",
            True,
            [128, 64],
            [256],
            2,
            {"matmul"},
        ),
        (
            "pointwise relu (unary)",
            "relufwd",
            False,
            [4, 256],
            [],
            1,
            {"relufwd"},
        ),
    ],
)
def test_full_pipeline(
    desc, op_name, is_reduction, output_shape, reduction_shape, num_inputs, expected_ops
):
    """SpyreKernel.create_op_spec -> parse_op_spec -> generate_sdsc -> validate."""
    op_spec = generate_mock_op_spec(
        op_name,
        is_reduction,
        output_shape,
        reduction_shape,
        DataFormats.SEN169_FP16,
        num_inputs,
    )

    assert op_spec.op in expected_ops, (
        f"OpSpec op '{op_spec.op}' not in {expected_ops} ({desc})"
    )

    sdsc_spec = parse_op_spec(op_spec)
    sdsc_json = generate_sdsc(sdsc_spec)
    validate_sdsc_structure(sdsc_json)


@pytest.mark.parametrize(
    "desc,op_name,is_reduction,output_shape,reduction_shape,num_inputs,golden_name",
    [
        (
            "add pipeline golden",
            "add",
            False,
            [64, 256],
            [],
            2,
            "pipeline_add_64x256",
        ),
        (
            "matmul pipeline golden",
            "matmul",
            True,
            [128, 64],
            [256],
            2,
            "pipeline_matmul_128x256x64",
        ),
    ],
)
def test_full_pipeline_golden(
    desc, op_name, is_reduction, output_shape, reduction_shape, num_inputs, golden_name
):
    """Regression: full pipeline SDSC JSON must match golden files."""
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
