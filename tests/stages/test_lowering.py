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
Seam 3: Lowering (FX Graph -> LoopLevelIR) tests.

Validates that SpyreKernel.create_op_spec() produces correct OpSpecs
for different op types. Tests call the real create_op_spec() method
with a mock SchedulerNode — no torch.compile or Spyre hardware needed.
"""

import pytest

from torch_spyre._C import DataFormats

from tests.stages.helpers import generate_mock_op_spec

pytestmark = pytest.mark.no_device


@pytest.mark.parametrize(
    "desc,op_name,is_reduction,output_shape,reduction_shape,expected_num_args",
    [
        (
            "mm -> reduction matmul",
            "matmul",
            True,
            [64, 128],
            [256],
            3,  # 2 inputs + 1 output
        ),
        (
            "add -> pointwise add",
            "add",
            False,
            [64, 256],
            [],
            3,  # 2 inputs + 1 output
        ),
    ],
)
def test_lowering(
    desc, op_name, is_reduction, output_shape, reduction_shape, expected_num_args
):
    """SpyreKernel.create_op_spec() should produce an OpSpec with the correct
    op name, reduction flag, iteration space, and arg count."""
    num_inputs = expected_num_args - 1
    op_spec = generate_mock_op_spec(
        op_name,
        is_reduction,
        output_shape,
        reduction_shape,
        DataFormats.SEN169_FP16,
        num_inputs,
    )

    assert op_spec.op == op_name, (
        f"Expected op '{op_name}', got '{op_spec.op}' ({desc})"
    )
    assert op_spec.is_reduction is is_reduction, (
        f"Expected is_reduction={is_reduction} for {op_name} ({desc})"
    )
    assert len(op_spec.args) == expected_num_args, (
        f"Expected {expected_num_args} args, got {len(op_spec.args)} ({desc})"
    )
    # Verify iteration_space has correct number of dimensions
    expected_dims = len(output_shape) + len(reduction_shape)
    assert len(op_spec.iteration_space) == expected_dims, (
        f"Expected {expected_dims} iteration_space dims ({desc})"
    )
