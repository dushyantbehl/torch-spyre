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
Seam 4: Layout Assignment (FixedLayout -> FixedTiledLayout) tests.

Validates that SpyreKernel.create_op_spec() produces OpSpecs with correct
device sizes and dtypes for input and output tensor args. Tests call
the real create_op_spec() method with a mock SchedulerNode — no
torch.compile or Spyre hardware needed.
"""

from torch_spyre._C import DataFormats

from tests.stages.helpers import generate_mock_op_spec


def test_pointwise_output_inherits_input_layout():
    """Pointwise output should have the same device layout as its input."""
    spec = generate_mock_op_spec(
        "add",
        False,
        [64, 256],
        [],
        DataFormats.SEN169_FP16,
        2,
    )

    input_arg = [a for a in spec.args if a.is_input][0]
    output_arg = [a for a in spec.args if not a.is_input][0]
    assert input_arg.device_size == output_arg.device_size
    assert input_arg.device_dtype == output_arg.device_dtype


def test_matmul_output_layout_derived_from_m_n():
    """mm(M*K, K*N) output layout should reflect M*N dimensions.

    SpyreTensorLayout converts logical dims to stick-format device sizes.
    We verify that create_op_spec produces the correct dtype and that
    the iteration space includes both output and reduction dimensions.
    """
    spec = generate_mock_op_spec(
        "matmul",
        True,
        [64, 128],
        [256],
        DataFormats.SEN169_FP16,
        2,
    )

    # All args share the same dtype
    for a in spec.args:
        assert a.device_dtype == DataFormats.SEN169_FP16

    # iteration_space includes output dims (i0, i1) and reduction dim (r0)
    assert len(spec.iteration_space) == 3
