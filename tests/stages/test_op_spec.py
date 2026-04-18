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
Seam 7: OpSpec Construction tests.

Validates that SpyreKernel.create_op_spec() produces correct OpSpecs
and that parse_op_spec() converts them to valid SDSCSpecs.
Also checks FP32 whitelist enforcement.
"""

import pytest
import torch
from sympy import Symbol

from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.constants import SPYRE_FP32_OPS
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import TensorArg
from torch_spyre._inductor.spyre_kernel import SpyreKernel

from .helpers import generate_mock_op_spec, make_mock_node

pytestmark = pytest.mark.no_device


def test_op_spec_construction_and_parse():
    """Build OpSpec via generate_mock_op_spec(), validate structure,
    then parse to SDSCSpec."""
    spec = generate_mock_op_spec(
        "add",
        False,
        [64, 256],
        [],
        DataFormats.SEN169_FP16,
        2,
    )

    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    assert len(inputs) == 2 and len(outputs) == 1
    assert spec.is_reduction is False
    assert spec.op == "add"
    assert len(spec.iteration_space) == 2

    # Parse to SDSCSpec
    sdsc_spec = parse_op_spec(spec)
    assert sdsc_spec.opfunc == "add"
    assert sdsc_spec.execution_unit == "sfp"
    assert sdsc_spec.num_inputs == 2
    assert sdsc_spec.num_cores >= 1


def test_fp32_whitelist_enforcement():
    """FP32 is allowed only for ops in SPYRE_FP32_OPS; others must be rejected."""
    assert "add" in SPYRE_FP32_OPS  # accepted
    assert "gelu" not in SPYRE_FP32_OPS  # rejected

    # Verify create_op_spec rejects FP32 for non-whitelisted ops
    shape = [64, 256]
    layout = SpyreTensorLayout(list(shape), torch.float16)
    it_syms = [Symbol(f"i{d}") for d in range(len(shape))]

    fp32_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.IEEE_FP32,
        device_size=layout.device_size,
        device_coordinates=list(it_syms),
        allocation=None,
    )

    ranges_dict = {it_syms[d]: size for d, size in enumerate(shape)}
    mock_node = make_mock_node(is_reduction=False, ranges_dict=ranges_dict)

    kernel = SpyreKernel()
    kernel.current_node = mock_node

    with pytest.raises(Unsupported):
        kernel.create_op_spec("gelu", False, [fp32_arg], op_info={})
