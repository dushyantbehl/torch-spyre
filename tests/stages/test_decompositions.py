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

import pytest
import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

from torch_spyre._inductor.decompositions import spyre_decompositions

pytestmark = pytest.mark.no_device


def _decompose(fn, *args):
    """Run fn through make_fx with Spyre decompositions and return op targets."""
    decomps = get_decompositions([])
    decomps.update(spyre_decompositions)
    gm = make_fx(fn, decomposition_table=decomps)(*args)
    targets = set()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            if callable(n.target):
                targets.add(getattr(n.target, "__name__", str(n.target)))
            else:
                targets.add(n.target)
    return targets


@pytest.mark.parametrize(
    "desc,fn,args,expected_present,expected_absent",
    [
        (
            "layer_norm -> native_layer_norm",
            lambda x, w, b: torch.nn.functional.layer_norm(x, [256], w, b),
            (torch.randn(4, 256), torch.randn(256), torch.randn(256)),
            {"native_layer_norm.default"},
            {"layer_norm.default"},
        ),
        (
            "addmm -> mm + add",
            lambda bias, x, w: torch.addmm(bias, x, w),
            (torch.randn(128), torch.randn(64, 256), torch.randn(256, 128)),
            {"mm.default"},
            {"addmm.default"},
        ),
    ],
)
def test_decomposition(desc, fn, args, expected_present, expected_absent):
    """Each registered decomposition should replace the source op with its
    target ops."""
    op_names = _decompose(fn, *args)
    for op in expected_present:
        assert op in op_names, f"{op} missing after decomposition ({desc})"
    for op in expected_absent:
        assert op not in op_names, f"{op} should have been decomposed away ({desc})"


def test_passthrough_ops_have_no_decomposition():
    """Ops without Spyre decompositions (e.g., add, mul) should not appear
    in the decomposition table."""
    for op in [torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor]:
        assert op not in spyre_decompositions
