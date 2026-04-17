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
Seam 6: SpyreOpFuncs (Op Name Mapping) tests.

Pure Python — no hardware or _C module needed.
Tests that every SpyreOpFuncs static method returns the correct
PointwiseOp/ReductionOp with the expected op name and constants.
"""

import inspect

import pytest

from torch_spyre._inductor.spyre_kernel import (
    PointwiseOp,
    SpyreOpFuncs,
)

pytestmark = pytest.mark.no_device

# Each row: (method_name, args, expected_op, expected_type, expected_constants)
# expected_type is None for special cases (e.g. exx2 returns a string).
# expected_constants is a dict of {constant_name: value} or None.
_OP_FUNC_CASES = [
    ("abs", ["x"], "abs", PointwiseOp, None),
    ("add", ["a", "b"], "add", PointwiseOp, None),
    ("clamp", ["x", -1.0, 6.0], "clip", PointwiseOp, {"clipMin": -1.0, "clipMax": 6.0}),
    ("eq", ["a", "b"], "equal", PointwiseOp, None),
    ("exp", ["x"], "exp", PointwiseOp, None),
    # exx2 returns a formatted string, not a PointwiseOp — tested separately
    ("exx2", ["a", "b", "c"], None, None, None),
    ("ge", ["a", "b"], "greaterequal", PointwiseOp, None),
    ("gelu", ["x"], "gelufwd", PointwiseOp, None),
    ("gt", ["a", "b"], "greaterthan", PointwiseOp, None),
    ("layernormnorm", ["a", "b", "c"], "layernormnorm", PointwiseOp, None),
    ("layernormscale", ["x", 1e-5], "layernormscale", PointwiseOp, {"eps": 1e-5}),
    ("le", ["a", "b"], "lesserequal", PointwiseOp, None),
    ("log", ["x"], "log", PointwiseOp, None),
    ("lt", ["a", "b"], "lesserthan", PointwiseOp, None),
    ("mul", ["a", "b"], "mul", PointwiseOp, None),
    ("ne", ["a", "b"], "notequal", PointwiseOp, None),
    ("neg", ["x"], "neg", PointwiseOp, None),
    (
        "overwrite",
        ["input", [1, 2], [0, 0], [0, 0]],
        "overwrite",
        PointwiseOp,
        None,
    ),
    ("reciprocal", ["x"], "reciprocal", PointwiseOp, None),
    ("relu", ["x"], "relufwd", PointwiseOp, None),
    ("rsqrt", ["x"], "rsqrt", PointwiseOp, None),
    ("sigmoid", ["x"], "sigmoid", PointwiseOp, None),
    (
        "softplus",
        ["x", 1.0, 20.0],
        "softplus",
        PointwiseOp,
        {"softplusBeta": 1.0, "softplusThresh": 20.0},
    ),
    ("sqrt", ["x"], "sqrt", PointwiseOp, None),
    ("square", ["x"], "mul", PointwiseOp, None),
    ("sub", ["a", "b"], "sub", PointwiseOp, None),
    ("tanh", ["x"], "tanh", PointwiseOp, None),
    ("to_dtype", ["x", "fp16", "fp32"], "to_dtype", PointwiseOp, None),
    ("truediv", ["a", "b"], "realdiv", PointwiseOp, None),
    ("where", ["x", "y", "z"], "where3", PointwiseOp, None),
]


@pytest.mark.parametrize(
    "op_name,args,expected_op,expected_type,expected_constants",
    _OP_FUNC_CASES,
    ids=[c[0] for c in _OP_FUNC_CASES],
)
def test_op_func_mapping(op_name, args, expected_op, expected_type, expected_constants):
    """Every SpyreOpFuncs method should return the right type, op name,
    and constants."""
    fn = getattr(SpyreOpFuncs, op_name)
    result = fn(*args)

    if expected_type is None:
        # Special case: exx2 returns a formatted string
        assert isinstance(result, str)
        return

    assert isinstance(result, expected_type), (
        f"{op_name}: expected {expected_type.__name__}, got {type(result).__name__}"
    )
    assert result.op == expected_op, (
        f"{op_name}: expected op '{expected_op}', got '{result.op}'"
    )
    if expected_constants:
        for key, val in expected_constants.items():
            assert result.op_info["constants"][key] == val, (
                f"{op_name}: constant '{key}' expected {val}, "
                f"got {result.op_info['constants'].get(key)}"
            )


def test_opfunc_completeness():
    """Every public method in SpyreOpFuncs should be covered by test_op_func_mapping."""
    tested_ops = {case[0] for case in _OP_FUNC_CASES}
    public_ops = {
        name
        for name, _ in inspect.getmembers(SpyreOpFuncs, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    untested = public_ops - tested_ops
    assert not untested, f"Untested SpyreOpFuncs methods: {untested}"
