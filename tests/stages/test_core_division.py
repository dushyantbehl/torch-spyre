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
Seam 5: Core Division Planning tests.

Pure math — no hardware or _C module needed.
"""

import pytest
from sympy import Symbol

from torch_spyre._inductor.core_division import (
    core_split,
    multi_dim_iteration_space_split,
)


@pytest.mark.parametrize(
    "size,max_cores,expected",
    [
        (32, 32, 32),  # exact fit
        (64, 32, 32),  # clamp to max
        (16, 32, 16),  # smaller than max
        (7, 32, 7),  # prime, fits in max
        (12, 32, 12),  # composite
        (100, 32, 25),  # largest divisor <= 32
        (7, 4, 1),  # prime exceeds max -> fallback to 1
    ],
)
def test_core_split(size, max_cores, expected):
    """core_split should return the largest divisor of size that is <= max_cores."""
    assert core_split(size, max_cores) == expected


@pytest.mark.parametrize(
    "desc,it_space,max_cores,priorities,min_splits,checks",
    [
        (
            "single dim",
            {"i": 128},
            32,
            ["i"],
            {},
            {"i_le": 32, "i_divides": 128},
        ),
        (
            "two dims, j priority",
            {"i": 64, "j": 256},
            32,
            ["j", "i"],
            {},
            {"total_le": 32},
        ),
        (
            "min_splits respected",
            {"i": 64, "j": 256},
            32,
            ["j"],
            {"i": 2},
            {"i_ge": 2, "total_le": 32},
        ),
    ],
)
def test_multi_dim_iteration_space_split(
    desc, it_space, max_cores, priorities, min_splits, checks
):
    """multi_dim_iteration_space_split should respect max_cores, priorities,
    and min_splits."""
    syms = {name: Symbol(name) for name in it_space}
    result = multi_dim_iteration_space_split(
        iteration_space={syms[k]: v for k, v in it_space.items()},
        max_cores=max_cores,
        priorities=[syms[k] for k in priorities],
        **(
            {"min_splits": {syms[k]: v for k, v in min_splits.items()}}
            if min_splits
            else {}
        ),
    )
    total = 1
    for sym in syms.values():
        total *= result[sym]
    if "total_le" in checks:
        assert total <= checks["total_le"], (
            f"total cores {total} > {checks['total_le']} ({desc})"
        )
    for name, sym in syms.items():
        if f"{name}_le" in checks:
            assert result[sym] <= checks[f"{name}_le"]
        if f"{name}_ge" in checks:
            assert result[sym] >= checks[f"{name}_ge"]
        if f"{name}_divides" in checks:
            assert checks[f"{name}_divides"] % result[sym] == 0
