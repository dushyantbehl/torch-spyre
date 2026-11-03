# Copyright 2026 The Torch-Spyre Authors.
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

"""Reading reference tensors, and comparing a launch result against them.

No device and no compile: both halves are host-side, which is why they are
library functions rather than CLI code.
"""

import pytest

torch = pytest.importorskip("torch")

from spyre_cli.io import (  # noqa: E402
    check_values,
    describe_match,
    load_tensors,
    tolerances_for,
)

SHAPE = [10, 512]


def _arg(arg_index=0, role="input", dtype="float16", shape=None):
    from torch_spyre._inductor.codegen.kernel_launchspec import (
        LaunchArg,
        LaunchArgLayout,
    )

    return LaunchArg(
        arg_index=arg_index,
        role=role,
        shape=list(SHAPE if shape is None else shape),
        dtype=dtype,
        layout=LaunchArgLayout(
            device_size=[8, 10, 64],
            stride_map=[64, 512, 1],
            device_dtype="SEN169_FP16",
            element_arrangement="STANDARD",
        ),
    )


# --------------------------------------------------------------------------
# Tolerances come from the dtype
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,want",
    [
        (torch.float64, (1e-7, 1e-7)),
        (torch.float32, (1.3e-6, 1e-5)),
        (torch.float16, (1e-3, 1e-5)),
        (torch.bfloat16, (1.6e-2, 1e-5)),
    ],
)
def test_float_tolerances_match_torch_testing(dtype, want):
    """Reused rather than invented, so "close" means the same here as in tree."""
    assert tolerances_for(dtype) == want


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.uint8])
def test_integers_and_bool_compare_exactly(dtype):
    assert tolerances_for(dtype) == (0.0, 0.0)


def test_a_specs_dtype_spelling_works_too():
    """A spec records "float16", not torch.float16."""
    assert tolerances_for("float16") == tolerances_for(torch.float16)


# --------------------------------------------------------------------------
# .pt is self-describing, so it is cross-checked
# --------------------------------------------------------------------------


def test_a_bare_tensor_loads(tmp_path):
    f = tmp_path / "a.pt"
    torch.save(torch.ones(SHAPE, dtype=torch.float16), f)
    loaded = load_tensors(f, [_arg()])
    assert len(loaded) == 1
    assert list(loaded[0].shape) == SHAPE


def test_the_opspec_lab_dict_loads(tmp_path):
    """The Lab writes {"tensors": [...]}, so a capture replays unconverted."""
    f = tmp_path / "lab.inputs.pt"
    torch.save({"tensors": [torch.ones(SHAPE, dtype=torch.float16)] * 2}, f)
    assert len(load_tensors(f, [_arg(0), _arg(1)])) == 2


def test_a_list_loads(tmp_path):
    f = tmp_path / "list.pt"
    torch.save([torch.ones(SHAPE, dtype=torch.float16)] * 3, f)
    assert len(load_tensors(f, [_arg(i) for i in range(3)])) == 3


def test_a_wrong_shape_is_an_error_not_a_reinterpretation(tmp_path):
    f = tmp_path / "wrong.pt"
    torch.save(torch.ones([512, 10], dtype=torch.float16), f)
    with pytest.raises(ValueError, match=r"expects shape \[10, 512\]"):
        load_tensors(f, [_arg()])


def test_a_wrong_dtype_is_an_error(tmp_path):
    f = tmp_path / "wrong_dtype.pt"
    torch.save(torch.ones(SHAPE, dtype=torch.float32), f)
    with pytest.raises(ValueError, match="expects dtype float16"):
        load_tensors(f, [_arg()])


def test_an_ambiguous_dict_is_refused(tmp_path):
    """Better to refuse than to guess which entry holds the arguments."""
    f = tmp_path / "ambiguous.pt"
    torch.save({"a": torch.ones(SHAPE), "b": torch.ones(SHAPE)}, f)
    with pytest.raises(ValueError, match="cannot tell which entry"):
        load_tensors(f, [_arg()])


# --------------------------------------------------------------------------
# .bin carries no metadata, so length is the only check
# --------------------------------------------------------------------------


def test_raw_bytes_are_interpreted_through_the_spec(tmp_path):
    f = tmp_path / "a.bin"
    values = torch.arange(10 * 512, dtype=torch.float16).reshape(SHAPE)
    f.write_bytes(values.numpy().tobytes())
    torch.testing.assert_close(load_tensors(f, [_arg()])[0], values)


def test_a_wrong_length_bin_is_refused(tmp_path):
    f = tmp_path / "short.bin"
    f.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="expected 10240 bytes"):
        load_tensors(f, [_arg()])


def test_a_bin_cannot_be_read_without_a_spec(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"\x00" * 10240)
    with pytest.raises(ValueError, match="needs a launch spec"):
        load_tensors(f, [])


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def test_an_identical_result_has_no_problems():
    want = torch.full(SHAPE, 2.0, dtype=torch.float16)
    assert check_values([want.clone()], [want]) == []


def test_a_difference_inside_the_dtype_tolerance_passes():
    """fp16's default rtol is 1e-3; one ULP at this magnitude is under it."""
    want = torch.full(SHAPE, 2.0, dtype=torch.float16)
    got = torch.full(SHAPE, 2.0 + 2**-9, dtype=torch.float16)
    assert check_values([got], [want]) == []


def test_a_real_difference_quotes_assert_close():
    """The message is most of the value, so it is not replaced by a worse one."""
    want = torch.ones(SHAPE, dtype=torch.float16)
    got = want.clone()
    got[0, 17] = 5.0
    problems = check_values([got], [want])
    assert len(problems) == 1
    assert "MISMATCH" in problems[0]
    assert "Mismatched elements: 1 / 5120" in problems[0]
    assert "Greatest absolute difference" in problems[0]


def test_exact_rejects_what_the_default_allows():
    want = torch.full(SHAPE, 2.0, dtype=torch.float16)
    got = torch.full(SHAPE, 2.0 + 2**-9, dtype=torch.float16)
    assert check_values([got], [want]) == []
    assert len(check_values([got], [want], exact=True)) == 1


def test_an_explicit_tolerance_overrides_the_default():
    want = torch.ones(SHAPE, dtype=torch.float16)
    got = want * 1.05
    assert len(check_values([got], [want])) == 1
    assert check_values([got], [want], rtol=0.1, atol=0.1) == []


def test_a_count_disagreement_is_reported_before_comparing():
    want = torch.ones(SHAPE, dtype=torch.float16)
    problems = check_values([want, want], [want])
    assert len(problems) == 1
    assert "expected 1 tensor(s) to compare, got 2" in problems[0]


def test_the_problem_names_the_argument():
    """A multi-output kernel has to say which output drifted."""
    want = torch.ones(SHAPE, dtype=torch.float16)
    bad = want.clone()
    bad[0, 0] = 9.0
    problems = check_values(
        [want.clone(), bad], [want, want], labels=["arg 7 (output)", "arg 9 (output)"]
    )
    assert len(problems) == 1
    assert problems[0].startswith("arg 9 (output)")


def test_the_success_line_reports_the_worst_difference():
    want = torch.full(SHAPE, 2.0, dtype=torch.float16)
    got = want.clone()
    got[0, 0] = 2.0 + 2**-9
    line = describe_match(got, want)
    assert "match" in line
    assert "max abs diff" in line
