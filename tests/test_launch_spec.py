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

"""Launch spec round-trip and validation.

The two cases that motivate the whole file are ``test_catches_transposed_shape``
and ``test_catches_wrong_dtype``: both of those launches previously exited 0 and
returned wrong data on hardware.

No device needed -- these exercise the spec, not a kernel.
"""

import json
import os

import pytest
import torch

from torch_spyre.execution.kernel_cache import (
    LAUNCH_SPEC_FILE,
    LAUNCH_SPEC_VERSION,
    check_launch_spec,
    load_launch_spec,
    save_launch_spec,
)


def _spec(**overrides):
    """The spec for a [10, 512] fp16 add: two inputs, one output, no pool."""
    spec = {
        "version": LAUNCH_SPEC_VERSION,
        "kernel_name": "sdsc_fused_add_0",
        "pool_size": 0,
        "bundle_symbolic_args": True,
        "emitter": "sdsc",
        "args": [
            {
                "arg_index": i,
                "role": "input" if i < 2 else "output",
                "shape": [10, 512],
                "dtype": "float16",
            }
            for i in range(3)
        ],
    }
    spec.update(overrides)
    return spec


def _tensors(shapes=None, dtypes=None):
    shapes = shapes or [[10, 512]] * 3
    dtypes = dtypes or [torch.float16] * 3
    return [torch.empty(s, dtype=d) for s, d in zip(shapes, dtypes)]


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    spec = _spec()
    save_launch_spec(str(tmp_path), spec)
    assert load_launch_spec(str(tmp_path)) == spec


def test_written_beside_the_folder_not_inside_it(tmp_path):
    """prepare_kernel owns spyreCodeDir/; the spec is its sibling."""
    os.makedirs(tmp_path / "spyreCodeDir")
    save_launch_spec(str(tmp_path), _spec())
    assert (tmp_path / LAUNCH_SPEC_FILE).is_file()
    assert not (tmp_path / "spyreCodeDir" / LAUNCH_SPEC_FILE).exists()


def test_absent_spec_is_not_an_error(tmp_path):
    """Folders compiled before this existed simply have no spec."""
    assert load_launch_spec(str(tmp_path)) is None


def test_unreadable_spec_raises(tmp_path):
    """Present but broken is an error: ignoring it resumes guessing."""
    (tmp_path / LAUNCH_SPEC_FILE).write_text("{not json")
    with pytest.raises(RuntimeError, match="could not read launch spec"):
        load_launch_spec(str(tmp_path))


def test_future_major_version_is_refused(tmp_path):
    save_launch_spec(str(tmp_path), _spec(version=LAUNCH_SPEC_VERSION + 1))
    with pytest.raises(RuntimeError, match="newer than this build"):
        load_launch_spec(str(tmp_path))


def test_missing_version_is_refused(tmp_path):
    spec = _spec()
    del spec["version"]
    (tmp_path / LAUNCH_SPEC_FILE).write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="no 'version'"):
        load_launch_spec(str(tmp_path))


# --------------------------------------------------------------------------
# Validation — the reproduced failures
# --------------------------------------------------------------------------


def test_correct_launch_has_no_problems():
    assert check_launch_spec(_spec(), _tensors()) == []


def test_catches_transposed_shape():
    """Previously: launched, exit 0, wrong data."""
    problems = check_launch_spec(
        _spec(), _tensors(shapes=[[512, 10], [10, 512], [10, 512]])
    )
    assert len(problems) == 1
    assert "expected shape [10, 512], got [512, 10]" in problems[0]
    assert "transposed" in problems[0]


def test_catches_wrong_dtype():
    """Previously: launched, exit 0, wrong data."""
    problems = check_launch_spec(
        _spec(), _tensors(dtypes=[torch.float32, torch.float16, torch.float16])
    )
    assert len(problems) == 1
    assert "expected dtype float16, got float32" in problems[0]


def test_catches_wrong_count():
    problems = check_launch_spec(_spec(), _tensors()[:2])
    assert problems == ["expected 3 tensors, got 2"]


def test_reports_every_problem_not_just_the_first():
    problems = check_launch_spec(
        _spec(),
        _tensors(
            shapes=[[512, 10], [10, 512], [7, 7]],
            dtypes=[torch.float32, torch.float16, torch.float16],
        ),
    )
    assert len(problems) == 3


# --------------------------------------------------------------------------
# Pool accounting — pool_size is the CALLER's tensor, not the kernel's pool
# --------------------------------------------------------------------------


def test_pooled_kernel_expects_a_prepended_pool_tensor():
    problems = check_launch_spec(_spec(pool_size=32768), _tensors())
    assert len(problems) == 1
    assert "expected 4 tensors" in problems[0]
    assert "caller-supplied pool tensor of 32768 bytes" in problems[0]


def test_pooled_kernel_accepts_the_pool_tensor():
    pool = torch.empty(32768, dtype=torch.uint8)
    assert check_launch_spec(_spec(pool_size=32768), [pool] + _tensors()) == []


def test_pool_tensor_of_the_wrong_size_is_caught():
    pool = torch.empty(4096, dtype=torch.uint8)
    problems = check_launch_spec(_spec(pool_size=32768), [pool] + _tensors())
    assert len(problems) == 1
    assert "expected 32768 bytes, got 4096" in problems[0]


def test_pool_offset_does_not_shift_arg_checks():
    """With a pool prepended, args must still be read at arg_index + 1."""
    pool = torch.empty(32768, dtype=torch.uint8)
    tensors = [pool] + _tensors(shapes=[[10, 512], [10, 512], [512, 10]])
    problems = check_launch_spec(_spec(pool_size=32768), tensors)
    assert len(problems) == 1
    assert problems[0].startswith("arg 2 (output)")


# --------------------------------------------------------------------------
# Symbolic dimensions
# --------------------------------------------------------------------------


def _symbolic_spec():
    spec = _spec(symbols={"s0": {}})
    for arg in spec["args"]:
        arg["shape"] = ["s0", 512]
    return spec


def test_unbound_symbol_is_reported_not_guessed():
    problems = check_launch_spec(_symbolic_spec(), _tensors())
    assert len(problems) == 3
    assert "symbolic and unbound" in problems[0]


def test_bound_symbol_validates_against_the_binding():
    assert check_launch_spec(_symbolic_spec(), _tensors(), {"s0": 10}) == []


def test_binding_to_the_wrong_extent_is_caught():
    problems = check_launch_spec(_symbolic_spec(), _tensors(), {"s0": 64})
    assert len(problems) == 3
    assert "expected shape [64, 512], got [10, 512]" in problems[0]


# --------------------------------------------------------------------------
# Layout — a non-default packing along the sticks
#
# The motivating case is a depthwise conv2d, compiled against a layout that puts
# the 64 channels in one stick. Shape and dtype match the default arrangement
# exactly, so nothing but the layout distinguishes a correct launch from one that
# exits 0 with wrong data.
# --------------------------------------------------------------------------


_DWCONV_LAYOUT = {
    "device_size": [32, 32, 1, 1, 64],
    "stride_map": [1, 32, -1, 65536, 1024],
    "device_dtype": "SEN169_FP16",
    "element_arrangement": "STANDARD",
}


def test_layout_round_trips_through_the_spec_file(tmp_path):
    spec = _spec()
    spec["args"][0]["layout"] = dict(_DWCONV_LAYOUT)
    save_launch_spec(str(tmp_path), spec)
    back = load_launch_spec(str(tmp_path))
    assert back["args"][0]["layout"] == _DWCONV_LAYOUT


def test_spyre_layout_from_spec_builds_the_recorded_layout():
    from torch_spyre.execution.kernel_cache import spyre_layout_from_spec

    layout = spyre_layout_from_spec(_DWCONV_LAYOUT)
    assert layout is not None
    assert list(layout.device_size) == _DWCONV_LAYOUT["device_size"]
    assert list(layout.stride_map) == _DWCONV_LAYOUT["stride_map"]


def test_spyre_layout_from_spec_returns_none_for_an_unusable_block():
    """An older spec, or an enum spelling this build does not know."""
    from torch_spyre.execution.kernel_cache import spyre_layout_from_spec

    assert spyre_layout_from_spec({"device_size": [1]}) is None  # no stride_map
    assert spyre_layout_from_spec(
        {**_DWCONV_LAYOUT, "device_dtype": "NOT_A_FORMAT"}
    ) is None


def test_a_spec_without_layout_is_not_layout_checked():
    """Older specs still validate on shape and dtype alone."""
    spec = _spec()
    for arg in spec["args"]:
        arg.pop("layout", None)
    assert check_launch_spec(spec, _tensors()) == []
