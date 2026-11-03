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

"""Reading reference tensors, and checking a launch result against them.

``check_kernelspec`` says a launch was well-formed; this says the answer was
right. ``torch.ones`` inputs tell you nothing about numerics -- an add of ones
gives 2.0 everywhere, so any elementwise op touching both inputs looks
plausible, and a transpose or reduction-axis bug is invisible when every
element is identical. Supplying real values and a reference is what turns
``spyre launch`` from "did it run" into "is it right".

Both halves live here rather than in the CLI because neither needs a terminal:
the reader needs the spec to interpret a ``.bin``, the comparison returns a
list of problems, and both are testable with no device and no compile.
"""

import os
from typing import Optional, Sequence

import torch

# torch.testing's own per-dtype defaults, reused rather than invented: a result
# this tool calls close is then one the rest of the test suite would also call
# close, and the maintenance is inherited. A tolerance meaningful for fp32 is
# meaningless for fp16, which is why this is a table and not a number.
_TOLERANCES = {
    "float64": (1e-7, 1e-7),
    "float32": (1.3e-6, 1e-5),
    "float16": (1e-3, 1e-5),
    "bfloat16": (1.6e-2, 1e-5),
}


def tolerances_for(dtype) -> tuple[float, float]:
    """``(rtol, atol)`` for ``dtype``; exact for integers and bool.

    Takes a torch dtype or its spelling without the ``torch.`` prefix, which
    is how a spec records it.
    """
    return _TOLERANCES.get(str(dtype).removeprefix("torch."), (0.0, 0.0))


def _tensors_in(obj) -> list:
    """Every tensor in a loaded ``.pt``, in file order.

    Several shapes are accepted because the OpSpec Lab does not write a bare
    tensor: it writes ``{"tensors": [...]}``, one entry per argument. A
    captured kernel then replays with no conversion, and one ``-i`` can supply
    several arguments.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        if "tensors" in obj:
            return _tensors_in(obj["tensors"])
        values = list(obj.values())
        if len(values) == 1:
            # The Lab keys on a script path, so the one value is unambiguous.
            return _tensors_in(values[0])
        raise ValueError(
            f"cannot tell which entry holds the tensors: keys {sorted(obj)[:4]}"
        )
    if isinstance(obj, (list, tuple)):
        found = []
        for item in obj:
            found.extend(_tensors_in(item))
        return found
    raise ValueError(f"expected a tensor, a list of them, or a dict, got {type(obj)}")


def load_tensors(path, args, bindings: Optional[dict] = None) -> list:
    """Host tensors from ``path``, for the spec arguments ``args``.

    ``.pt`` goes through ``torch.load`` and is self-describing, so its shape
    and dtype are cross-checked against the spec: a disagreement is an error
    rather than a reinterpretation.

    ``.bin`` is raw little-endian row-major element data, with shape and dtype
    taken from the spec. Length is the only check the format permits, so a
    file of the right size but the wrong dtype is undetectable -- the reason
    ``.pt`` is preferred, and why a folder with no spec cannot use ``.bin`` at
    all.

    Returns as many tensors as the file holds, so the caller matches them to
    ``args`` positionally.
    """
    path = str(path)

    if path.endswith(".bin"):
        if not args:
            raise ValueError(
                f"{path}: a .bin carries no shape or dtype, so it needs a launch "
                "spec to interpret it"
            )
        arg = args[0]
        shape = arg.resolved_shape(bindings)
        dtype = getattr(torch, arg.dtype)
        want = torch.empty((), dtype=dtype).element_size()
        for dim in shape:
            want *= dim
        got = os.path.getsize(path)
        if got != want:
            raise ValueError(
                f"{path}: expected {want} bytes for {shape} {arg.dtype}, got {got}"
            )
        with open(path, "rb") as f:
            raw = bytearray(f.read())
        return [torch.frombuffer(raw, dtype=dtype).reshape(shape)]

    # weights_only: a .pt is arbitrary pickle, and these come from wherever the
    # caller got them.
    found = _tensors_in(torch.load(path, map_location="cpu", weights_only=True))

    for tensor, arg in zip(found, args):
        want_shape = arg.resolved_shape(bindings)
        if list(tensor.shape) != want_shape:
            raise ValueError(
                f"{path}: arg {arg.arg_index} expects shape {want_shape}, "
                f"file has {list(tensor.shape)}"
            )
        got_dtype = str(tensor.dtype).removeprefix("torch.")
        if got_dtype != arg.dtype:
            raise ValueError(
                f"{path}: arg {arg.arg_index} expects dtype {arg.dtype}, "
                f"file has {got_dtype}"
            )
    return found


def check_values(
    actual: Sequence,
    expected: Sequence,
    labels: Optional[Sequence[str]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    exact: bool = False,
) -> list[str]:
    """Return the ways ``actual`` differs from ``expected``; empty means agreement.

    A sibling of ``check_kernelspec``: the same contract, a list of
    human-readable problems, so the caller decides whether to print it, raise
    on it, or ignore it.

    The comparison is ``torch.testing.assert_close``, for its failure message
    -- how many elements differ, the worst absolute and relative difference,
    and where. That message is most of the value here, and a hand-rolled
    ``allclose`` would report less.

    Tolerances default to the expected tensor's own dtype
    (:func:`tolerances_for`); ``exact`` demands bitwise equality. The relative
    term is scaled by ``expected``, so the reference is always passed there.
    """
    if len(actual) != len(expected):
        return [f"expected {len(expected)} tensor(s) to compare, got {len(actual)}"]

    problems = []
    for i, (got, want) in enumerate(zip(actual, expected)):
        where = labels[i] if labels and i < len(labels) else f"output {i}"
        if exact:
            use_rtol, use_atol = 0.0, 0.0
        else:
            default_rtol, default_atol = tolerances_for(want.dtype)
            use_rtol = default_rtol if rtol is None else rtol
            use_atol = default_atol if atol is None else atol
        try:
            torch.testing.assert_close(got, want, rtol=use_rtol, atol=use_atol)
        except AssertionError as e:
            detail = "\n".join("  " + line for line in str(e).strip().splitlines())
            problems.append(f"{where}: MISMATCH\n{detail}")
    return problems


def describe_match(actual, expected, rtol: Optional[float] = None) -> str:
    """The one-line success report for a matching output."""
    shown = tolerances_for(expected.dtype)[0] if rtol is None else rtol
    diff = (actual.float() - expected.float()).abs().max().item()
    return f"match  (max abs diff {diff:.2e}, rtol {shown:g})"
