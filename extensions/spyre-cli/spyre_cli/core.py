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

from pathlib import Path


def _launch(path, tensors):
    """Launch the bundle and return the runner.

    The returned runner MUST be kept alive until the outputs have been read
    back. Dropping it frees the JobPlan, and with it the device allocation and
    pinned buffers that the still in-flight launch is reading.
    """
    # delayed import, for easy testing
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    # this is the magic line
    # should have already compiled at this point
    runner = SpyreSDSCKernelRunner("spyre-cli", str(path))
    runner.run(*tensors)
    return runner


def launch_from_cli(
    path,
    inputs,
    outputs,
    bindings=None,
    rtol=None,
    atol=None,
    exact=False,
    save_output=None,
):
    """Launch ``path``, verifying the result when a reference was given.

    With no files, every tensor comes from the folder's spec and the outputs
    are printed -- "did it run". Given ``-i`` files those values are used, and
    given ``-o`` files as well the result is compared against them -- "is it
    right".

    Returns True when every output matched, or when there was nothing to
    compare; False on a mismatch. A malformed invocation raises instead, since
    that is a different failure and a script has to tell them apart.
    """
    path = Path(path)
    import torch

    # Deferred, like the other torch-spyre imports here: these import torch at
    # module level, and `spyre --help` must work without it.
    from .io import check_values, describe_match, load_tensors
    from .launch_spec import _tensors_from_spec, check_kernelspec, load_kernelspec

    bindings = bindings or {}
    spec = load_kernelspec(path)

    if spec is None:
        # Everything here is addressed through the spec: which arguments exist,
        # what order they bind in, and how to interpret a .bin. Without one
        # there is nothing to launch against.
        raise ValueError(
            f"{path} has no launch_spec.json, so its arguments are unknown. "
            "Recompile to get a folder that describes itself."
        )

    in_args = [a for a in spec.args if a.role != "output"]
    out_args = [a for a in spec.args if a.role != "input"]
    # Position in the tensor list, which the pool shifts when the kernel takes
    # one. By role, not by index: an output need not be the trailing argument.
    out_positions = [a.arg_index + spec.arg_offset for a in out_args]

    expected = []
    for f in outputs:
        expected.extend(load_tensors(f, out_args[len(expected) :], bindings))
    if expected and len(expected) != len(out_args):
        raise ValueError(
            f"{path} has {len(out_args)} output(s), but {len(outputs)} file(s) "
            f"gave {len(expected)} expected tensor(s)"
        )

    if inputs:
        values = []
        for f in inputs:
            values.extend(load_tensors(f, in_args[len(values) :], bindings))
        if len(values) != len(in_args):
            raise ValueError(
                f"{path} takes {len(in_args)} input(s), but {len(inputs)} file(s) "
                f"gave {len(values)}"
            )
        tensors = _tensors_from_spec(spec, bindings, values=values)
    else:
        # No values given: ones for inputs, which says nothing about numerics
        # but does say whether the folder loads and runs.
        tensors = _tensors_from_spec(spec, bindings)

    print(f"using {spec.kernel_name} launch spec from {path}")

    problems = check_kernelspec(spec, tensors, bindings)
    if problems:
        raise ValueError(
            "the tensors do not match what this kernel expects:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    runner = _launch(path, tensors)
    # Reading every output back is also what makes a failed launch visible,
    # since .cpu() is where a dead kernel surfaces.
    actual = [tensors[pos].cpu() for pos in out_positions]
    del runner

    if save_output:
        torch.save(actual[0] if len(actual) == 1 else actual, save_output)
        print(f"wrote {save_output}")

    labels = [f"arg {a.arg_index} ({a.role})" for a in out_args]

    if not expected:
        for label, got in zip(labels, actual):
            if len(actual) > 1:
                print(f"{label}:")
            print(got)
        return True

    problems = check_values(
        actual, expected, labels=labels, rtol=rtol, atol=atol, exact=exact
    )
    for p in problems:
        print(p)
    for label, got, want in zip(labels, actual, expected):
        if not any(p.startswith(label) for p in problems):
            print(f"{label}: {describe_match(got, want, rtol)}")
    print(f"{len(expected) - len(problems)} output verified, {len(problems)} failed")
    return not problems


def launch(*tensors, path="."):
    """Launch a bundle; returns the runner, which the caller MUST keep alive.

    Bind the result (``runner = launch(a, b, c)``) and keep it in scope until
    the outputs have been read back. Calling this as a bare statement frees the
    JobPlan while the launch is still in flight, and the device then faults
    with what looks like a hardware error.
    """
    path = Path(path)
    return _launch(path, tensors)
