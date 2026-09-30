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


dtype_mapping = {
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
}


def create_tensor_info(tinfo):
    """
    Parses strings of type: "10x1024@fp16".
    """

    # defaults
    dtype = "float16"
    dims = ""

    parts = tinfo.split("@")

    if len(parts) == 1:
        print(f"{tinfo}: No dtype found. Assuming fp16")
        dims = parts[0]
    elif len(parts) > 2:
        raise ValueError(f"Unexpected tensor info: {tinfo}. Expected a single @")
    else:
        dims = parts[0]
        dtype = parts[1]
        if dtype not in list(dtype_mapping.keys()):
            raise ValueError(
                f"Unexpected dtype: {dtype}. Wanted one of: {list(dtype_mapping.keys())}"
            )
        dtype = dtype_mapping[dtype]

    dims = dims.split("x")
    dims = list(filter(lambda x: x != "", dims))

    if len(dims) == 0:
        raise ValueError(f"Found no dimensions in: {tinfo}")

    try:
        dims = list(map(lambda x: int(x), dims))
    except ValueError:
        raise ValueError(f"Found non integer dimension in: {tinfo}")

    return (dims, dtype)


def _load_spec(path):
    """The launch spec beside ``path``, or None when the folder has none.

    Imported lazily, like ``_launch``: this package does not depend on
    torch-spyre at install time.
    """
    from torch_spyre.execution.kernel_cache import load_launch_spec

    return load_launch_spec(str(path))


def _contiguous_strides(shape):
    """Row-major host strides for ``shape``.

    ``spyre_empty_with_layout`` takes host strides separately from the device
    layout; the host view of a CLI-built tensor is always contiguous.
    """
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _tensors_from_spec(spec, bindings):
    """Build every tensor the kernel expects, straight from the spec.

    Inputs are ``ones`` and outputs ``empty``, matching what the explicit path
    does. The pool tensor, when the kernel takes one, is prepended here exactly
    as inductor's ``call_kernel`` does -- it is not one of ``args``.
    """
    import torch

    tensors = []
    pool_size = spec.get("pool_size", 0)
    if pool_size > 0:
        tensors.append(torch.empty(pool_size, dtype=torch.uint8, device="spyre"))

    for arg in sorted(spec["args"], key=lambda a: a["arg_index"]):
        shape = []
        for dim in arg["shape"]:
            if isinstance(dim, int):
                shape.append(dim)
            elif dim in bindings:
                shape.append(int(bindings[dim]))
            else:
                raise ValueError(
                    f"arg {arg['arg_index']}: dimension '{dim}' is symbolic. "
                    f"Bind it with --bind {dim}=N"
                )
        dtype = getattr(torch, arg["dtype"])
        is_input = arg["role"] == "input"

        # Allocate WITH the recorded device layout when there is one. Shape and
        # dtype alone are not enough: an op can be compiled against a specific
        # packing along the sticks -- a depthwise conv2d puts the channels in one
        # stick -- and a tensor in the default arrangement launches fine and
        # returns wrong data. The layout is part of the execution contract.
        layout = None
        if arg.get("layout"):
            from torch_spyre.execution.kernel_cache import spyre_layout_from_spec

            layout = spyre_layout_from_spec(arg["layout"])

        if layout is None:
            tensor = (torch.ones if is_input else torch.empty)(
                shape, dtype=dtype, device="spyre"
            )
        else:
            from torch_spyre._C import spyre_empty_with_layout

            # spyre_empty_with_layout goes straight to the allocator, which needs
            # a live RuntimeContext. Ordinary factory calls bring it up lazily;
            # this one does not, and it can be the first allocation we make.
            torch.spyre._impl._lazy_init()
            tensor = spyre_empty_with_layout(
                tuple(shape),
                tuple(_contiguous_strides(shape)),
                dtype,
                layout,
                torch.device("spyre"),
            )
            if is_input:
                tensor.fill_(1.0)
        tensors.append(tensor)
    return tensors


def launch_from_cli(path, inputs, outputs, bindings=None):
    path = Path(path)
    import torch

    bindings = bindings or {}
    spec = _load_spec(path)

    if spec is not None and not inputs and not outputs:
        # The folder describes itself, so there is nothing to retype.
        tensors = _tensors_from_spec(spec, bindings)
        # By role, not by position: an output need not be a trailing argument,
        # and an in-place arg is both. The pool, when there is one, sits at 0 and
        # is not in args, so shift by the same offset used to build the tensors.
        offset = 1 if spec.get("pool_size", 0) > 0 else 0
        out_positions = [
            a["arg_index"] + offset
            for a in sorted(spec["args"], key=lambda a: a["arg_index"])
            if a["role"] != "input"
        ]
        print(f"using {spec.get('kernel_name', '?')} launch spec from {path}")
    else:
        tensors = []
        for iarg in inputs:
            shape, dtype = create_tensor_info(iarg)
            tensor = torch.ones(
                shape,
                dtype=getattr(torch, dtype),
                device="spyre",
            )
            tensors.append(tensor)

        for oarg in outputs:
            shape, dtype = create_tensor_info(oarg)
            tensor = torch.empty(
                shape,
                dtype=getattr(torch, dtype),
                device="spyre",
            )
            tensors.append(tensor)
        # Explicit form appends outputs after inputs, so they are the tail.
        out_positions = list(range(len(inputs), len(tensors)))

        if spec is not None:
            # Explicit arguments AND a spec: check them rather than trusting
            # them. This is the case that used to launch and return wrong data.
            from torch_spyre.execution.kernel_cache import check_launch_spec

            problems = check_launch_spec(spec, tensors, bindings)
            if problems:
                raise ValueError(
                    "the tensors given do not match what this kernel expects:\n"
                    + "\n".join(f"  - {p}" for p in problems)
                    + "\n\nOmit -i/-o to build them from the spec instead."
                )

    runner = _launch(path, tensors)

    # Every output, not just the last: reading them all back is also what makes
    # a failed launch visible, since .cpu() is where a dead kernel surfaces.
    if not out_positions:
        out_positions = [len(tensors) - 1]
    for pos in out_positions:
        if len(out_positions) > 1:
            print(f"output (arg {pos}):")
        print(tensors[pos].cpu())
    del runner


def launch(*tensors, path="."):
    """Launch a bundle; returns the runner, which the caller MUST keep alive.

    Bind the result (``runner = launch(a, b, c)``) and keep it in scope until
    the outputs have been read back. Calling this as a bare statement frees the
    JobPlan while the launch is still in flight, and the device then faults
    with what looks like a hardware error.
    """
    path = Path(path)
    return _launch(path, tensors)
