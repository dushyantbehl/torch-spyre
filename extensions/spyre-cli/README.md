# spyre-cli

## Install

Needs a working Spyre stack environment.

```
git clone ...
uv pip install .
```

## CLI

A compiled folder carries a `launch_spec.json` (written by the compiler next to
`spyreCodeDir/`) describing every argument, so there is nothing to type:

```
spyre launch <folder with spyreCode>
```

Inputs are filled with ones and the outputs printed. That tells you the folder
loads and runs, and nothing about the numbers: an add of ones gives 2.0
everywhere, so any elementwise op touching both inputs looks plausible, and a
transpose or reduction-axis bug is invisible when every element is identical.

### Verifying the answer

Supply real values with `-i`, and a reference with `-o`, and the same command
checks the result instead of printing it:

```
spyre launch -i a.pt -i b.pt -o expected.pt <folder>
```

```
arg 2 (output): match  (max abs diff 4.88e-04, rtol 0.001)
1 output verified, 0 failed
```

On a mismatch it reports what differed and exits non-zero:

```
arg 2 (output): MISMATCH
  Tensor-likes are not close!
  Mismatched elements: 3 / 5120 (0.1%)
  Greatest absolute difference: 0.0312 at index (0, 17) (up to 1e-05 allowed)
  Greatest relative difference: 0.0156 at index (0, 17) (up to 0.001 allowed)
0 output verified, 1 failed
```

`-i` and `-o` are positional in `arg_index` order, matching how the spec binds
arguments. The pool tensor, when a kernel takes one, is never named by a flag —
the launcher allocates and prepends it.

| Exit code | Meaning |
|---|---|
| 0 | launched, and every output matched |
| 1 | a mismatch: well-formed launch, wrong numbers |
| 2 | could not launch: bad arguments, unreadable file, spec disagreement |

A script needs 1 and 2 separated: "this kernel is wrong" and "this invocation
was malformed" are different failures.

**File formats.** `.pt` goes through `torch.load` and is self-describing, so its
shape and dtype are cross-checked against the spec and a disagreement is an
error. It is the format the OpSpec Lab's `--save-inputs` already writes, so a
captured kernel replays with no conversion — including its
`{"tensors": [...]}` layout, where one `-i` supplies several arguments.

`.bin` is raw little-endian row-major element data, with shape and dtype taken
from the spec. It exists for a non-PyTorch producer. Because it carries no
metadata the only possible check is length, so a file of the right size but the
wrong dtype is undetectable — prefer `.pt`.

**Tolerances** default to the output dtype's own, taken from `torch.testing`, so
a result this tool calls close is one the rest of the test suite would too:

| dtype | rtol | atol |
|---|---|---|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| float64 | 1e-7 | 1e-7 |
| integer, bool | exact | exact |

Override with `--rtol`/`--atol`, or demand bitwise equality with `--exact`.
`--save-output FILE` writes the device result, for making a reference or
inspecting a failure.

### Ops compiled against a specific device layout

Some ops are compiled against a particular packing along the sticks rather than
the default arrangement — a depthwise conv2d, for instance, puts the channels in
one stick, which the caller sets with `.to(device_layout=SpyreTensorLayout(...))`.
The spec records that layout per argument, and `spyre launch` allocates each
tensor with it, so:

```
spyre launch <folder with a dwconv spyreCode>
```

is enough. Shape and dtype alone are **not**: they match the default arrangement
exactly, so a tensor built without the layout launches fine and returns wrong
data. Tensors loaded from `-i` files are allocated with the recorded layout for
the same reason, and checked against it before the launch.

For a kernel with a symbolic dimension, bind it:

```
spyre launch --bind s0=128 <folder with spyreCode>
```

### Without a launch spec

A folder with no `launch_spec.json` cannot be launched: its arguments are
unknown, there is no order to bind `-i` files in, and a `.bin` has nothing to
interpret it. Recompile to get a folder that describes itself.

## SDK

Example, in a folder with the "spyreCode" for a torch.add operation:

```
import torch
import spyre_cli

a = torch.ones([512, 1024], device="spyre", dtype=torch.float16)
b = torch.ones([512, 1024], device="spyre", dtype=torch.float16)
c = torch.empty([512, 1024], device="spyre", dtype=torch.float16)

runner = spyre_cli.launch(a, b, c)

print(c.cpu())
```

Bind the return value and keep it in scope until the outputs have been read
back. Dropping it frees the JobPlan while the launch is still in flight, and the
device then reports what looks like a hardware error.

`spyre_cli.launch` takes the tensors you hand it and does not consult a launch
spec, so on this path:

1. You need to pass the right input and output list.
2. If the shapes don't match, there is no error - the code will silently work.
3. The same goes for a device layout, which is easier to get wrong: shape and
   dtype can be right while the packing along the sticks is not.

The spec API reports all three. `load_kernelspec` reads what an artifact
requires; `check_kernelspec` checks tensors against it:

```
import torch  # before torch_spyre, which torch loads as a backend extension
import spyre_cli

spec = spyre_cli.load_kernelspec(folder)     # None when the folder has no spec
if spec is not None:
    for arg in spec.args:
        print(arg.arg_index, arg.role, arg.shape, arg.dtype, arg.layout)

    problems = spyre_cli.check_kernelspec(spec, [a, b, c])
    if problems:
        raise ValueError("\n".join(problems))
```

They are two calls because they take different things at different times:
`load_kernelspec` takes a path and is what you read to decide what tensors to
build, so at that point none exist to validate; `check_kernelspec` takes those
tensors once they do. Problems come back as a list rather than an exception, so
the policy stays with you.

### Verifying in Python

The comparison the CLI performs is available directly, for a script that owns
its own reference:

```
import torch
import spyre_cli

expected = fn(*cpu_inputs)                   # your reference, however you make it

runner = spyre_cli.launch(*device_inputs, device_output, path=folder)
torch.testing.assert_close(device_output.cpu(), expected, rtol=1e-2, atol=1e-2)
```

or, to get the CLI's per-dtype defaults and its failure report as a list:

```
problems = spyre_cli.check_values([device_output.cpu()], [expected])
```

`spyre_cli.load_tensors(path, args)` reads a `.pt` or `.bin` the same way the
CLI does, cross-checking it against the spec's argument records.
