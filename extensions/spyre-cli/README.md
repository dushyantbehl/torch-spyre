# spyre-cli

## Install

Needs a working Spyre stack environment.

```
git clone ...
uv pip install .
```

## CLI

If the folder carries a `launch_spec.json` (written by the compiler next to
`spyreCodeDir/`), there is nothing to type — the tensors are built from it:

```
spyre launch <folder with spyreCode>
```

Otherwise, describe the tensors positionally:

```
spyre launch -i 10x512@fp16 -i 10x512@fp16 -o 10x512@fp16 <folder with spyreCode>
```

The inputs and outputs are of the form `"axbxc@d"`, where:

1. the first part is the tensor dimension info.
2. d specifies the dtype short form, like "fp16", "fp32" or "bf16".

Given both a spec and explicit `-i`/`-o`, the arguments are **checked** against
the spec and the launch is refused if they disagree:

```
$ spyre launch -i 512x10@fp16 -i 10x512@fp16 -o 10x512@fp16 <folder>
ValueError: the tensors given do not match what this kernel expects:
  - arg 0 (input): expected shape [10, 512], got [512, 10] -- same extents in a
    different order (transposed?)
```

A kernel whose bundle takes a caller-supplied pool tensor records its size in the
spec, and `spyre launch` allocates and prepends it. Passing `-i`/`-o` alone for
such a kernel is refused, since the pool is not one of the listed arguments.

For a kernel with a symbolic dimension, bind it:

```
spyre launch --bind s0=128 <folder with spyreCode>
```

### Without a launch spec

A folder compiled before the spec existed, or by a build that does not write one,
still launches from `-i`/`-o` exactly as before — but nothing is checked, so the
two constraints below apply.

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

To get those checked, either use `spyre launch` on a folder that has a spec, or
validate first:

```
import torch  # before torch_spyre, which torch loads as a backend extension
from torch_spyre.execution.kernel_cache import load_launch_spec, check_launch_spec

spec = load_launch_spec(folder)          # None when the folder has no spec
if spec is not None:
    problems = check_launch_spec(spec, [a, b, c])
    if problems:
        raise ValueError("\n".join(problems))
```
