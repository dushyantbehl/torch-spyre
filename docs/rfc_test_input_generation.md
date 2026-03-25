# RFC: Test Input/Output Generation Strategy for Torch-Spyre Pipeline Stages

**Authors:** Torch-Spyre Team
**Status:** Draft
**Date:** 2026-03-05
**Companion:** `docs/rfc_component_testing.md`

## 1. Overview

This document answers a concrete engineering question: **for each pipeline
stage, how do we generate the inputs and validate the outputs in a test?**

There are two approaches:

1. **Direct construction** — build the input objects from scratch using
   constructors and factory functions. Preferred when the data structures are
   simple dataclasses or have public constructors.

2. **Golden sample capture** — run the real pipeline once, serialize the
   intermediate data at each boundary, and use the captured data as test
   fixtures. Preferred when the data structures require heavy infrastructure
   to construct (e.g., `SchedulerNode`).

For each stage we analyze: (a) can the input be constructed? (b) can the
input be captured from a real run? (c) how do we validate the output?

## 2. Constructibility Summary

| Object | Constructible? | Dependencies | Approach |
|--------|:-:|------|----------|
| `SpyreTensorLayout` | **Yes** | `_C` module only | Direct: `SpyreTensorLayout([256,128], torch.float16)` |
| `TensorArg` | **Yes** | `SpyreTensorLayout` | Direct: `TensorArg(True, 0, fp16, Size([256,128]), {}, stl)` |
| `ConstantArg` | **Yes** | Nothing | Direct: `ConstantArg(1e-5, torch.float16)` |
| `KernelSpec` | **Yes** | `TensorArg` list | Direct: `KernelSpec("exx2", True, [256,128], args, scales, op_info)` |
| `FixedTiledLayout` | **Yes** | `SpyreTensorLayout` + sympy | Direct: `FixedTiledLayout(device, dtype, size, stride, stl)` |
| `PointwiseOp` / `ReductionOp` | **Yes** | Nothing | Direct: `PointwiseOp("layernormscale", [x], {constants:{eps:1e-5}})` |
| FX Graph (for decomposition) | **Yes** | `import torch_spyre` | Direct: `torch.fx.symbolic_trace(fn)` where fn uses `torch.ops.spyre.*` |
| `generate_sdsc()` input dicts | **Yes** | `SpyreTensorLayout` | Direct: hand-built dicts with `name`, `scale`, `device_layout`, `host_size`, `lx_addr` |
| `SchedulerNode` | **No** | Full `torch._inductor` scheduler | Golden capture from `torch.compile` run |
| `TensorBox` / `Pointwise` / `Reduction` | **No** | `V.graph` virtualized global | Golden capture or pipeline hook |
| `SchedNodeArg` | **Partial** | `MemoryDep` (heavy) + `FixedTiledLayout` (easy) | Golden capture; or mock `MemoryDep` |

## 3. Stage-by-Stage Input/Output Generation

---

### Stage 1: Decomposition

#### Input Generation: Direct Construction

The input is a Python function containing `torch.ops.spyre.*` calls. All
Spyre custom ops have `register_fake` implementations, so
`torch.fx.symbolic_trace` works without a device.

**Requirements:** `import torch_spyre` (triggers op registration).

```python
import torch
import torch_spyre  # registers torch.ops.spyre.* custom ops

def make_layernorm_fx_graph():
    """Construct an FX graph containing spyre.layer_norm."""
    def fn(x, weight, bias):
        return torch.ops.spyre.layer_norm(x, [128], weight, bias, 1e-5)
    return torch.fx.symbolic_trace(fn)

def make_softplus_fx_graph():
    """Construct an FX graph containing spyre.softplus."""
    def fn(x):
        return torch.ops.spyre.softplus(x, 1.0, 20.0)
    return torch.fx.symbolic_trace(fn)
```

No golden samples needed. The FX graph is trivial to construct.

#### Output Validation

After applying decompositions, inspect the graph's nodes:

```python
def get_op_set(gm: torch.fx.GraphModule) -> set:
    """Extract all call_function targets from a graph module."""
    return {n.target for n in gm.graph.nodes if n.op == "call_function"}

# Test: layer_norm decomposes
ops = get_op_set(decomposed_gm)
assert torch.ops.spyre.layer_norm not in ops
assert torch.ops.spyre.exx2 in ops
assert torch.ops.spyre.layernormscale in ops
assert torch.ops.spyre.layernormnorm in ops
```

#### Golden Samples: Not Needed

FX graphs are cheap to construct. No value in capturing them.

---

### Stage 2: Lowering

#### Input Generation: Golden Capture Required

The lowering functions (`lower_exx2`, `lower_layernormscale`, etc.) take
`TensorBox` objects which require `V.graph` (Inductor's virtualized graph
context). These cannot be constructed outside a `torch.compile` run.

**Option A: Capture via pipeline hook**

Instrument the lowering functions to serialize their inputs/outputs:

```python
# In a test harness, capture the IR state after lowering
captured_ir = []

@contextmanager
def capture_lowered_ir():
    original_realize = TensorBox.realize
    def capturing_realize(self):
        captured_ir.append({
            "type": type(self.data).__name__,
            "size": list(self.data.get_size()) if hasattr(self.data, 'get_size') else None,
            "reduction_type": getattr(self.data, 'reduction_type', None),
        })
        return original_realize(self)
    TensorBox.realize = capturing_realize
    try:
        yield captured_ir
    finally:
        TensorBox.realize = original_realize
```

**Option B: Test the lowering contract indirectly**

Instead of testing `lower_exx2()` directly, test the *properties* of its
output by capturing IR nodes after the scheduler pass. This is the approach
used by `capture_sdsc()` from the component testing strategy.

#### Output Validation

The lowering output properties that matter:

| Property | How to Validate |
|----------|----------------|
| IR node type | `isinstance(node.data, SpyreReduction)` vs `isinstance(node.data, Pointwise)` |
| `reduction_type` | `node.data.reduction_type == "exx2"` |
| `ranges` (output shape) | `node.data.ranges == [256, 64]` |
| `op_info` on `SpyreReduction` | `node.data.op_info["constants"]["exx2scale"] == 0.0078125` |

#### Golden Samples: Recommended

Capture the IR node types and shapes from a known-good run:

```json
// golden/lowering/layernorm_256x128.json
{
  "ops": [
    {"type": "SpyreReduction", "reduction_type": "exx2", "ranges": [256, 64]},
    {"type": "Pointwise", "op_origin": "layernormscale", "ranges": [256, 64]},
    {"type": "Pointwise", "op_origin": "layernormnorm", "ranges": [256, 128]}
  ]
}
```

---

### Stage 3a: Layout Assignment

#### Input Generation: Hybrid (Direct + Golden)

The **`SpyreTensorLayout`** objects that serve as inputs to layout
decisions can be constructed directly:

```python
from torch_spyre._C import SpyreTensorLayout

# These are the actual layouts from the layer_norm trace
input_stl = SpyreTensorLayout([256, 128], torch.float16)
assert input_stl.device_size == [2, 256, 64]
assert input_stl.dim_map == [1, 0, 1]

weight_stl = SpyreTensorLayout([128], torch.float16)
assert weight_stl.device_size == [2, 64]
assert weight_stl.dim_map == [0, 0]

# Sparse layout (exx2 output) — constructed with explicit dim_order
exx2_out_stl = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])
assert exx2_out_stl.device_size == [64, 256, 64]
assert exx2_out_stl.dim_map == [1, 0, -1]
```

However, the full `pointwise_layout()` / `reduction_layout()` functions
take `SchedulerNode` and `SchedNodeArg` which require the Inductor runtime.

**Recommended approach:** Extract the layout decision logic into testable
pure functions. Until that refactoring is done, use golden samples:

```json
// golden/layout/layernorm_256x128.json
{
  "nodes": [
    {
      "name": "buf0",
      "op": "Reduction(exx2)",
      "input_args": [
        {"host_size": [256, 128], "device_size": [2, 256, 64], "dim_map": [1, 0, 1]}
      ],
      "output": {
        "host_size": [256, 64], "device_size": [64, 256, 64],
        "dim_map": [1, 0, -1], "device_dtype": "SEN169_FP16"
      }
    },
    {
      "name": "buf1",
      "op": "Pointwise(layernormscale)",
      "input_args": [
        {"host_size": [256, 64], "device_size": [64, 256, 64], "dim_map": [1, 0, -1]}
      ],
      "output": {
        "host_size": [256, 64], "device_size": [64, 256, 64],
        "dim_map": [1, 0, -1], "device_dtype": "SEN169_FP16"
      }
    },
    {
      "name": "buf2",
      "op": "Pointwise(layernormnorm)",
      "input_args": [
        {"host_size": [256, 128], "device_size": [2, 256, 64], "dim_map": [1, 0, 1]},
        {"host_size": [256, 64], "device_size": [64, 256, 64], "dim_map": [1, 0, -1]},
        {"host_size": [256, 64], "device_size": [64, 256, 64], "dim_map": [1, 0, -1]},
        {"host_size": [128], "device_size": [2, 64], "dim_map": [0, 0]},
        {"host_size": [128], "device_size": [2, 64], "dim_map": [0, 0]}
      ],
      "output": {
        "host_size": [256, 128], "device_size": [2, 256, 64],
        "dim_map": [1, 0, 1], "device_dtype": "SEN169_FP16"
      }
    }
  ]
}
```

#### Output Validation

What to check on the output `FixedTiledLayout`:

```python
def validate_layout(output, expected):
    assert list(output.device_layout.device_size) == expected["device_size"]
    assert list(output.device_layout.dim_map) == expected["dim_map"]
```

But the key **property-based** validations that don't need golden files:

```python
# Pointwise with same-size input: output layout == input layout
assert output_stl.device_size == input_stl.device_size

# Reduction output is sparse when reducing stick dim
assert output_stl.dim_map[-1] == -1

# layernormnorm output matches first arg (not sparse args)
assert output_stl.dim_map == first_arg_stl.dim_map
```

---

### Stage 3b: Core Division

#### Input Generation: Direct Construction

`core_split()` and `multi_dim_core_split()` are pure math functions. No
construction needed — just pass integers:

```python
from torch_spyre._inductor.core_division import core_split, multi_dim_core_split

# From trace: buf1 (layernormscale)
# device_size=[64, 256, 64], stick dim index=0
assert core_split(64, 32) == 32   # 64 sticks / 32 cores = 2 per core

# From trace: buf0 (exx2) — reduction, no split
# Non-matmul reductions get n_cores=1 (by design)
```

The full `divide_pointwise_op()` / `divide_reduction_op()` functions need
`SchedulerNode`, so golden capture is needed for integration tests. But the
core math is testable directly.

#### Output Validation

```python
import math

def validate_core_division(n_cores_used, core_division):
    """Validate invariants on core division output."""
    for splits in core_division:
        assert math.prod(splits) == n_cores_used
    assert n_cores_used >= 1
    assert n_cores_used <= 32
```

#### Golden Samples: Optional

Golden samples are useful for regression testing the full
`divide_pointwise_op` path, but the pure math functions don't need them.

```json
// golden/core_division/layernorm_256x128.json
{
  "max_cores": 32,
  "nodes": [
    {"name": "buf0", "op": "exx2", "n_cores_used": 1,
     "core_division": [[1,1,1], [1,1,1]]},
    {"name": "buf1", "op": "layernormscale", "n_cores_used": 32,
     "core_division": [[32,1,1], [32,1,1]]},
    {"name": "buf2", "op": "layernormnorm", "n_cores_used": 1,
     "core_division": [[1,1,1],[1,1,1],[1,1,1],[1,1],[1,1],[1,1,1]]}
  ]
}
```

---

### Stage 4a: SpyreOpFuncs

#### Input Generation: Direct Construction

`SpyreOpFuncs` methods are static with zero dependencies. Pass strings or
any objects — they just get wrapped in dataclasses:

```python
from torch_spyre._inductor.spyre_kernel import SpyreOpFuncs, PointwiseOp

# Direct construction — no mocks, no golden samples
result = SpyreOpFuncs.layernormscale("x", 1e-5)
result = SpyreOpFuncs.layernormnorm("x", "mean", "scale", "weight", "bias")
result = SpyreOpFuncs.softplus("x", 1.0, 20.0)
result = SpyreOpFuncs.add("a", "b")
```

#### Output Validation

```python
def test_layernormscale():
    result = SpyreOpFuncs.layernormscale("x", 1e-5)
    assert isinstance(result, PointwiseOp)
    assert result.op == "layernormscale"
    assert result.arguments == ["x"]
    assert result.op_info["constants"]["eps"] == 1e-5
```

#### Golden Samples: Not Needed

The outputs are trivial to validate directly.

---

### Stage 4b: KernelSpec Construction

#### Input Generation: Direct Construction

`KernelSpec` and `TensorArg` are plain dataclasses. Build them from scratch
using `SpyreTensorLayout` (which is also directly constructible):

```python
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.runtime import TensorArg, KernelSpec

def make_exx2_kernel_spec():
    """Build the exact KernelSpec from the layer_norm trace."""
    stl_in = SpyreTensorLayout([256, 128], torch.float16)
    stl_out = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])

    return KernelSpec(
        op="exx2",
        is_reduction=True,
        dimensions=[256, 128],
        args=[
            TensorArg(True, 0, torch.float16, torch.Size([256, 128]), {}, stl_in),
            TensorArg(False, 1, torch.float16, torch.Size([256, 64]), {}, stl_out),
        ],
        scales=[[0, 1], [0, -1]],
        op_info={
            "constants": {"exx2scale": 0.0078125, "useZeroMean": False},
            "core_division": [[1, 1, 1], [1, 1, 1]],
            "n_cores_used": 1,
        },
    )

def make_layernormscale_kernel_spec():
    stl = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])
    return KernelSpec(
        op="layernormscale",
        is_reduction=False,
        dimensions=[256, 64],
        args=[
            TensorArg(True, 0, torch.float16, torch.Size([256, 64]), {}, stl),
            TensorArg(False, 1, torch.float16, torch.Size([256, 64]), {}, stl),
        ],
        scales=[[0, 1], [0, 1]],
        op_info={
            "core_division": [[32, 1, 1], [32, 1, 1]],
            "n_cores_used": 32,
            "constants": {"eps": 1e-5},
        },
    )

def make_layernormnorm_kernel_spec():
    stl_dense = SpyreTensorLayout([256, 128], torch.float16)
    stl_sparse = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])
    stl_1d = SpyreTensorLayout([128], torch.float16)

    return KernelSpec(
        op="layernormnorm",
        is_reduction=False,
        dimensions=[256, 128],
        args=[
            TensorArg(True, 0, torch.float16, torch.Size([256, 128]), {}, stl_dense),
            TensorArg(True, 1, torch.float16, torch.Size([256, 64]), {}, stl_sparse),
            TensorArg(True, 2, torch.float16, torch.Size([256, 64]), {}, stl_sparse),
            TensorArg(True, 3, torch.float16, torch.Size([128]), {}, stl_1d),
            TensorArg(True, 4, torch.float16, torch.Size([128]), {}, stl_1d),
            TensorArg(False, 5, torch.float16, torch.Size([256, 128]), {}, stl_dense),
        ],
        scales=[[0, 1], [0, 1], [0, 1], [-1, 0], [-1, 0], [0, 1]],
        op_info={
            "core_division": [[1,1,1],[1,1,1],[1,1,1],[1,1],[1,1],[1,1,1]],
            "n_cores_used": 1,
        },
    )
```

#### Output Validation

Structural invariants that always hold:

```python
def validate_kernel_spec(ks: KernelSpec):
    # scales dimensions match
    assert len(ks.scales) == len(ks.args)
    for s in ks.scales:
        assert len(s) == len(ks.dimensions)

    # exactly one output
    n_out = sum(1 for a in ks.args if isinstance(a, TensorArg) and not a.is_input)
    assert n_out >= 1

    # reduction flag consistency
    if ks.is_reduction:
        assert any(-1 in s for s in ks.scales), "reduction must have a -1 scale on output"

    # broadcast dims are -1
    for i, arg in enumerate(ks.args):
        if isinstance(arg, TensorArg) and arg.is_input:
            for j, sv in enumerate(ks.scales[i]):
                if sv == -1:
                    pass  # broadcast or reduction dim — OK
                elif sv >= 0:
                    assert sv < len(arg.host_size), f"scale {sv} out of range for {arg.host_size}"
```

#### Golden Samples: Optional but Valuable

The hand-built KernelSpecs above ARE the golden samples — they're copied
from the trace. Storing them as fixtures avoids re-deriving them:

```python
# tests/_inductor/fixtures/kernelspecs.py
LAYERNORM_256x128_KERNELSPECS = {
    "exx2": make_exx2_kernel_spec(),
    "layernormscale": make_layernormscale_kernel_spec(),
    "layernormnorm": make_layernormnorm_kernel_spec(),
}
```

---

### Stage 5: SuperDSC Generation

#### Input Generation: Direct Construction

`generate_sdsc()` takes a `pointers` dict and keyword arguments. The
`inputs`/`outputs` are plain dicts with a `device_layout` field that must be
a `SpyreTensorLayout` (the code accesses `.device_size`, `.dim_map`, and
`.device_dtype` on it).

**You can construct the input entirely by hand:**

```python
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.codegen.superdsc import generate_sdsc
from torch_spyre._inductor.constants import SEGMENT_OFFSETS

def make_exx2_sdsc_input():
    """Build the exact input to generate_sdsc() from the trace."""
    stl_in = SpyreTensorLayout([256, 128], torch.float16)
    stl_out = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])

    pointers = dict(zip(
        ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"],
        SEGMENT_OFFSETS,
    ))

    kernel_descriptor = dict(
        name="test_exx2",
        op="exx2",
        reduction=True,
        dimensions=[256, 128],
        inputs=[{
            "name": "arg0",
            "scale": [0, 1],
            "device_layout": stl_in,
            "host_size": torch.Size([256, 128]),
            "lx_addr": None,
        }],
        outputs=[{
            "name": "arg1",
            "scale": [0, -1],
            "device_layout": stl_out,
            "host_size": torch.Size([256, 64]),
            "lx_addr": None,
        }],
        op_info={
            "constants": {"exx2scale": 0.0078125, "useZeroMean": False},
            "core_division": [[1, 1, 1], [1, 1, 1]],
            "n_cores_used": 1,
        },
    )
    return pointers, kernel_descriptor

# Call it:
pointers, kd = make_exx2_sdsc_input()
sdsc = generate_sdsc(pointers, **kd)
```

**Alternatively, build from a KernelSpec** (mirroring what `async_compile.py` does):

```python
def kernel_spec_to_sdsc_input(ks: KernelSpec, kernel_name: str):
    """Convert a KernelSpec to generate_sdsc() arguments.

    This mirrors SpyreAsyncCompile.sdsc() logic.
    """
    _arg_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]

    inputs, outputs = [], []
    for index, ts in enumerate(ks.args):
        entry = {
            "name": _arg_names[index],
            "scale": ks.scales[index],
            "device_layout": ts.device_layout,
            "host_size": ts.host_size,
            "lx_addr": None,
        }
        if ts.is_input:
            inputs.append(entry)
        else:
            outputs.append(entry)

    kernel_descriptor = {
        "name": kernel_name,
        "reduction": ks.is_reduction,
        "op": ks.op,
        "dimensions": ks.dimensions,
        "inputs": inputs,
        "outputs": outputs,
    }
    if ks.op_info:
        kernel_descriptor["op_info"] = ks.op_info

    pointers = dict(zip(_arg_names, SEGMENT_OFFSETS))
    return pointers, kernel_descriptor
```

This means you can build a `KernelSpec` by hand (Stage 4b), convert it
to `generate_sdsc` input, and test the full codegen path — all without
running `torch.compile`.

#### Output Validation: Structural + Golden Files

**Structural (always run):**

```python
import json

def validate_sdsc(sdsc: dict, op: str, n_inputs: int, n_outputs: int):
    errors = []

    # JSON serializable
    try:
        json.dumps(sdsc)
    except TypeError as e:
        errors.append(f"Not JSON serializable: {e}")

    # Round-trip
    rt = json.loads(json.dumps(sdsc))
    if rt != sdsc:
        errors.append("JSON round-trip mismatch")

    # Top-level key matches op
    if op not in sdsc:
        errors.append(f"Missing top-level key '{op}'")
    else:
        compute = sdsc[op]
        # Must have dscs_ array
        if "dscs_" not in compute:
            errors.append("Missing 'dscs_' key")
        else:
            dsc = compute["dscs_"][0]
            # Check computeOp_
            if "computeOp_" not in dsc:
                errors.append("Missing 'computeOp_'")
            else:
                cop = dsc["computeOp_"][0]
                if cop.get("opFuncName") != op:
                    errors.append(f"opFuncName={cop.get('opFuncName')} != {op}")
                if cop.get("exUnit") != "sfp":
                    errors.append(f"exUnit={cop.get('exUnit')} != sfp")

                # Tensor count
                n_in = len(cop.get("inputLabeledDs", []))
                n_out = len(cop.get("outputLabeledDs", []))
                if n_in != n_inputs:
                    errors.append(f"inputLabeledDs count {n_in} != {n_inputs}")
                if n_out != n_outputs:
                    errors.append(f"outputLabeledDs count {n_out} != {n_outputs}")

    return errors
```

**Golden file (regression):**

```python
import pathlib

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden_sdsc"

def test_sdsc_golden(op_name, sdsc_dict):
    actual = json.dumps(sdsc_dict, indent=2, sort_keys=True)
    golden_path = GOLDEN_DIR / f"{op_name}.json"

    if golden_path.exists():
        expected = golden_path.read_text()
        assert actual == expected, (
            f"SuperDSC output changed for {op_name}. "
            f"To update: SPYRE_UPDATE_GOLDEN=1 pytest ..."
        )
    else:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual)
        pytest.skip(f"Golden file created: {golden_path}")
```

#### Golden Samples: Strongly Recommended

The SuperDSC JSON is large (500-1200 lines per kernel). Golden files are
the most practical way to detect regressions. Capture once, diff thereafter.

---

### Stage 6: Backend Compile + Runtime

#### Input Generation: From Stage 5 Output

The input is the `sdsc.json` file produced by Stage 5. This can be:

1. Generated from the `generate_sdsc()` test above, or
2. Captured from a real run (the log shows the exact paths).

```python
def test_dxp_standalone_accepts_sdsc(tmp_path):
    """Verify dxp_standalone can compile our SuperDSC JSON."""
    # Generate from hand-built KernelSpec
    ks = make_exx2_kernel_spec()
    pointers, kd = kernel_spec_to_sdsc_input(ks, "test_exx2")
    sdsc = generate_sdsc(pointers, **kd)

    # Write to expected directory structure
    kernel_dir = tmp_path / "execute" / "test_exx2"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "sdsc.json").write_text(json.dumps(sdsc, indent=2))

    # Run dxp_standalone
    result = subprocess.run(
        ["dxp_standalone", "-d", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"dxp_standalone failed:\n{result.stderr}"
```

#### Output Validation

- **Without hardware:** `dxp_standalone` returns 0 (compilation success)
- **With hardware:** Numerical comparison against CPU

---

## 4. Decision Matrix: When to Use Which Approach

| Stage | Input Type | Output Type | Generate Input How? | Validate Output How? | Golden Samples? | Hardware? |
|-------|-----------|-------------|--------------------|--------------------|:---:|:---:|
| **1. Decomposition** | `torch.Tensor` + scalars (via FX graph) | FX graph nodes (set of `call_function` targets) | **Direct:** `torch.fx.symbolic_trace(fn)` where `fn` calls `torch.ops.spyre.layer_norm(...)`. Only needs `import torch_spyre`. | Inspect graph: assert `spyre.exx2`, `spyre.layernormscale`, `spyre.layernormnorm` present; assert `spyre.layer_norm` absent. Check exx2Scale == `1/N`. | No | No |
| **2. Lowering** | `TensorBox` + scalars (Inductor buffers with `.get_size()`, `.get_dtype()`) | `Pointwise` or `SpyreReduction` IR nodes (with `ranges`, `reduction_type`, `op_info`) | **Golden capture:** hook into `torch.compile` to capture IR node properties (type, ranges, reduction_type, op_info). Cannot construct `TensorBox` standalone — requires `V.graph`. | Assert IR type (`SpyreReduction` vs `Pointwise`), `reduction_type`, `ranges` shape, `op_info.constants`. E.g., exx2 → `SpyreReduction`, `ranges=[256,64]`, `exx2scale=0.0078125`. | **Yes** — capture `{type, ranges, reduction_type, op_info}` per node | No |
| **3a. Layout** | `SchedulerNode` with `FixedLayout` + `list[SchedNodeArg]` | `FixedTiledLayout` containing `SpyreTensorLayout` (with `device_size`, `dim_map`, `device_dtype`) | **Hybrid:** `SpyreTensorLayout` objects are directly constructible: `SpyreTensorLayout([256,128], fp16)`. But full `pointwise_layout()`/`reduction_layout()` need `SchedulerNode` → **golden capture** for integration, **direct construction** for STL unit tests. Long-term: refactor layout logic into pure functions. | Assert `device_size`, `dim_map`, `device_dtype` on output. Property checks: exx2 output is sparse (`dim_map[-1]==-1`), pointwise propagates input layout, layernormnorm output matches `arg[0]`. | **Yes** — capture `{node_name, input_args[].{device_size,dim_map}, output.{device_size,dim_map}}` | No (needs `_C`) |
| **3b. Core Division** | `device_size: list[int]` + `max_cores: int` (for pure math); `SchedulerNode` (for full path) | `n_cores_used: int` + `core_division: list[list[int]]` | **Direct for `core_split()`:** `core_split(64, 32)` — pure integers, zero deps. **Golden capture for `divide_*_op()`:** needs `SchedulerNode` context. | Assert: `core_split(64,32)==32`, `core_split(7,4)==1`. Invariant: `math.prod(splits)==n_cores_used` for every tensor. Check broadcast disables division. | Optional — useful for `divide_*_op()` regression | No |
| **4a. OpFuncs** | Python scalars or strings (any `RValue`) | `PointwiseOp(op, arguments, op_info)` or `str` (for exx2) | **Direct:** `SpyreOpFuncs.layernormscale("x", 1e-5)` — static methods, zero deps, pass dummy string args. | Assert `result.op`, `result.arguments`, `result.op_info`. E.g., `softplus` → `op_info.constants.softplusBeta==1.0`. Check `exx2` returns `str` not `PointwiseOp`. | No | No |
| **4b. KernelSpec** | `op: str` + `TensorArg`s + `scales` + `op_info` | `KernelSpec` dataclass | **Direct:** build `SpyreTensorLayout` → `TensorArg` → `KernelSpec` entirely by hand. All plain dataclasses. Use trace data as the source of truth for field values. Can also bridge: build `KernelSpec`, then convert to `generate_sdsc()` input via `kernel_spec_to_sdsc_input()`. | Structural: `len(scales)==len(args)`, `len(scales[i])==len(dimensions)`, exactly 1 output arg, reduction has `-1` in output scale, broadcast args have `-1` scales. dtype validation: bfloat16 raises `Unsupported`. | Optional — hand-built KernelSpecs from trace ARE the golden data | No (needs `_C`) |
| **5. SuperDSC** | `pointers: dict[str,int]` + `kernel_descriptor: dict` (with `op`, `dimensions`, `inputs[]`, `outputs[]`, `op_info`) | `dict` (JSON-serializable SuperDSC) | **Direct:** hand-build input dicts with `SpyreTensorLayout` objects for `device_layout`. Or chain from Stage 4b: `KernelSpec` → `kernel_spec_to_sdsc_input()` → `generate_sdsc()`. No `torch.compile` needed. | Structural: JSON serializable, round-trip, top-key == op, `dscs_` exists, `computeOp_[0].opFuncName == op`, `exUnit == "sfp"`, `dataFormat_ == "SEN169_FP16"`, tensor count matches. **Golden files** for exact regression (500-1200 lines per kernel). | **Strongly recommended** — capture full JSON, diff on subsequent runs | No (needs `_C`) |
| **6. Runtime** | `sdsc.json` file (from Stage 5) + device tensors | `g2.graph.cbor` → numerical result tensor | **From Stage 5 output:** write `generate_sdsc()` result to `sdsc.json`, run `dxp_standalone`. Or use captured `sdsc.json` from a golden run. | Without HW: `dxp_standalone` returns 0. With HW: `torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)`. | `sdsc.json` from Stage 5 | `dxp_standalone` binary; Spyre HW for numerical |

## 5. The Golden Capture Harness

For stages that need golden capture, a single reusable harness can collect
data at every boundary during a `torch.compile` run:

```python
class PipelineCapture:
    """Run torch.compile and capture intermediate state at every seam."""

    def __init__(self):
        self.decomposed_ops = []     # Stage 1: list of op names
        self.ir_nodes = []           # Stage 2: IR node properties
        self.layouts = []            # Stage 3a: layout assignments
        self.core_divisions = []     # Stage 3b: core splits
        self.kernel_specs = []       # Stage 4: KernelSpec objects
        self.sdsc_jsons = []         # Stage 5: SuperDSC dicts

    def capture(self, fn, *args):
        """Run fn through torch.compile and capture all stages."""
        # Hook into each stage to collect data
        # (Uses the debug_trace infrastructure already in place)
        ...
        return self

    def save_golden(self, path: Path):
        """Serialize all captured data as golden files."""
        ...

    def compare_golden(self, path: Path) -> list[str]:
        """Compare current capture against golden files, return diffs."""
        ...
```

This harness would be used to generate the initial golden samples, then
the individual stage tests can load and use them as fixtures.

## 6. Recommended Priority

1. **Immediate (no infrastructure needed):**
   - SpyreOpFuncs tests (direct construction)
   - `core_split()` tests (direct construction)
   - Decomposition graph tests (FX trace)

2. **Next (needs `_C` module):**
   - KernelSpec construction tests (direct construction from trace data)
   - `generate_sdsc()` tests (direct construction + golden files)
   - SpyreTensorLayout property tests (direct construction)

3. **Later (needs capture harness):**
   - Lowering property tests (golden capture)
   - Full layout assignment tests (golden capture + refactoring)
   - Pipeline capture end-to-end (all stages)
