# RFC: Component-Level Testing for the Torch-Spyre Compilation Pipeline

**Authors:** Torch-Spyre Team
**Status:** Draft
**Date:** 2026-03-05

## 1. Problem Statement

Today, torch-spyre tests are almost entirely end-to-end: a Python function
runs through `torch.compile`, the Spyre backend, `dxp_standalone`, and
hardware execution, then the numerical result is compared against CPU. A
failure anywhere in the pipeline surfaces as a single "wrong answer" or crash
with no isolation.

This RFC proposes a concrete testing strategy for each stage of the
compilation pipeline, based on observed data from an instrumented run of
`F.layer_norm` on a `[256, 128]` FP16 input tensor. The layer_norm example
is used because it exercises every stage — one high-level call decomposes
into three separate kernels (a reduction and two pointwise ops), each flowing
through layout assignment, core division, KernelSpec construction, and
SuperDSC JSON generation.

The trace log analyzed in this RFC is at `layer_norm.log` and was produced
by running `examples/layer_norm.py` with `SPYRE_DEBUG_TRACE=1`.

## 2. Pipeline Overview

```
F.layer_norm(x, [128], weight, bias, eps=1e-5)
  │
  │  Stage 1: Decomposition
  │  ──────────────────────
  ├──→ spyre.layer_norm → [exx2, layernormscale, layernormnorm]
  │         1 op becomes 3 FX nodes
  │
  │  Stage 2: Lowering
  │  ─────────────────
  ├──→ 3 FX nodes → 3 Inductor IR nodes (1 SpyreReduction + 2 Pointwise)
  │
  │  Stage 3: Stickification (Layout + Core Division)
  │  ──────────────────────────────────────────────────
  ├──→ 3 FixedLayout → 3 FixedTiledLayout (with SpyreTensorLayout)
  ├──→ 3 core division decisions
  │
  │  Stage 4: Kernel Code Generation (OpFuncs + KernelSpec)
  │  ──────────────────────────────────────────────────────
  ├──→ 3 KernelSpecs
  │
  │  Stage 5: SuperDSC Generation
  │  ────────────────────────────
  ├──→ 3 SuperDSC JSON dicts → 3 sdsc.json files
  │
  │  Stage 6: Backend Compile + Runtime
  │  ──────────────────────────────────
  └──→ dxp_standalone → g2.graph.cbor → launch_kernel → result
```

## 3. Stage-by-Stage Analysis

### 3.1 Stage 1: Decomposition

**Source:** `torch_spyre/_inductor/decompositions.py`

**What happens:** High-level PyTorch ops are intercepted and rewritten into
Spyre-specific custom ops. For ops with registered decompositions, Inductor
further decomposes them into lower-level custom ops.

#### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `torch.Tensor` + Python scalars | A call to `F.layer_norm(input, normalized_shape, weight, bias, eps)` where `input.device.type == "spyre"` |
| **Output** | FX graph nodes | Three new FX nodes replace the original single node |

#### Observed Data (from `layer_norm.log` lines 2–14)

```
Input:
  input.shape: [256, 128]
  input.dtype: torch.float16
  normalized_shape: [128]
  eps: 1e-05
  has_weight: True
  has_bias: True

Output: 3 FX nodes
  mean      = spyre.exx2(input, 0.0078125, False)
  norm_mean = spyre.layernormscale(mean, 1e-05)
  output    = spyre.layernormnorm(input, mean, norm_mean, weight, bias)
```

#### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| Layer norm decomposes into exactly 3 ops | `spyre.layer_norm(x[256,128], [128], w, b, 1e-5)` | The FX graph contains `spyre.exx2`, `spyre.layernormscale`, `spyre.layernormnorm` and does NOT contain `spyre.layer_norm` |
| exx2 scale is `1/N` | `normalized_shape=[N]` for various N | `exx2Scale == 1.0/N` (e.g., `1/128 = 0.0078125`) |
| Non-decomposed ops pass through unchanged | `spyre.softplus(x, 1.0, 20.0)` | `spyre.softplus` is still present in the graph |
| Interception only fires on Spyre device | `F.layer_norm(cpu_tensor, ...)` | Calls the original `torch.nn.functional.layer_norm`, no Spyre ops |
| Interception only fires for 1D normalized_shape | `F.layer_norm(x, [8, 128], ...)` | Falls back to original `layer_norm` (not intercepted) |

#### How to Test (no hardware needed)

Use `torch.fx.symbolic_trace` to create an FX graph, apply
`@register_decomposition`, and inspect the resulting graph nodes.

```python
def test_layernorm_decomposes_to_three_ops():
    def fn(x, weight, bias):
        return torch.ops.spyre.layer_norm(x, [128], weight, bias, 1e-5)

    gm = torch.fx.symbolic_trace(fn)
    # apply Inductor decompositions
    decomposed = apply_decompositions(gm)

    ops = {n.target for n in decomposed.graph.nodes if n.op == "call_function"}
    assert torch.ops.spyre.layer_norm not in ops
    assert torch.ops.spyre.exx2 in ops
    assert torch.ops.spyre.layernormscale in ops
    assert torch.ops.spyre.layernormnorm in ops
```

---

### 3.2 Stage 2: Lowering

**Source:** `torch_spyre/_inductor/lowering.py`

**What happens:** Each FX op is converted into an Inductor LoopLevel IR
node (`Pointwise`, `Reduction`, or `SpyreReduction`). This is where the
abstract op becomes a concrete computation graph with iteration ranges
and an inner function.

#### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `TensorBox` + Python scalars | Inductor buffer objects with `.get_size()`, `.get_dtype()`, `.get_device()` |
| **Output** | `Pointwise` or `SpyreReduction` (subclass of `Reduction`) | IR node with `ranges`, `reduction_ranges`, `reduction_type`, `inner_fn`, `op_info` |

#### Observed Data (from `layer_norm.log` lines 16–54)

| Op | Function | Input | Output IR |
|----|----------|-------|-----------|
| exx2 | `lower_exx2(x, 0.0078125, False)` | `x: TensorBox[256,128] fp16` | `SpyreReduction(type="exx2", ranges=[256, 64], reduction_ranges=[last dim], op_info={exx2scale: 0.0078125, useZeroMean: False})` |
| layernormscale | `lower_layernormscale(x, 1e-5)` | `x: TensorBox[256,64] fp16` | `Pointwise(ranges=[256, 64], inner_fn=layernormscale(load(x), 1e-5))` |
| layernormnorm | `lower_layernormnorm(x, mean, norm_mean, w, b)` | `x:[256,128], mean:[256,64], norm_mean:[256,64], weight:[128], bias:[128]` | `Pointwise(ranges=[256, 128], inner_fn=layernormnorm(x, mean, norm_mean, weight, bias))` |

#### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| exx2 produces `SpyreReduction` not `Pointwise` | `lower_exx2(x[256,128], 0.0078125, False)` | `isinstance(result.data, SpyreReduction)` and `result.data.reduction_type == "exx2"` |
| exx2 output range has `elem_in_stick` as last dim | `x[256,128] fp16` (stick=64 for fp16) | `ranges == [256, 64]` |
| exx2 carries `op_info.constants` | any exx2 | `op_info["constants"]["exx2scale"] == 0.0078125` and `op_info["constants"]["useZeroMean"] == False` |
| layernormscale produces `Pointwise` | `lower_layernormscale(x[256,64], 1e-5)` | `isinstance(result.data, Pointwise)` and `ranges == [256, 64]` |
| layernormnorm preserves original input shape | `lower_layernormnorm(x[256,128], ...)` | `ranges == [256, 128]` — same as `x.get_size()` |
| layernormnorm handles optional weight/bias | weight=None, bias=None | Does not crash; inner_fn loads only 3 inputs |

#### How to Test

Lowering functions use `V.graph` (Inductor's virtualized global). Testing
requires either:

- **Approach A (Capture):** Run `torch.compile` with a hook to capture IR
  nodes after lowering but before codegen.
- **Approach B (Mock):** Set up a minimal `GraphLowering` to call
  `lower_exx2()` directly.

Approach A is simpler initially; Approach B enables true unit tests.

---

### 3.3 Stage 3: Stickification

**Source:** `torch_spyre/_inductor/stickify.py` (layout) and
`torch_spyre/_inductor/core_division.py` (work distribution)

**What happens:** Two passes run over the scheduler nodes in sequence:

1. **Layout Assignment** converts each node's `FixedLayout` (host shape)
   into a `FixedTiledLayout` containing a `SpyreTensorLayout` (device shape
   with stick tiling and dimension mapping).

2. **Core Division** decides how to split each node's work across Spyre
   cores (up to 32).

#### 3.3.1 Layout Assignment

##### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `SchedulerNode` with `FixedLayout` + list of `SchedNodeArg` | Each arg has `.layout` (which is `FixedTiledLayout` for graph inputs, or the output layout of a predecessor node) |
| **Output** | `FixedTiledLayout` | Contains `device_layout: SpyreTensorLayout` with `device_size: list[int]`, `dim_map: list[int]`, `device_dtype: DataFormats` |

##### Observed Data (from `layer_norm.log` lines 56–134)

| Node | Op / Function | Input Arg Layouts | Output Layout |
|------|---------------|-------------------|---------------|
| buf0 | `reduction_layout(n, args)` for exx2 | `arg[0]: host=[256,128], device=[2,256,64], dim_map=[1,0,1]` | `host=[256,64], device=[64,256,64], dim_map=[1,0,-1]` (sparse: `-1`) |
| buf1 | `pointwise_layout(n, args)` for layernormscale | `arg[0]: host=[256,64], device=[64,256,64], dim_map=[1,0,-1]` | `host=[256,64], device=[64,256,64], dim_map=[1,0,-1]` (propagated) |
| buf2 | `pointwise_layout(n, args)` for layernormnorm | `arg[0]: host=[256,128], device=[2,256,64], dim_map=[1,0,1]` (dense), `arg[1,2]: [256,64] dim_map=[1,0,-1]` (sparse), `arg[3,4]: [128] dim_map=[0,0]` (1D) | `host=[256,128], device=[2,256,64], dim_map=[1,0,1]` (dense, from arg[0]) |

##### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| 2D tensor default layout | `SpyreTensorLayout([256,128], fp16)` | `device_size == [2, 256, 64]`, `dim_map == [1, 0, 1]` |
| 1D tensor default layout | `SpyreTensorLayout([128], fp16)` | `device_size == [2, 64]`, `dim_map == [0, 0]` |
| exx2 output is sparse | Reduction(type="exx2") on dense `[256,128]` input | `dim_map[-1] == -1` (sparse) |
| Pointwise propagates input layout | Same-shape pointwise (layernormscale) | Output `device_size` and `dim_map` identical to input |
| layernormnorm uses first arg layout | 5 args with mixed dense/sparse layouts | Output layout matches `arg[0]` (dense `[2,256,64]`), NOT `arg[1]` (sparse) |
| Broadcast input does not clobber layout | weight `[128]` != output `[256,128]` | Output not derived from weight layout |

##### How to Test (needs `_C` module, no hardware)

`SpyreTensorLayout` is constructed from the `_C` module. Layout assignment
logic can be tested by constructing `SpyreTensorLayout` objects directly and
verifying `device_size`, `dim_map`, and `host_stick_dim()`.

```python
def test_exx2_produces_sparse_layout():
    stl = SpyreTensorLayout([256, 128], torch.float16)
    assert stl.device_size == [2, 256, 64]
    assert stl.dim_map == [1, 0, 1]

    # After exx2 reduction, output is sparse
    out_stl = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])
    assert out_stl.dim_map[-1] == -1  # sparse
    assert out_stl.device_size == [64, 256, 64]
```

For the full `pointwise_layout` / `reduction_layout` logic, extracting the
layout decision into pure functions (that take `SpyreTensorLayout` objects
rather than `SchedulerNode`s) would enable direct unit testing.

#### 3.3.2 Core Division

##### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `SchedulerNode` with `FixedTiledLayout` + `max_cores: int` | The layout carries `device_layout.device_size` |
| **Output** | Two attributes set on the node: `n.n_cores_used: int` and `n.spyre_core_division: list[list[int]]` | One `list[int]` per tensor (inputs + output), product of each = `n_cores_used` |

##### Observed Data (from `layer_norm.log` lines 136–159)

| Node | Op Type | device_size | n_cores_used | core_division | Why |
|------|---------|-------------|-------------|---------------|-----|
| buf0 | Reduction(exx2) | [64,256,64] | 1 | `[[1,1,1],[1,1,1]]` | Non-matmul reduction → no split |
| buf1 | Pointwise | [64,256,64] | 32 | `[[32,1,1],[32,1,1]]` | `core_split(64, 32) = 32` on stick dim |
| buf2 | Pointwise | [2,256,64] | 1 | `[[1,1,1],×6]` | Broadcasts (weight/bias) → early return |

##### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| `core_split` pure math | `core_split(64, 32)` | `== 32` |
| `core_split` prime size | `core_split(7, 32)` | `== 7` (7 is prime, fits) |
| `core_split` no fit | `core_split(7, 4)` | `== 1` (7 is prime, > 4) |
| Pointwise splits stick dim | device_size=`[64,256,64]`, max_cores=32 | `n_cores_used=32`, split on dim 0 |
| Broadcast disables core division | Any arg where `arg.size != output.size` | `n_cores_used=1` |
| Non-matmul reduction gets 1 core | Reduction(type="exx2") | `n_cores_used=1` |
| Product invariant | Any valid division | `math.prod(splits_for_tensor_i) == n_cores_used` for all tensors |

##### How to Test (no hardware needed)

`core_split()` and `multi_dim_core_split()` are pure math functions that can
be tested directly with no dependencies.

```python
def test_core_split():
    assert core_split(64, 32) == 32
    assert core_split(256, 32) == 32
    assert core_split(2, 32) == 2
    assert core_split(7, 4) == 1
```

The full `divide_pointwise_op` / `divide_reduction_op` functions take
`SchedulerNode`s. Extracting the splitting logic into pure functions that
take `(device_sizes, dim_maps, op_type, max_cores)` would enable unit
testing without scheduler infrastructure.

---

### 3.4 Stage 4: Kernel Code Generation

**Source:** `torch_spyre/_inductor/spyre_kernel.py`

**What happens:** Two things happen in sequence for each kernel:

1. **Op Name Mapping** (`SpyreOpFuncs`): The Inductor op handler calls a
   static method that maps a PyTorch op name to a Spyre op name and wraps
   it in a `PointwiseOp` or `ReductionOp` dataclass.

2. **KernelSpec Construction** (`SpyreKernel.store` / `.store_reduction`):
   The `PointwiseOp`/`ReductionOp` is combined with tensor access
   information (layouts, scales, dimensions) into a `KernelSpec` dataclass.

#### 3.4.1 Op Name Mapping (SpyreOpFuncs)

##### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `RValue` objects (typically `TensorAccess`) + Python scalars | Arguments from the Inductor IR's inner function |
| **Output** | `PointwiseOp(op: str, arguments: list[RValue], op_info: dict)` | For reductions: `ReductionOp` or raw string |

##### Observed Data (from `layer_norm.log` lines 174–192)

| Op | Method Call | Output |
|----|-----------|--------|
| exx2 | `SpyreOpFuncs.exx2(a, b, c)` | `"spyre.exx2(...)"` (raw string — reduction path) |
| layernormscale | `SpyreOpFuncs.layernormscale(x, 1e-5)` | `PointwiseOp(op="layernormscale", args=[x], op_info={constants:{eps: 1e-05}})` |
| layernormnorm | `SpyreOpFuncs.layernormnorm(arg0..arg4)` | `PointwiseOp(op="layernormnorm", args=[arg0,arg1,arg2,arg3,arg4])` |

##### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| layernormscale carries eps | `SpyreOpFuncs.layernormscale("x", 1e-5)` | `result.op == "layernormscale"`, `result.op_info["constants"]["eps"] == 1e-5` |
| layernormnorm accepts 3-5 args | `SpyreOpFuncs.layernormnorm("x","m","s")` then `...("x","m","s","w","b")` | `len(result.arguments) == 3` then `== 5` |
| softplus carries beta and threshold | `SpyreOpFuncs.softplus("x", 1.0, 20.0)` | `result.op == "softplus"`, `op_info["constants"]["softplusBeta"] == 1.0` |
| Completeness: every method returns valid type | All methods on `SpyreOpFuncs` | Returns `PointwiseOp`, `ReductionOp`, or `str` |

##### How to Test (no hardware needed, zero dependencies)

`SpyreOpFuncs` methods are static with no external dependencies. They can be
called directly with dummy string arguments.

```python
def test_layernormscale_carries_eps():
    result = SpyreOpFuncs.layernormscale("x", 1e-5)
    assert isinstance(result, PointwiseOp)
    assert result.op == "layernormscale"
    assert result.op_info["constants"]["eps"] == 1e-5
```

#### 3.4.2 KernelSpec Construction

##### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `op: str`, `is_reduction: bool`, `dims: list[DimensionInfo]`, `args: Sequence[TensorArg \| ConstantArg]`, `scales: list[list[int]]`, `op_info: dict` | Collected from `SpyreKernel.store()` / `.store_reduction()` |
| **Output** | `KernelSpec` dataclass | Fields: `op`, `is_reduction`, `dimensions: list[int]`, `args`, `scales`, `op_info` |

##### Observed Data (from `layer_norm.log` lines 162–206)

**exx2 KernelSpec:**
```
op: exx2
is_reduction: True
dimensions: [256, 128]
scales: [[0, 1], [0, -1]]
op_info: {constants: {exx2scale: 0.0078125, useZeroMean: False},
          core_division: [[1,1,1],[1,1,1]], n_cores_used: 1}
args (2):
  [0] TensorArg(in,  [256,128], STL([2,256,64],  [1,0,1]))
  [1] TensorArg(out, [256,64],  STL([64,256,64], [1,0,-1]))
```

**layernormscale KernelSpec:**
```
op: layernormscale
is_reduction: False
dimensions: [256, 64]
scales: [[0, 1], [0, 1]]
op_info: {core_division: [[32,1,1],[32,1,1]], n_cores_used: 32,
          constants: {eps: 1e-05}}
args (2):
  [0] TensorArg(in,  [256,64], STL([64,256,64], [1,0,-1]))
  [1] TensorArg(out, [256,64], STL([64,256,64], [1,0,-1]))
```

**layernormnorm KernelSpec:**
```
op: layernormnorm
is_reduction: False
dimensions: [256, 128]
scales: [[0,1], [0,1], [0,1], [-1,0], [-1,0], [0,1]]
op_info: {core_division: [[1,1,1],×6], n_cores_used: 1}
args (6):
  [0] TensorArg(in,  [256,128], STL([2,256,64],  [1,0,1]))    ← input
  [1] TensorArg(in,  [256,64],  STL([64,256,64], [1,0,-1]))   ← mean
  [2] TensorArg(in,  [256,64],  STL([64,256,64], [1,0,-1]))   ← scale
  [3] TensorArg(in,  [128],     STL([2,64],      [0,0]))      ← weight
  [4] TensorArg(in,  [128],     STL([2,64],      [0,0]))      ← bias
  [5] TensorArg(out, [256,128], STL([2,256,64],  [1,0,1]))    ← output
```

##### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| `len(scales) == len(args)` | Any KernelSpec | `len(ks.scales) == len(ks.args)` |
| `len(scales[i]) == len(dimensions)` | Any KernelSpec | `all(len(s) == len(ks.dimensions) for s in ks.scales)` |
| Reduction has `is_reduction=True` | exx2 KernelSpec | `ks.is_reduction is True` |
| Pointwise has `is_reduction=False` | layernormscale KernelSpec | `ks.is_reduction is False` |
| Broadcast dims have scale `-1` | weight/bias in layernormnorm | `ks.scales[3] == [-1, 0]` and `ks.scales[4] == [-1, 0]` |
| Reduction dim has scale `-1` on output | exx2 output | `ks.scales[1] == [0, -1]` |
| layernormnorm has 5 inputs + 1 output | 6-arg KernelSpec | `sum(a.is_input for a in ks.args) == 5` and `sum(not a.is_input for a in ks.args) == 1` |
| dtype validation rejects bfloat16 | `TensorArg(dtype=bfloat16)` | `raises Unsupported` |
| FP32 allowed for layernorm ops | `op="exx2"` with `TensorArg(dtype=fp32)` | Does not raise (exx2 is in `SPYRE_FP32_OPS`) |

##### How to Test (needs `_C` module, no hardware)

`KernelSpec`, `TensorArg`, and `SpyreTensorLayout` are plain dataclasses.
Construct them directly.

```python
def test_kernelspec_scales_dimensions_match():
    stl = SpyreTensorLayout([256, 128], torch.float16)
    ks = KernelSpec(
        op="exx2", is_reduction=True, dimensions=[256, 128],
        args=[
            TensorArg(True, 0, torch.float16, torch.Size([256,128]), {}, stl),
            TensorArg(False, 1, torch.float16, torch.Size([256,64]), {},
                      SpyreTensorLayout([256,64], torch.float16, [0,1,-1])),
        ],
        scales=[[0, 1], [0, -1]], op_info={},
    )
    assert len(ks.scales) == len(ks.args)
    for s in ks.scales:
        assert len(s) == len(ks.dimensions)
```

---

### 3.5 Stage 5: SuperDSC Generation

**Source:** `torch_spyre/_inductor/runtime/async_compile.py` →
`torch_spyre/_inductor/codegen/superdsc.py` →
`torch_spyre/_inductor/codegen/compute_ops.py`

**What happens:** Each `KernelSpec` is unpacked into a kernel descriptor
dict and passed to `generate_sdsc()`, which produces a SuperDSC JSON dict
that the backend compiler (`dxp_standalone`) consumes.

This is the **highest-value testing seam** because it validates the entire
front-end compiler output without requiring the backend compiler or hardware.

#### Boundary Definition

| | Type | Description |
|---|------|-------------|
| **Input** | `pointers: dict[str, int]`, `kernel_descriptor: dict` | `kernel_descriptor` has keys: `name`, `op`, `reduction`, `dimensions`, `inputs` (list of dicts with `name`, `scale`, `device_layout`, `host_size`, `lx_addr`), `outputs` (same format), `op_info` |
| **Output** | `dict` (SuperDSC JSON) | JSON-serializable dict with top-level key = op name (e.g., `"exx2"`), containing `dscs_` array with `computeOp_` entries |

#### Observed Data (from `layer_norm.log` lines 209–570)

**Kernel 0: exx2**

Input kernel_descriptor:
```
name: sdsc_fused_layer_norm_0
op: exx2, reduction: True, dimensions: [256, 128]
inputs:  [{arg0, scale=[0,1], host=[256,128],
           device_size=[2,256,64], dim_map=[1,0,1]}]
outputs: [{arg1, scale=[0,-1], host=[256,64],
           device_size=[64,256,64], dim_map=[1,0,-1]}]
op_info: {constants:{exx2scale:0.0078125}, n_cores_used:1}
```

Output SuperDSC JSON (key properties):
```json
{"exx2": {
  "numCoresUsed_": 1,
  "dscs_": [{"computeOp_": [{
    "exUnit": "sfp",
    "opFuncName": "exx2",
    "dataFormat_": "SEN169_FP16",
    "inputLabeledDs": ["Tensor0-idx0", "Tensor1-idx1"],
    "outputLabeledDs": ["Tensor1-idx1"]
  }]}]
}}
```

**Kernel 1: layernormscale**

Input: `op: layernormscale, dimensions: [256,64], n_cores_used:32, constants:{eps:1e-5}`

Output: `{"layernormscale": {..., inputLabeledDs: ["Tensor0-idx0"], outputLabeledDs: ["Tensor1-idx1"]}}`

**Kernel 2: layernormnorm**

Input: `op: layernormnorm, dimensions: [256,128], 5 inputs + 1 output, n_cores_used:1`

Output: `{"layernormnorm": {..., inputLabeledDs: ["Tensor0-idx0".."Tensor4-idx4"], outputLabeledDs: ["Tensor5-idx5"]}}`

#### What to Test

| Test | Input | Assert on Output |
|------|-------|-----------------|
| JSON serializable | Any `generate_sdsc()` output | `json.dumps(sdsc)` does not raise |
| JSON round-trip | Any `generate_sdsc()` output | `json.loads(json.dumps(sdsc)) == sdsc` |
| Top-level key matches op | `op="exx2"` | `list(sdsc.keys()) == ["exx2"]` |
| Compute ops have `"dscs_"` key | non-data ops (exx2, layernormscale, layernormnorm) | `"dscs_" in sdsc[op]` |
| `opFuncName` matches op | All kernels | `sdsc[op]["dscs_"][0]["computeOp_"][0]["opFuncName"] == op` |
| `exUnit` is `"sfp"` for all SFP ops | exx2, layernormscale, layernormnorm | `computeOp_[0]["exUnit"] == "sfp"` |
| `dataFormat_` is `"SEN169_FP16"` for fp16 | fp16 inputs | `computeOp_[0]["attributes_"]["dataFormat_"] == "SEN169_FP16"` |
| Input/output tensor count matches | exx2: 2 tensors, layernormnorm: 6 tensors | `len(inputLabeledDs) + len(outputLabeledDs) == expected` |
| Regression via golden files | Known input → compare JSON against saved file | `actual_json == golden_json` |

#### How to Test (needs `_C` module, no hardware)

`generate_sdsc()` is a pure function. Construct input dicts by hand.

```python
def test_sdsc_exx2_structure():
    stl_in = SpyreTensorLayout([256, 128], torch.float16)
    stl_out = SpyreTensorLayout([256, 64], torch.float16, [0, 1, -1])

    sdsc = generate_sdsc(
        pointers={"arg0": 0x0, "arg1": 0x400000000},
        op="exx2", reduction=True, dimensions=[256, 128],
        name="test_exx2",
        inputs=[{"name": "arg0", "scale": [0, 1],
                 "device_layout": stl_in,
                 "host_size": torch.Size([256, 128]), "lx_addr": None}],
        outputs=[{"name": "arg1", "scale": [0, -1],
                  "device_layout": stl_out,
                  "host_size": torch.Size([256, 64]), "lx_addr": None}],
        op_info={"constants": {"exx2scale": 0.0078125, "useZeroMean": False},
                 "core_division": [[1,1,1],[1,1,1]], "n_cores_used": 1},
    )

    assert isinstance(sdsc, dict)
    json.dumps(sdsc)  # must not raise
    assert "exx2" in sdsc
    dsc = sdsc["exx2"]["dscs_"][0]
    assert dsc["computeOp_"][0]["opFuncName"] == "exx2"
    assert dsc["computeOp_"][0]["exUnit"] == "sfp"
```

---

### 3.6 Stage 6: Backend Compile + Runtime

**Source:** `torch_spyre/_inductor/runtime/async_compile.py` →
`dxp_standalone` → `torch_spyre/_inductor/runtime/kernel_runner.py`

**What happens:** Each `sdsc.json` is compiled by `dxp_standalone` into
`g2.graph.cbor`, then `SpyreSDSCKernelRunner.run()` calls `launch_kernel()`
to execute on the device.

#### Observed Data (from `layer_norm.log` lines 571–591)

```
Generating .../sdsc_fused_layer_norm_0_.../sdsc.json
Generating .../sdsc_fused_layer_norm_1_.../sdsc.json
Generating .../sdsc_fused_layer_norm_2_.../sdsc.json
RUN: sdsc_fused_layer_norm_0 .../g2.graph.cbor
RUN: sdsc_fused_layer_norm_1 .../g2.graph.cbor
RUN: sdsc_fused_layer_norm_2 .../g2.graph.cbor
Max delta Compiled Spyre vs. CPU: 0.025390625
```

#### What to Test

| Test | Input | Assert on Output | Requires |
|------|-------|-----------------|----------|
| `dxp_standalone` accepts JSON | `sdsc.json` from Stage 5 | `returncode == 0` | `dxp_standalone` binary (no hardware) |
| `convert_artifacts` produces CBOR | Compiled output dir | `g2.graph.cbor` exists | `dxp_standalone` binary |
| Numerical correctness | `launch_kernel(...)` result | `torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)` | Spyre hardware |

This is the only stage that requires hardware. With the preceding stages
validated, E2E tests here serve as a final sanity check, not the primary
debugging mechanism.

---

## 4. Testing Pyramid

```
                        ┌────────────────┐
                        │  E2E on HW     │ Numerical correctness
                        │  (Stage 6)     │ Needs Spyre device
                        └───────┬────────┘
                                │
                     ┌──────────┴──────────┐
                     │ SuperDSC Generation  │ Structural + golden file
                     │ (Stage 5)           │ Needs _C module
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
         ┌────┴────┐      ┌────┴────┐       ┌────┴────┐
         │ Kernel  │      │ Layout  │       │  Core   │
         │ Spec    │      │ Assign  │       │Division │
         │(Stage 4)│      │(Stg 3a) │       │(Stg 3b) │
         └────┬────┘      └────┬────┘       └────┬────┘
              │                │                  │         Needs _C
         ─────┼────────────────┼──────────────────┼─────────────────
              │                │                  │         Zero deps
         ┌────┴────┐      ┌────┴────┐       ┌────┴────┐
         │  Op     │      │ Decomp  │       │ core_   │
         │ Funcs   │      │  graph  │       │ split() │
         │(Stage 4)│      │(Stage 1)│       │(Stage 3)│
         └─────────┘      └─────────┘       └─────────┘
```

**Bottom row (zero dependencies, run anywhere):**
- SpyreOpFuncs: static methods, pure Python
- Decomposition graph inspection: FX tracing, pure Python
- `core_split()` / `multi_dim_core_split()`: pure math

**Middle row (needs `_C` module, no hardware):**
- Layout assignment: needs `SpyreTensorLayout` from `_C`
- KernelSpec construction: needs `SpyreTensorLayout`
- SuperDSC generation: needs `SpyreTensorLayout`, returns pure dicts

**Top row (needs hardware):**
- E2E numerical correctness

## 5. Recommended Phased Approach

### Phase 1: Zero-Dependency Tests (Immediate, ~2 days)

- **SpyreOpFuncs**: Every method returns correct op name and op_info.
- **`core_split()`**: Comprehensive input coverage.
- **Decomposition graph**: Verify `layer_norm` splits into 3 ops; verify
  non-decomposed ops pass through.

### Phase 2: KernelSpec + SuperDSC Tests (~1 week)

- **KernelSpec construction by hand**: Build for exx2, layernormscale,
  layernormnorm; validate structural invariants.
- **`generate_sdsc()` direct calls**: Build input dicts by hand; validate
  JSON structure, op names, tensor counts.
- **Golden files**: Capture known-good SuperDSC JSON for each layer_norm
  kernel; diff on subsequent runs.

### Phase 3: Layout + Core Division Refactoring (~2 weeks)

- **Extract layout logic** from `pointwise_layout()` / `reduction_layout()`
  into pure functions that take `SpyreTensorLayout` objects.
- **Extract core division logic** from `divide_pointwise_op()` /
  `divide_reduction_op()` into functions that take `(device_sizes, op_type,
  max_cores)`.
- **Add unit tests** for extracted functions.

### Phase 4: Pipeline Capture Harness (ongoing)

- **`capture_sdsc()` harness**: Run `torch.compile` and intercept all
  `KernelSpec`s and SuperDSC JSONs without invoking `dxp_standalone`.
- **Cross-seam contract tests**: Verify the output of one stage is valid
  input for the next.

## 6. Appendix: Complete Observed Data for LayerNorm

The following table shows the complete data at every boundary for
`F.layer_norm(x[256,128], [128], weight[128], bias[128], eps=1e-5)`:

| Stage | Function | Input Type | Input Data | Output Type | Output Data |
|-------|----------|-----------|------------|-------------|-------------|
| 1 Decompose | `layernorm_decomp()` | `torch.Tensor` + scalars | `x[256,128] fp16, [128], w[128], b[128], 1e-5` | 3 FX nodes | `exx2(x, 0.0078125, False)`, `layernormscale(mean, 1e-5)`, `layernormnorm(x, mean, scale, w, b)` |
| 2 Lower (exx2) | `lower_exx2()` | `TensorBox` + scalars | `x[256,128], 0.0078125, False` | `SpyreReduction` | `type="exx2", ranges=[256,64], op_info={exx2scale:0.0078125}` |
| 2 Lower (scale) | `lower_layernormscale()` | `TensorBox` + scalar | `x[256,64], 1e-5` | `Pointwise` | `ranges=[256,64]` |
| 2 Lower (norm) | `lower_layernormnorm()` | 5× `TensorBox` | `x[256,128], mean[256,64], scale[256,64], w[128], b[128]` | `Pointwise` | `ranges=[256,128]` |
| 3a Layout (buf0) | `reduction_layout()` | `SchedNodeArg` | `arg[0]: device=[2,256,64] dim_map=[1,0,1]` | `FixedTiledLayout` | `device=[64,256,64] dim_map=[1,0,-1]` (sparse) |
| 3a Layout (buf1) | `pointwise_layout()` | `SchedNodeArg` | `arg[0]: device=[64,256,64] dim_map=[1,0,-1]` | `FixedTiledLayout` | `device=[64,256,64] dim_map=[1,0,-1]` (propagated) |
| 3a Layout (buf2) | `pointwise_layout()` | 5× `SchedNodeArg` | `arg[0]:[2,256,64] dense, arg[1,2]:[64,256,64] sparse, arg[3,4]:[2,64]` | `FixedTiledLayout` | `device=[2,256,64] dim_map=[1,0,1]` (from arg[0]) |
| 3b Cores (buf0) | `divide_reduction_op()` | `max_cores=32` | exx2 reduction | `int + list` | `n_cores=1, [[1,1,1],[1,1,1]]` |
| 3b Cores (buf1) | `divide_pointwise_op()` | `max_cores=32` | `device_size=[64,256,64]` | `int + list` | `n_cores=32, [[32,1,1],[32,1,1]]` |
| 3b Cores (buf2) | `divide_pointwise_op()` | `max_cores=32` | broadcasts detected | `int + list` | `n_cores=1, [[1,1,1],×6]` |
| 4a OpFuncs (scale) | `SpyreOpFuncs.layernormscale()` | `TensorAccess` + scalar | `x=buf0, eps=1e-5` | `PointwiseOp` | `op="layernormscale", op_info={eps:1e-5}` |
| 4a OpFuncs (norm) | `SpyreOpFuncs.layernormnorm()` | 5× `TensorAccess` | `arg0..arg4` | `PointwiseOp` | `op="layernormnorm", args=[×5]` |
| 4b KernelSpec (exx2) | `create_kernel_spec()` | op + dims + args + scales | `"exx2", True, [256,128], 2 TensorArgs, [[0,1],[0,-1]]` | `KernelSpec` | `op="exx2", 2 args, op_info={exx2scale, core_div, n_cores}` |
| 4b KernelSpec (scale) | `create_kernel_spec()` | op + dims + args + scales | `"layernormscale", False, [256,64], 2 TensorArgs, [[0,1],[0,1]]` | `KernelSpec` | `op="layernormscale", 2 args, n_cores=32` |
| 4b KernelSpec (norm) | `create_kernel_spec()` | op + dims + args + scales | `"layernormnorm", False, [256,128], 6 TensorArgs, [[0,1],[0,1],[0,1],[-1,0],[-1,0],[0,1]]` | `KernelSpec` | `op="layernormnorm", 6 args, n_cores=1` |
| 5 SuperDSC (exx2) | `generate_sdsc()` → `generate_sfp_op()` | `pointers` + descriptor | `op="exx2", dims=[256,128], 1 in + 1 out` | `dict` | `{"exx2": {dscs_: [{computeOp_: [{opFuncName:"exx2", exUnit:"sfp"}]}]}}` |
| 5 SuperDSC (scale) | `generate_sdsc()` → `generate_sfp_op()` | `pointers` + descriptor | `op="layernormscale", dims=[256,64], 1 in + 1 out` | `dict` | `{"layernormscale": {..., opFuncName:"layernormscale"}}` |
| 5 SuperDSC (norm) | `generate_sdsc()` → `generate_sfp_op()` | `pointers` + descriptor | `op="layernormnorm", dims=[256,128], 5 in + 1 out` | `dict` | `{"layernormnorm": {..., inputLabeledDs:[×5], outputLabeledDs:[×1]}}` |
| 6 Runtime | `launch_kernel()` | `g2.graph.cbor` + tensors | 3 kernels executed in sequence | `torch.Tensor` | `result[256,128] fp16, delta=0.0254 vs CPU` |
