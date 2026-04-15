# Complete Flow of `torch.add()` Through Torch-Spyre

## Overview

Torch-spyre is an **out-of-tree PyTorch backend** for the IBM Spyre AI Accelerator. It registers itself as a first-class PyTorch device named `"spyre"` using PyTorch's **PrivateUse1** mechanism. There are **two distinct execution paths** for any operation like `torch.add()`:

1. **Eager mode** — direct `torch.add(a, b)` on Spyre tensors
2. **Compiled mode** — inside `torch.compile()`, which is the primary/optimized path

---

## Layer 0: Backend Registration & Initialization

When PyTorch discovers the backend (via the `pyproject.toml` entry point), `torch_spyre._autoload()` fires:

```
pyproject.toml entry: torch_spyre:_autoload
    │
    ├─ torch.utils.rename_privateuse1_backend("spyre")     # PrivateUse1 → "spyre"
    ├─ torch._register_device_module("spyre", module)       # Register torch.spyre module
    ├─ _patch_tensor_for_spyre()                            # Monkey-patch Tensor.__repr__, .to()
    ├─ import torch_spyre.codegen_ops                       # Load auto-generated eager wrappers
    ├─ _register_spyre_dispatchkey_kernels_permanently()    # PrivateUse1 dispatch keys
    └─ _light_autoload() → enable_spyre_compile_fx_wrapper()# Hook into torch.compile
```

On **first device access**, `_lazy_init()` loads the C++ extension and starts the hardware runtime:

```
_lazy_init()
    ├─ import torch_spyre._C               # Load C++ extension (pybind11)
    └─ _C.start_runtime()                   # flex::initializeRuntime() → connects to Spyre HW
```

**Key files:**
- `torch_spyre/__init__.py` — `_autoload()`, `_SpyreImpl._lazy_init()`
- `torch_spyre/csrc/module.cpp` — `startRuntime()` calls `flex::initializeRuntime()`

---

## Layer 1: Tensor Creation & Memory Layout

Before `torch.add()` can run, tensors must exist on the Spyre device. Spyre uses a **stick-based memory layout** (128-byte aligned chunks):

```
torch.empty([64, 256], dtype=torch.float16, device="spyre")
    │
    ├─ SpyreTensorLayout(size=[64,256], dtype=fp16)
    │      └─ Computes device dimensions, dim_map, stride_map
    │      └─ elems_per_stick = 128 / sizeof(fp16) = 64
    │
    ├─ SpyreAllocator::allocate(size_bytes)
    │      └─ flex::DeviceMemoryAllocator::TryAllocate()  → actual HW memory
    │      └─ Returns SharedOwnerCtx { DeviceMemoryAllocationPtr }
    │
    └─ SpyreTensorImpl { storage, spyre_layout, dma_sizes, dma_strides }
```

**Type mappings** (`types_mapping.h`):
| PyTorch dtype | CPU format | Device format |
|---|---|---|
| float16 | IEEE_FP16 | **SEN169_FP16** (Spyre-native) |
| float32 | IEEE_FP32 | IEEE_FP32 |
| bfloat16 | BFLOAT16 | BFLOAT16 |
| int64 | IEEE_INT64 | **IEEE_INT32** (downcast!) |

**Data movement** (host <-> device) uses DMA graphs through `sendnn`:

```
tensor.to("spyre")
    └─ spyre_copy_from()
        └─ create_dma_graph()
            ├─ Stage 1: SenDataTransfer execution graph
            ├─ Stage 2: SenSuperNodeV2 wrapper
            └─ Stage 3: LoadGraph → CompileGraph → ParseGraph → gl.Copy()
```

**Key files:**
- `torch_spyre/csrc/spyre_tensor_impl.cpp` — `SpyreTensorLayout`
- `torch_spyre/csrc/spyre_allocator.cpp` — `SpyreAllocator::allocate()`
- `torch_spyre/csrc/spyre_mem.cpp` — `spyre_empty()`, `create_dma_graph()`, `spyre_copy_from()`

---

## Path A: Eager Mode (`torch.add(a, b)` directly)

For eager mode, `torch.add()` is **not explicitly registered** for the Spyre device:

- Not in `torch_spyre/ops/eager.py` (only `mm`, `fill_`, `zero_`, `normal_`, etc. are registered)
- Not in `torch_spyre/ops/fallbacks.py` (only `arange`, `sin`, `cos`, `embedding`, etc.)
- Not in `spyre_decompositions_via_dispatchkey` in `decompositions.py`

**Dispatch chain:**

```
torch.add(spyre_tensor_a, spyre_tensor_b)
    │
    ├─ PyTorch dispatcher checks PrivateUse1 dispatch key
    │      └─ No kernel registered for aten::add.Tensor on PrivateUse1
    │
    └─ Falls back to CPU:
           ├─ Move both tensors to CPU (DMA device→host)
           ├─ Execute add on CPU
           └─ Move result back to Spyre (DMA host→device)
```

This is **slow** — two round-trips of data movement. Eager mode is not the intended fast path for pointwise ops like `add`.

---

## Path B: Compiled Mode (`torch.compile()`) — THE PRIMARY PATH

This is where `torch.add()` gets fully accelerated on Spyre hardware. It passes through **8 distinct layers**:

### B1. Dynamo Tracing (Graph Capture)

```python
@torch.compile
def f(a, b):
    return torch.add(a, b)
```

PyTorch Dynamo traces the function into an FX graph:

```
%arg0 : Tensor[spyre, fp16, [64, 256]]
%arg1 : Tensor[spyre, fp16, [64, 256]]
%add  = torch.ops.aten.add.Tensor(%arg0, %arg1)
return (%add,)
```

The `_wrapper()` in `torch_spyre/_inductor/__init__.py` intercepts `compile_fx` and injects Spyre-specific context.

### B2. Decomposition

**File:** `torch_spyre/_inductor/decompositions.py`

The `enable_spyre_decompositions()` context manager applies Spyre-specific decompositions. For `torch.add()`:

- **No custom Spyre decomposition exists** — add is a basic pointwise op
- It passes through unchanged as `aten.add.Tensor`
- Complex ops like `rms_norm`, `layer_norm`, `gelu`, `softplus` ARE decomposed here into simpler primitives or routed to custom Spyre ops

### B3. Lowering (FX Graph -> Inductor IR)

**File:** `torch_spyre/_inductor/lowering.py`

The `enable_spyre_lowerings()` context manager registers Spyre-specific lowerings. For `torch.add()`:

- **No custom Spyre lowering exists** — add uses PyTorch Inductor's built-in pointwise lowering
- The in-tree Inductor creates a **Pointwise IR node**:

```python
# Conceptual result from torch._inductor.lowering
Pointwise.create(
    device="spyre",
    dtype=torch.float16,
    inner_fn=lambda index: ops.add(x_loader(index), y_loader(index)),
    ranges=[64, 256],
)
```

Custom lowerings exist for `mm`, `bmm`, `gelu`, `softplus`, `clamp`, `mean.dim`, etc.

### B4. Scheduling

**File:** `torch_spyre/_inductor/scheduler.py` — `SuperDSCScheduling`

The custom scheduler takes the Pointwise IR node and invokes code generation:

```python
class SuperDSCScheduling:
    def codegen_node(self, node):
        # Creates a SpyreKernel context
        # Calls node.run() which invokes the inner_fn through the ops handler
        # The ops handler routes to SpyreOpFuncs
```

### B5. Kernel Code Generation (IR -> OpSpec)

**File:** `torch_spyre/_inductor/spyre_kernel.py`

This is the core translation layer. When the Pointwise node's `inner_fn` executes:

1. **`SpyreOpFuncs.add(a, b)`** :
   ```python
   @staticmethod
   def add(a, b):
       return PointwiseOp("add", [a, b])
   ```

2. **`SpyreKernel.store()`** receives the `PointwiseOp` and creates:
   ```python
   OpSpec(
       op='add',
       is_reduction=False,
       iteration_space={i: (64, 1), j: (256, 1)},
       args=[
           TensorArg(is_input=True, arg_index=0, device_dtype=SEN169_FP16, ...),
           TensorArg(is_input=True, arg_index=1, device_dtype=SEN169_FP16, ...),
           TensorArg(is_input=False, arg_index=2, device_dtype=SEN169_FP16, ...),  # output
       ],
   )
   ```

3. **`codegen_kernel()`** serializes this OpSpec list into Python source code.

**Key files:**
- `torch_spyre/_inductor/op_spec.py` — `OpSpec` and `TensorArg` dataclasses
- `torch_spyre/_inductor/ir.py` — `SpyreReduction`, `FixedTiledLayout`

### B6. Python Wrapper Code Generation

**File:** `torch_spyre/_inductor/wrapper.py` — `SpyrePythonWrapperCodegen`

The wrapper generates the final executable Python module:

```python
# Generated compiled module (conceptual):
sdsc_add_0 = async_compile.sdsc('sdsc_add_0', [
    OpSpec(op='add', is_reduction=False, iteration_space={...}, args=[...])
])

def call(args):
    t0, t1 = args
    t_out = torch.ops.spyre.empty(...)
    sdsc_add_0.run(t0, t1, t_out)
    return (t_out,)
```

### B7. Async Compilation (OpSpec -> SuperDSC -> Machine Code)

**File:** `torch_spyre/execution/async_compile.py` — `SpyreAsyncCompile.sdsc()`

```
OpSpec list
    │
    ├─ generate_bundle(kernel_name, output_dir, op_specs)
    │      └─ Converts OpSpec → SuperDSC JSON with:
    │           "opfunc": "add"
    │           "num_inputs": 2
    │           "data_format": "SEN169_FP16"
    │           "iteration_space": core division info
    │
    ├─ subprocess("dxp_standalone", "--bundle", output_dir)
    │      └─ Spyre backend compiler: JSON → g2.graph.cbor (binary)
    │
    ├─ convert_artifacts(output_dir)
    │      └─ Post-process compiled artifacts
    │
    └─ Returns SpyreSDSCKernelRunner
```

**Key files:**
- `torch_spyre/_inductor/codegen/superdsc.py` — SuperDSC JSON generation
- `torch_spyre/_inductor/codegen/bundle.py` — Bundle packaging
- `torch_spyre/_inductor/core_division.py` — Multi-core work division (up to 32 cores)

### B8. Hardware Execution

**File:** `torch_spyre/execution/kernel_runner.py` — `SpyreSDSCKernelRunner`

```python
class SpyreSDSCKernelRunner:
    def run(self, *args):
        g2 = os.path.join(self.code_dir, "g2.graph.cbor")
        launch_kernel(g2, list(args))  # → C++ _C.launch_kernel()
```

**C++ side** (`module.cpp` — `launchKernel()`):

```
launch_kernel(g2_path, [tensor_a, tensor_b, tensor_out])
    │
    ├─ Load g2.graph.cbor via sendnn::GraphLoader
    ├─ Compile graph on Spyre device
    ├─ Create sendnn::ConstTensor for inputs (attach DeviceMemoryAllocationPtr)
    ├─ Create sendnn::Tensor for outputs (attach DeviceMemoryAllocationPtr)
    └─ gl.Compute(outputs, inputs)  → executes on Spyre hardware (32 cores)
```

---

## Complete End-to-End Flow Diagram

```
USER CODE:
    @torch.compile
    def f(a, b): return torch.add(a, b)
    result = f(spyre_tensor_a, spyre_tensor_b)

  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 0: BACKEND REGISTRATION                                  │
  │  pyproject.toml → _autoload() → PrivateUse1 → "spyre"          │
  │  _lazy_init() → _C.start_runtime() → flex::initializeRuntime() │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 1: DYNAMO TRACING                                        │
  │  Captures FX graph: %add = aten.add.Tensor(%arg0, %arg1)       │
  │  _inductor/__init__.py wraps compile_fx for Spyre context       │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 2: DECOMPOSITION                                         │
  │  _inductor/decompositions.py → enable_spyre_decompositions()    │
  │  torch.add: NO custom decomposition → passes through unchanged  │
  │  (gelu, rms_norm, softplus etc. ARE decomposed here)            │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 3: LOWERING (FX → Inductor IR)                           │
  │  _inductor/lowering.py → enable_spyre_lowerings()               │
  │  torch.add: NO custom lowering → uses in-tree Pointwise IR      │
  │  Result: Pointwise(inner_fn=ops.add(x[i], y[i]))               │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 4: SCHEDULING                                            │
  │  _inductor/scheduler.py → SuperDSCScheduling.codegen_node()     │
  │  Creates SpyreKernel context, sets up SpyreKernelOpsHandler     │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 5: KERNEL CODEGEN (Inductor IR → OpSpec)                 │
  │  _inductor/spyre_kernel.py                                      │
  │  SpyreOpFuncs.add(a,b) → PointwiseOp("add", [a,b])            │
  │  SpyreKernel.store() → OpSpec(op='add', args=[in0,in1,out])    │
  │  codegen_kernel() → Python source of OpSpec list                │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 6: WRAPPER CODEGEN                                       │
  │  _inductor/wrapper.py → SpyrePythonWrapperCodegen               │
  │  Generates: async_compile.sdsc('sdsc_add_0', [OpSpec(...)])     │
  │  Generates: sdsc_add_0.run(t0, t1, t_out)                      │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 7: ASYNC COMPILATION (OpSpec → SuperDSC → Machine Code)  │
  │  execution/async_compile.py → SpyreAsyncCompile.sdsc()          │
  │  generate_bundle() → SuperDSC JSON {"opfunc":"add",...}         │
  │  dxp_standalone --bundle → g2.graph.cbor (compiled binary)      │
  │  convert_artifacts() → final artifact                           │
  │  Returns SpyreSDSCKernelRunner                                  │
  └─────────────────────┬───────────────────────────────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 8: HARDWARE EXECUTION                                    │
  │  execution/kernel_runner.py → SpyreSDSCKernelRunner.run()       │
  │  → _C.launch_kernel(g2.graph.cbor, [t0, t1, t_out])            │
  │  → sendnn::GraphLoader → Load → Compile → Parse                │
  │  → SetSpyreData(DeviceMemoryAllocationPtr) on each tensor       │
  │  → gl.Compute(outputs, inputs) → Spyre HW (up to 32 cores)    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Insights

1. **Eager mode is a fallback path** for ops like `add` — it round-trips through CPU. The compiled path (`torch.compile`) is the real accelerated path.

2. **`torch.add()` needs no custom handling** in decomposition or lowering because it's a simple pointwise op. PyTorch Inductor's built-in lowering creates a `Pointwise` IR node, and Spyre's `SpyreOpFuncs.add()` knows how to map it to the hardware's native `add` opfunc.

3. **The compilation pipeline** is: FX Graph -> Decomposition -> Inductor IR (Pointwise) -> OpSpec -> SuperDSC JSON -> `dxp_standalone` compiler -> `g2.graph.cbor` binary -> `launch_kernel()` on hardware.

4. **Data lives in stick-based layout** (128-byte aligned) on the device, with `SEN169_FP16` as the native float16 format. All host<->device data movement goes through DMA graphs built with `sendnn`.

5. **The sendnn/flex stack** is the hardware abstraction layer — `sendnn::GraphLoader` compiles and executes graphs, `flex::Runtime` manages device memory and streams.

6. **Multi-core execution**: Spyre supports up to 32 cores, with work division handled by `core_division.py` during the OpSpec -> SuperDSC JSON translation.
