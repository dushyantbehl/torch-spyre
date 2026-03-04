# Copyright 2025 The Torch-Spyre Authors.
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

"""
Pipeline debug tracing for torch-spyre.

Enable by setting the environment variable SPYRE_DEBUG_TRACE=1 before running.
Each pipeline seam prints its input and output with a clear header.

Seams traced (matching component_testing_strategy.md):
  Seam 1  - Decompositions      (decompositions.py)
  Seam 2  - Lowering             (lowering.py)
  Seam 3  - Layout Assignment    (stickify.py)
  Seam 4  - Core Division        (core_division.py)
  Seam 5  - SpyreOpFuncs         (spyre_kernel.py)
  Seam 6  - KernelSpec           (spyre_kernel.py)
  Seam 7  - SuperDSC Generation  (superdsc.py / async_compile.py)
"""

import json
import os
import textwrap

_ENABLED: bool | None = None


def is_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SPYRE_DEBUG_TRACE", "0") == "1"
    return _ENABLED


# ── Formatting helpers ──────────────────────────────────────────────

_SEPARATOR = "=" * 80
_THIN_SEP = "-" * 60


def _header(seam: str, title: str) -> str:
    return f"\n{_SEPARATOR}\n[SPYRE TRACE] {seam}: {title}\n{_SEPARATOR}"


def _sub(label: str) -> str:
    return f"\n{_THIN_SEP}\n  {label}\n{_THIN_SEP}"


def _indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(str(text), prefix)


# ── Seam 1: Decompositions ─────────────────────────────────────────

def trace_decomposition(func_name: str, input_info: dict, output_info: str):
    """Trace an F.xxx → spyre.xxx interception in decompositions.py."""
    if not is_enabled():
        return
    print(_header("Seam 1 — Decompositions", f"{func_name}"))
    print(f"  Input:")
    for k, v in input_info.items():
        print(f"    {k}: {v}")
    print(f"  Output → {output_info}")


# ── Seam 2: Lowering ───────────────────────────────────────────────

def trace_lowering(op_name: str, input_info: dict, output_info: dict):
    """Trace a lowering function (FX op → Inductor IR node)."""
    if not is_enabled():
        return
    print(_header("Seam 2 — Lowering", f"{op_name}"))
    print("  Input:")
    for k, v in input_info.items():
        print(f"    {k}: {v}")
    print("  Output (IR Node):")
    for k, v in output_info.items():
        print(f"    {k}: {v}")


# ── Seam 3: Layout Assignment ──────────────────────────────────────

def trace_layout_input(node_name: str, node_type: str, input_layouts: list[dict]):
    """Trace the input to layout assignment for a single node."""
    if not is_enabled():
        return
    print(_header("Seam 3 — Layout Assignment", f"node={node_name}"))
    print(f"  Node type: {node_type}")
    if input_layouts:
        print("  Input arg layouts:")
        for i, info in enumerate(input_layouts):
            print(f"    arg[{i}]:")
            for k, v in info.items():
                print(f"      {k}: {v}")


def trace_layout_output(node_name: str, output_layout: dict):
    """Trace the output of layout assignment for a single node."""
    if not is_enabled():
        return
    print(_sub(f"Layout Assignment output for node={node_name}"))
    for k, v in output_layout.items():
        print(f"    {k}: {v}")


# ── Seam 4: Core Division ──────────────────────────────────────────

def trace_core_division(node_name: str, op_type: str, info: dict):
    """Trace the core division decision for a node."""
    if not is_enabled():
        return
    print(_header("Seam 4 — Core Division", f"node={node_name}"))
    print(f"  Op type: {op_type}")
    for k, v in info.items():
        print(f"  {k}: {v}")


# ── Seam 5: SpyreOpFuncs ───────────────────────────────────────────

def trace_opfunc(op_name: str, spyre_op: str, args_info: list[str], op_info: dict | None = None):
    """Trace a SpyreOpFuncs mapping (PyTorch op → Spyre op name)."""
    if not is_enabled():
        return
    print(_header("Seam 5 — SpyreOpFuncs", f"{op_name} → {spyre_op}"))
    print(f"  Arguments: {args_info}")
    if op_info:
        print(f"  op_info: {op_info}")


# ── Seam 6: KernelSpec Construction ────────────────────────────────

def trace_kernel_spec(kernel_spec):
    """Trace a completed KernelSpec."""
    if not is_enabled():
        return
    print(_header("Seam 6 — KernelSpec", f"op={kernel_spec.op}"))
    print(f"  op: {kernel_spec.op}")
    print(f"  is_reduction: {kernel_spec.is_reduction}")
    print(f"  dimensions: {kernel_spec.dimensions}")
    print(f"  scales: {kernel_spec.scales}")
    print(f"  op_info: {kernel_spec.op_info}")
    print(f"  args ({len(kernel_spec.args)}):")
    for i, arg in enumerate(kernel_spec.args):
        print(f"    [{i}] {arg!r}")


# ── Seam 7: SuperDSC JSON Generation ──────────────────────────────

def trace_sdsc_input(kernel_name: str, kernel_descriptor: dict):
    """Trace the input to generate_sdsc (the kernel descriptor)."""
    if not is_enabled():
        return
    print(_header("Seam 7 — SuperDSC Generation", f"kernel={kernel_name}"))
    print("  Kernel descriptor:")
    # Pretty-print but replace non-serializable objects with repr
    for k, v in kernel_descriptor.items():
        if k in ("inputs", "outputs"):
            print(f"  {k}:")
            for item in v:
                print(f"    - name: {item.get('name')}")
                print(f"      scale: {item.get('scale')}")
                print(f"      host_size: {item.get('host_size')}")
                stl = item.get("device_layout")
                if stl is not None:
                    print(f"      device_layout.device_size: {stl.device_size}")
                    print(f"      device_layout.dim_map: {stl.dim_map}")
                    print(f"      device_layout.device_dtype: {stl.device_dtype}")
                print(f"      lx_addr: {item.get('lx_addr')}")
        else:
            print(f"  {k}: {v}")


def trace_sdsc_output(kernel_name: str, sdsc_json: dict):
    """Trace the output SuperDSC JSON."""
    if not is_enabled():
        return
    print(_sub(f"SuperDSC JSON output for kernel={kernel_name}"))
    try:
        formatted = json.dumps(sdsc_json, indent=2, default=str)
        # Limit output to avoid flooding the terminal
        lines = formatted.split("\n")
        if len(lines) > 80:
            print(_indent("\n".join(lines[:40])))
            print(f"    ... ({len(lines) - 80} lines omitted) ...")
            print(_indent("\n".join(lines[-40:])))
        else:
            print(_indent(formatted))
    except Exception as e:
        print(f"    (Could not serialize SDSC JSON: {e})")
        print(f"    Keys: {list(sdsc_json.keys()) if isinstance(sdsc_json, dict) else type(sdsc_json)}")
