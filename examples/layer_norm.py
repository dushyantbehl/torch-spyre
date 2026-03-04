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
Layer Norm example — demonstrates the full Spyre compilation pipeline.

This is the richest single-op example because F.layer_norm decomposes into
three separate Spyre custom ops, each producing its own kernel:

  F.layer_norm(input, [N], weight, bias, eps)
    → spyre.layer_norm(input, [N], weight, bias, eps)       # Seam 1: interception
      → spyre.exx2(input, 1/N, False)                       # Seam 1: decomposition (Reduction)
      → spyre.layernormscale(exx2_out, eps)                  # Seam 1: decomposition (Pointwise)
      → spyre.layernormnorm(input, mean, scale, weight, bias)# Seam 1: decomposition (Pointwise)

Each of the three ops then flows through:
  Seam 2: Lowering (FX op → Inductor IR)
  Seam 3: Layout Assignment (FixedLayout → FixedTiledLayout)
  Seam 4: Core Division Planning
  Seam 5: SpyreOpFuncs (op name mapping)
  Seam 6: KernelSpec Construction
  Seam 7: SuperDSC JSON Generation

Run with pipeline tracing enabled to see every stage:
    SPYRE_DEBUG_TRACE=1 python examples/layer_norm.py
"""

import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Create input tensor and layer norm parameters
# Shape: [256, 128] — 256 tokens, 128-dim embedding
# normalized_shape = [128] means we normalize over the last dimension
x = torch.rand(256, 128, dtype=torch.float16)
weight = torch.rand(128, dtype=torch.float16)
bias = torch.zeros(128, dtype=torch.float16)

# Compute layer_norm on the CPU for reference
cpu_result = F.layer_norm(x, [128], weight=weight, bias=bias)

# Send tensors to device
x_device = x.to(DEVICE)
weight_device = weight.to(DEVICE)
bias_device = bias.to(DEVICE)

# Compile and run on Spyre
# This triggers the full pipeline:
#   F.layer_norm → spyre.layer_norm → [exx2, layernormscale, layernormnorm]
#   Each op: lowering → layout → core division → KernelSpec → SuperDSC → dxp_standalone
compiled_ln = torch.compile(
    lambda inp, w, b: F.layer_norm(inp, [128], weight=w, bias=b)
)
compiled_result = compiled_ln(x_device, weight_device, bias_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
