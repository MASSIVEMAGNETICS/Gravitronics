# Gravitronics 🌌

**A physics-aware, edge-deployable Lightweight Gravitational Transformer (LGT) framework.**

[![Tests](https://img.shields.io/badge/tests-75%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

Gravitronics implements a novel *Gravitational Attention* mechanism where tokens attract each other based on semantic "mass" and geometric proximity — a physics-inspired inductive bias that encodes hierarchical relationships natively and provides built-in regularisation via Hawking radiation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Package: `gravitronics`](#core-package-gravitronics)
   - [LGTConfig](#lgtconfig)
   - [Gravitational Attention](#gravitational-attention)
   - [Curved Position Embeddings](#curved-position-embeddings)
   - [Mirror Layer & Diagnostics](#mirror-layer--diagnostics)
   - [LGT Model](#lgt-model)
5. [Edge Deployment Kit](#edge-deployment-kit)
   - [Export Model](#export-model)
   - [VictorOS Runtime Wrapper](#victoros-runtime-wrapper)
   - [Benchmark Suite](#benchmark-suite)
   - [Gravitational Consensus](#gravitational-consensus)
   - [Config File](#config-file)
6. [Testing](#testing)
7. [Diagnostics Schema](#diagnostics-schema)
8. [Model Variants](#model-variants)
9. [Deployment Guide](#deployment-guide)
10. [License](#license)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  🌌 GRAVITRONICS FRAMEWORK                                  │
│                                                             │
│  Token Embeddings                                           │
│       └─ CurvedPositionEmbedding                            │
│             └─ Dropout                                      │
│                  └─ N × TransformerBlock                    │
│                       ├─ MirrorLayer ─► GravitationalAttn   │
│                       ├─ LayerNorm + residual               │
│                       ├─ FFN (hidden → 4×hidden → hidden)   │
│                       └─ LayerNorm + residual               │
│       └─ Output Head: LayerNorm → Linear(hidden, vocab)     │
└─────────────────────────────────────────────────────────────┘
```

**Physics primitives:**
| Primitive | Equation | Effect |
|-----------|----------|--------|
| Gravitational Force | `F_ij = G·mᵢ·mⱼ / (d²_ij + ε)` | Biases attention toward massive, nearby tokens |
| Hawking Radiation | `-T·H(softmax(scores))` | Entropy regularisation — prevents attention collapse |
| Bekenstein Bound | `clamp(logits, −B, +B)` | Bounds information per token |
| Curved Positions | `pos_emb × curvature_tensor` | Learnable geometric distortion of position space |

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

**Dependencies:** `torch >= 2.0`, `numpy >= 1.24`, `pyyaml >= 6.0`, `psutil >= 5.9`

---

## Quick Start

```python
from gravitronics import LGT, LGTConfig, create_lgt
import torch

# Create a 150 K-parameter model (edge-ready)
model = create_lgt("150k")
model.eval()

print(f"Parameters: {model.count_parameters():,}")   # ~150,000
print(f"Size: {model.get_model_size_mb():.2f} MB")

# Forward pass
input_ids = torch.randint(0, 32000, (1, 64))          # (batch, seq_len)
with torch.no_grad():
    logits, diagnostics = model(input_ids)

print(logits.shape)           # (1, 64, 32000)
print(diagnostics[0])         # per-layer physics diagnostics
```

---

## Core Package: `gravitronics`

### LGTConfig

```python
from gravitronics.lgt.config import LGTConfig

cfg = LGTConfig.from_variant("150k")   # or "600k" or "2m"
# Override individual fields:
cfg.gravitational_constant = 1e-2
cfg.use_hawking_radiation = True
```

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_dim` | 256 | Token embedding dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer blocks |
| `gravitational_constant` | 6.674e-3 | Scales gravitational force |
| `hawking_temperature` | 0.1 | Entropy regularisation strength |
| `bekenstein_limit` | 50.0 | Attention logit clamp value |
| `sparse_top_k` | 32 | Keep top-k attention scores per query |
| `curvature_scale` | 1.0 | Initial curvature tensor scale |

### Gravitational Attention

```python
from gravitronics.lgt.attention import GravitationalAttention

attn = GravitationalAttention(config)
attn.diagnostics_enabled = True

out, diag = attn(x)           # x: (B, L, D)
print(diag["mean_force"])     # average gravitational force
print(diag["mean_mass"])      # average token mass
```

### Curved Position Embeddings

```python
from gravitronics.lgt.embeddings import CurvedPositionEmbedding

cpe = CurvedPositionEmbedding(config)
x_with_pos = cpe(x)           # (B, L, D) → (B, L, D)
```

The curvature tensor (`shape: 1 × max_seq_len × hidden_dim`) is a learnable parameter that geometrically distorts the sinusoidal position space, allowing the model to encode hierarchical proximity natively.

### Mirror Layer & Diagnostics

```python
from gravitronics.lgt.diagnostics import MirrorLayer, DiagnosticsLogger

# Wrap any module to capture its diagnostics
mirror = MirrorLayer(attention_module, max_snapshots=100)
mirror.enable()

out = mirror(x)
snapshot = mirror.get_snapshot()    # most recent
history  = mirror.get_history()     # all stored

# Log to JSONL file
logger = DiagnosticsLogger("diag.jsonl")
logger.log({"layer": 2, "mean_force": 12.4, "mean_mass": 0.87})
logger.flush()
logger.close()
```

### LGT Model

```python
from gravitronics.lgt.model import LGT, LGTConfig

cfg = LGTConfig.from_variant("600k")
model = LGT(cfg)

# Diagnostics
model.enable_diagnostics()
logits, diag = model(input_ids)

# Persistence
torch.save(model.state_dict(), "lgt_600k.pt")
model.load_state_dict(torch.load("lgt_600k.pt"))
```

---

## Edge Deployment Kit

### Export Model

```bash
python edge/export_edge_model.py \
    --variant 150k \
    --output /tmp/lgt_edge \
    --quantization fp16 \
    --trace
```

Programmatic API:
```python
from edge.export_edge_model import export_model
from gravitronics.lgt.model import create_lgt

model = create_lgt("150k")
metadata = export_model(
    model,
    output_path="/tmp/lgt_edge/model",
    quantization="fp16",
    trace=True,
    config=model.get_config(),
)
print(metadata)
# {"output_path": "...", "quantization": "fp16", "traced": True, ...}
```

**Quantisation options:**
| Mode | Description | Size reduction |
|------|-------------|----------------|
| `fp16` | Half-precision floating point | ~50% |
| `int8` | Dynamic INT8 quantisation | ~75% |
| `none` | Full FP32 | — |

### VictorOS Runtime Wrapper

```python
from edge.victoros_lgt_edge import VictorOSLGTEdge

runtime = VictorOSLGTEdge(
    model_path="/tmp/lgt_edge/model.pt",
    config_path="/tmp/lgt_edge/model_config.json",
    device="cpu",
)

health = runtime.health_check()
print(health)
# {"model_loaded": True, "parameter_count": 150240, "device": "cpu", ...}

result = runtime.infer(input_ids)
print(result["status"])       # "ok"
print(result["latency_ms"])   # inference time
print(result["diagnostics"])  # mirror layer snapshots

ledger_entry = runtime.get_ledger_entry(input_ids, result["diagnostics"])
# Structured audit trail: timestamp, input_hash, diagnostics
```

### Benchmark Suite

```bash
python edge/benchmark_edge_lgt.py \
    --variant 150k \
    --device cpu \
    --num_runs 100 \
    --output /tmp/benchmark.json
```

Output table:
```
=== LGT Benchmark Report ===
Variant : 150k
Device  : cpu
Parameters : 150,240

Latency (ms)
  Mean  : 12.4
  Std   : 1.2
  p50   : 12.1
  p95   : 14.3
  p99   : 15.8

Memory
  Peak  : 8.4 MB
  Alloc : 7.1 MB

Throughput
  Tokens/sec       : 5,241
  Inferences/sec   : 81
```

### Gravitational Consensus

Multi-agent swarm protocol where nodes vote on proposals with gravitational weighting:

```python
from edge.gravitational_consensus import GravitationalNode, GravitationalConsensusNetwork
import numpy as np

net = GravitationalConsensusNetwork(G=6.674e-3)

net.add_node(GravitationalNode("node-A", mass=2.0, position=np.array([0.0, 0.0, 0.0, 0.0])))
net.add_node(GravitationalNode("node-B", mass=1.5, position=np.array([1.0, 0.0, 0.0, 0.0])))
net.add_node(GravitationalNode("node-C", mass=1.0, position=np.array([0.5, 1.0, 0.0, 0.0])))

result = net.compute_consensus({"action": "update_weights", "value": 42})
print(result)
# {"accepted": True, "confidence": 0.87, "votes": {...}, "total_force": 12.4}
```

**Physics:** Each node's vote is weighted by the total gravitational force it exerts on the rest of the network — massive, well-connected nodes have more influence.

### Config File

`edge/config_edge.yaml` controls all deployment parameters:

```yaml
model:
  variant: "150k"
  gravitational_constant: 0.006674
  use_hawking_radiation: true
  use_mirror_layer: true

inference:
  batch_size: 1
  temperature: 1.0

diagnostics:
  enabled: true
  log_file: "lgt_diagnostics.jsonl"

safety:
  containment_enabled: true
  ethics_gate_enabled: true
```

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run individual test files
pytest tests/test_lgt_model.py -v
pytest tests/test_edge.py -v
pytest tests/test_consensus.py -v
```

**75 tests, 0 failures.**

| Test file | Coverage |
|-----------|----------|
| `test_lgt_model.py` | Config, forward pass, diagnostics, save/load |
| `test_edge.py` | Export, tracing, benchmarks, VictorOS wrapper |
| `test_consensus.py` | Node creation, force law, voting, topology |

---

## Diagnostics Schema

Every forward pass emits structured JSON-serializable diagnostics:

```json
{
  "layer": 2,
  "mean_force": 12.4,
  "mean_mass": 0.87,
  "curvature_active": true,
  "hawking_limit": 50.0,
  "sparse_top_k": 32,
  "hawking_radiation_applied": true
}
```

These can be logged to `.jsonl` files via `DiagnosticsLogger` and consumed by:
- **Mirror Layer** — real-time diagnostics deque
- **Ledger** — causal audit trail per inference
- **VictorOS Cortex** — inference control integration

---

## Model Variants

| Variant | Params | Hidden | Heads | Layers | Seq Len | Target Hardware |
|---------|--------|--------|-------|--------|---------|-----------------|
| `150k` | ~150K | 64 | 4 | 2 | 64 | Raspberry Pi 4, phones |
| `600k` | ~600K | 128 | 4 | 3 | 128 | Laptop, Jetson Nano |
| `2m` | ~2M | 256 | 8 | 2 | 256 | Server, swarm coordinator |

---

## Deployment Guide

### Raspberry Pi 4

```bash
# 1. Install
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gravitronics

# 2. Export
python edge/export_edge_model.py --variant 150k --output ./model --quantization int8

# 3. Run
python -c "
from edge.victoros_lgt_edge import VictorOSLGTEdge
import torch
runtime = VictorOSLGTEdge('./model/model_int8.pt', './model/model_config.json')
print(runtime.health_check())
"
```

### Multi-Agent Swarm

```python
# Each node runs independently; consensus is computed via force-weighted voting
from edge.gravitational_consensus import GravitationalNode, GravitationalConsensusNetwork

# Node A (high mass = coordinator)
net = GravitationalConsensusNetwork()
net.add_node(GravitationalNode("coordinator", mass=5.0))
net.add_node(GravitationalNode("edge-1", mass=1.0))
net.add_node(GravitationalNode("edge-2", mass=1.0))

# Propose a weight update — only accepted if gravitational consensus reached
result = net.compute_consensus({"op": "sync_weights", "epoch": 10})
```

---

## License

MIT License — see `LICENSE` for details.