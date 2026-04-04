# Gravitronics 🌌

**A physics-aware, edge-deployable Lightweight Gravitational Transformer (LGT) framework.**

[![Tests](https://img.shields.io/badge/tests-120%20passed-brightgreen)](#testing)
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
5. [Windows Setup Wizard](#windows-setup-wizard)
6. [Training Subsystem](#training-subsystem)
   - [TrainingConfig](#trainingconfig)
   - [Trainer — Live Training Loop](#trainer--live-training-loop)
   - [Auto Self-Training](#auto-self-training)
   - [Checkpoints & Resume](#checkpoints--resume)
   - [Model Export](#model-export)
7. [CLI Reference](#cli-reference)
8. [Edge Deployment Kit](#edge-deployment-kit)
   - [Export Model](#export-model)
   - [VictorOS Runtime Wrapper](#victoros-runtime-wrapper)
   - [Benchmark Suite](#benchmark-suite)
   - [Gravitational Consensus](#gravitational-consensus)
   - [Config File](#config-file)
9. [Testing](#testing)
10. [Diagnostics Schema](#diagnostics-schema)
11. [Model Variants](#model-variants)
12. [Deployment Guide](#deployment-guide)
13. [License](#license)

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

## Windows Setup Wizard

A production-grade multi-step graphical wizard (`tkinter`, ships with Python on Windows) for configuring and launching training jobs.

### Running the wizard

```bash
# Via the CLI
python -m gravitronics.cli wizard

# Or directly
python -m gravitronics.wizard.setup_wizard
```

### Wizard steps

| Step | Description |
|------|-------------|
| 1. Welcome | Prerequisites check (Python version, PyTorch, CUDA, ONNX) |
| 2. Data | Choose data directory / validation split |
| 3. Model | Select variant (150k / 600k / 2m) and preset (basic / advanced) |
| 4. Training params | Epochs, batch size, learning rate, device, dry-run option |
| 5. Checkpoints & export | Checkpoint dir/frequency, export format, auto self-training |
| 6. Summary | JSON config preview, save, dry-run, and "Start Training" |

The wizard:
- Validates all inputs before advancing each step.
- Saves the configuration to `training_config.json` (path is configurable).
- Offers a **dry-run** to verify the config and build the model without running the loop.
- Opens a live progress window when you click **Start Training**.

### Windows installation

```powershell
# 1. Install Python 3.9+ (tkinter is bundled)
# 2. Install Gravitronics
pip install torch numpy pyyaml psutil
pip install -e .

# 3. (Optional) Install ONNX support
pip install onnx onnxruntime

# 4. Launch the wizard
python -m gravitronics.cli wizard
```

---

## Training Subsystem

### TrainingConfig

`gravitronics.training.config.TrainingConfig` is a dataclass that holds **all** training settings and is the single source of truth shared between the wizard, CLI, and Python API.

```python
from gravitronics.training.config import TrainingConfig

cfg = TrainingConfig(
    model_variant="150k",
    epochs=20,
    batch_size=32,
    learning_rate=3e-4,
    device="auto",           # auto-selects CUDA > MPS > CPU
    checkpoint_dir="checkpoints",
    export_dir="exports",
    export_format="pt",
    seed=42,
)

# Save / load
cfg.save("training_config.json")
cfg2 = TrainingConfig.load("training_config.json")
```

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `model_variant` | `"150k"` | LGT size — `"150k"`, `"600k"`, or `"2m"` |
| `epochs` | `10` | Full passes over training data |
| `max_steps` | `0` | Hard step cap (0 = use epochs only) |
| `batch_size` | `32` | Samples per gradient step |
| `learning_rate` | `3e-4` | AdamW initial LR |
| `device` | `"auto"` | `"cpu"` / `"cuda"` / `"mps"` / `"auto"` |
| `seed` | `42` | Deterministic seed (`-1` = non-deterministic) |
| `checkpoint_dir` | `"checkpoints"` | Checkpoint output dir |
| `checkpoint_every_steps` | `500` | Step-based checkpoint frequency |
| `checkpoint_every_epochs` | `1` | Epoch-based checkpoint frequency |
| `keep_last_n_checkpoints` | `3` | Retention limit for periodic checkpoints |
| `save_best_checkpoint` | `True` | Maintain `best_model.pt` |
| `export_dir` | `"exports"` | Export output dir |
| `export_format` | `"pt"` | `"pt"` / `"onnx"` / `"both"` |
| `export_quantization` | `"fp16"` | `"fp16"` / `"int8"` / `"none"` |
| `auto_self_train` | `False` | Enable periodic self-training |
| `self_train_policy` | `"time"` | Trigger policy (see below) |
| `max_wall_clock_sec` | `0.0` | Safety guard — wall-clock limit (0 = none) |
| `early_stop_patience` | `0` | Epochs without improvement before stopping (0 = none) |

### Trainer — Live Training Loop

```python
from gravitronics.training.config import TrainingConfig
from gravitronics.training.trainer import Trainer
import threading

cfg = TrainingConfig(model_variant="150k", epochs=10, max_steps=1000)
stop_event = threading.Event()

def on_progress(info):
    print(f"Step {info['step']} | Loss {info['loss']:.4f} | ETA {info['eta_sec']:.0f}s")

trainer = Trainer(cfg, progress_callback=on_progress, stop_event=stop_event)
result = trainer.train()
# {"status": "ok", "steps": 1000, "epochs": 10, "final_loss": 0.312}

# To stop from another thread:
stop_event.set()
```

Features:
- **Progress callback** — called every step with `step`, `epoch`, `loss`, `eta_sec`, `progress`.
- **Deterministic seeding** via `TrainingConfig.seed`.
- **Device selection** — auto-selects best available (CUDA > MPS > CPU).
- **Gradient clipping** via `TrainingConfig.grad_clip`.
- **Early stopping** — `early_stop_patience` epochs without validation improvement.
- **Wall-clock guard** — `max_wall_clock_sec` aborts runaway training.

### Auto Self-Training

When `auto_self_train=True` the trainer enters a second loop that periodically re-trains (fine-tunes) the model on the same (or grown) dataset.

**Trigger policies** (`self_train_policy`):

| Policy | Trigger condition |
|--------|-------------------|
| `"time"` | Every `self_train_interval_sec` seconds of wall-clock time |
| `"data_threshold"` | When the dataset has grown by ≥ `self_train_data_threshold` new samples |
| `"metric_threshold"` | When validation loss drifts > `self_train_metric_threshold` above best |
| `"disabled"` | Self-training is disabled |

Safety guards:
- `self_train_max_rounds` — maximum number of self-training rounds (0 = unlimited).
- `max_wall_clock_sec` — global wall-clock cap applies across all rounds.
- The global `stop_event` terminates self-training immediately.

```python
cfg = TrainingConfig(
    auto_self_train=True,
    self_train_policy="time",
    self_train_interval_sec=3600,   # retrain every hour
    self_train_max_rounds=5,        # at most 5 fine-tuning rounds
    max_wall_clock_sec=86400,       # hard stop after 24 h
)
trainer = Trainer(cfg)
trainer.train()
```

### Checkpoints & Resume

```python
from gravitronics.training.checkpoint import CheckpointManager
import torch

ckpt = CheckpointManager(
    checkpoint_dir="checkpoints",
    keep_last_n=3,       # retain 3 most-recent periodic checkpoints
    save_best=True,      # also keep best_model.pt
)

# Save a checkpoint manually
ckpt.save(model, optimizer, step=500, epoch=2, loss=0.85)

# Save best-model checkpoint (only saved when val_loss improves)
ckpt.save_best(model, optimizer, step=500, epoch=2, val_loss=0.72)

# List / locate
latest = ckpt.latest_checkpoint()   # → "checkpoints/checkpoint_step_00000500.pt"
best   = ckpt.best_checkpoint()     # → "checkpoints/best_model.pt"

# Verify integrity
CheckpointManager.verify(latest)    # → True / False

# Load payload dict (includes model_state_dict, optimizer_state_dict, step, …)
payload = CheckpointManager.load(latest)

# Resume training
trainer = Trainer(cfg)
result = trainer.train(resume_from=latest)
```

**CLI resume:**
```bash
python -m gravitronics.cli resume \
    --config training_config.json \
    --checkpoint checkpoints/checkpoint_step_00001000.pt
```

Checkpoint file format (PyTorch pickle):
```python
{
  "model_state_dict": {...},
  "optimizer_state_dict": {...},
  "step": 1000,
  "epoch": 5,
  "loss": 0.72,
  "timestamp": "2026-04-02T14:00:00",
}
```

### Model Export

```python
from gravitronics.training.export import export_trained_model, load_exported_model
from gravitronics.lgt.model import create_lgt
from gravitronics.training.config import TrainingConfig

model = create_lgt("150k")
cfg = TrainingConfig(export_dir="exports", export_format="pt", export_quantization="fp16")

result = export_trained_model(model, cfg, step=1000, epoch=5, val_loss=0.72)
# result["files"] → ["exports/model.pt", "exports/model_config.json"]

# Load back for inference
loaded = load_exported_model("exports", device="cpu")
loaded.eval()
with torch.no_grad():
    logits, _ = loaded(input_ids)
```

Export outputs:
| File | Description |
|------|-------------|
| `model.pt` | Model state dict (or TorchScript if `trace=True`) |
| `model.onnx` | ONNX graph (when `export_format` is `"onnx"` or `"both"`) |
| `model_config.json` | LGTConfig sidecar for re-loading |
| `export_metadata.json` | Export run metadata (step, epoch, val_loss, timestamp) |

**CLI export:**
```bash
python -m gravitronics.cli export \
    --config training_config.json \
    --checkpoint checkpoints/best_model.pt
```

---

## CLI Reference

```
gravitronics COMMAND [OPTIONS]

Commands:
  wizard           Launch the Windows graphical setup wizard
  train            Run training from a config file
  resume           Resume training from a checkpoint
  export           Export a model to disk
  validate-config  Validate a training config file
```

### Examples

```bash
# Launch the wizard
python -m gravitronics.cli wizard

# Validate a config (no training)
python -m gravitronics.cli validate-config --config training_config.json

# Train from config
python -m gravitronics.cli train --config training_config.json

# Dry run (build model, skip loop)
python -m gravitronics.cli train --config training_config.json --dry-run

# Resume
python -m gravitronics.cli resume \
    --config training_config.json \
    --checkpoint checkpoints/checkpoint_step_00001000.pt

# Export
python -m gravitronics.cli export \
    --config training_config.json \
    --checkpoint checkpoints/best_model.pt
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
pytest tests/test_training.py -v
```

**120 tests, 0 failures.**

| Test file | Coverage |
|-----------|----------|
| `test_lgt_model.py` | Config, forward pass, diagnostics, save/load |
| `test_edge.py` | Export, tracing, benchmarks, VictorOS wrapper |
| `test_consensus.py` | Node creation, force law, voting, topology |
| `test_training.py` | TrainingConfig validation/serialisation, CheckpointManager save/load/prune/best, Trainer dry-run + smoke test, export round-trip |

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