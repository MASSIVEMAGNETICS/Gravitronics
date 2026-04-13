"""
Tests for the Gravitronics training subsystem.

Covers:
- TrainingConfig: validation, serialisation, device resolution
- CheckpointManager: save/load/resume/prune/best
- Trainer: smoke test (dry-run + minimal real loop)
- Export: export_trained_model + load_exported_model round-trip
"""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.lgt.model import create_lgt
from gravitronics.training.checkpoint import CheckpointManager
from gravitronics.training.config import TrainingConfig
from gravitronics.training.trainer import Trainer


# --------------------------------------------------------------------------- #
# TrainingConfig — validation                                                  #
# --------------------------------------------------------------------------- #


class TestTrainingConfigValidation:
    def test_default_config_valid(self):
        cfg = TrainingConfig()
        cfg.validate()  # should not raise

    def test_invalid_model_variant_raises(self):
        with pytest.raises(ValueError, match="model_variant"):
            TrainingConfig(model_variant="999m")

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="preset"):
            TrainingConfig(preset="expert")

    def test_invalid_val_split_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            TrainingConfig(val_split=0.0)

    def test_invalid_val_split_too_high_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            TrainingConfig(val_split=1.0)

    def test_invalid_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            TrainingConfig(device="tpu")

    def test_invalid_export_format_raises(self):
        with pytest.raises(ValueError, match="export_format"):
            TrainingConfig(export_format="safetensors")

    def test_invalid_export_quantization_raises(self):
        with pytest.raises(ValueError, match="export_quantization"):
            TrainingConfig(export_quantization="bf16")

    def test_invalid_self_train_policy_raises(self):
        with pytest.raises(ValueError, match="self_train_policy"):
            TrainingConfig(self_train_policy="random")

    def test_invalid_keep_last_n_raises(self):
        with pytest.raises(ValueError, match="keep_last_n"):
            TrainingConfig(keep_last_n_checkpoints=0)

    def test_max_steps_zero_is_valid(self):
        cfg = TrainingConfig(max_steps=0)
        cfg.validate()

    def test_early_stop_zero_is_valid(self):
        cfg = TrainingConfig(early_stop_patience=0)
        cfg.validate()

    def test_wall_clock_zero_is_valid(self):
        cfg = TrainingConfig(max_wall_clock_sec=0.0)
        cfg.validate()


# --------------------------------------------------------------------------- #
# TrainingConfig — serialisation                                               #
# --------------------------------------------------------------------------- #


class TestTrainingConfigSerialisation:
    def test_to_dict_round_trip(self):
        cfg = TrainingConfig(model_variant="600k", epochs=5)
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg == cfg2

    def test_from_dict_ignores_unknown_keys(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        d["__unknown_key__"] = "surprise"
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg == cfg2

    def test_save_and_load(self, tmp_path):
        cfg = TrainingConfig(model_variant="2m", epochs=3)
        path = str(tmp_path / "cfg.json")
        cfg.save(path)
        assert os.path.isfile(path)
        cfg2 = TrainingConfig.load(path)
        assert cfg == cfg2

    def test_saved_json_is_valid(self, tmp_path):
        cfg = TrainingConfig()
        path = str(tmp_path / "cfg.json")
        cfg.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["model_variant"] == "150k"


# --------------------------------------------------------------------------- #
# TrainingConfig — device resolution                                           #
# --------------------------------------------------------------------------- #


class TestTrainingConfigDevice:
    def test_explicit_cpu(self):
        cfg = TrainingConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_auto_returns_string(self):
        cfg = TrainingConfig(device="auto")
        result = cfg.resolve_device()
        assert isinstance(result, str)
        assert result in ("cpu", "cuda", "mps")


# --------------------------------------------------------------------------- #
# CheckpointManager — save / load / resume                                     #
# --------------------------------------------------------------------------- #


class TestCheckpointManagerSaveLoad:
    def _make_model_and_opt(self):
        model = create_lgt("150k")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    def test_save_creates_file(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3)
        model, opt = self._make_model_and_opt()
        path = ckpt.save(model, opt, step=100, epoch=0, loss=1.23)
        assert os.path.isfile(path)

    def test_save_payload_fields(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3)
        model, opt = self._make_model_and_opt()
        path = ckpt.save(model, opt, step=50, epoch=2, loss=0.5)
        payload = CheckpointManager.load(path)
        assert payload["step"] == 50
        assert payload["epoch"] == 2
        assert abs(payload["loss"] - 0.5) < 1e-6

    def test_verify_nonexistent_returns_false(self):
        assert CheckpointManager.verify("/tmp/__nonexistent__.pt") is False

    def test_verify_existing_returns_true(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3)
        model, opt = self._make_model_and_opt()
        path = ckpt.save(model, opt, step=1, epoch=0, loss=0.9)
        assert CheckpointManager.verify(path) is True

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load("/tmp/__no_such_file__.pt")

    def test_resume_restores_weights(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3)
        model, opt = self._make_model_and_opt()

        # Perturb weights, save, then restore into fresh model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        path = ckpt.save(model, opt, step=10, epoch=1, loss=0.7)

        model2, opt2 = self._make_model_and_opt()
        step, epoch, loss = ckpt.resume(model2, opt2, path, device="cpu")
        assert step == 10
        assert epoch == 1

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# --------------------------------------------------------------------------- #
# CheckpointManager — best-model checkpoint                                    #
# --------------------------------------------------------------------------- #


class TestCheckpointManagerBest:
    def _make_model_and_opt(self):
        model = create_lgt("150k")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    def test_best_saved_on_improvement(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3, save_best=True)
        model, opt = self._make_model_and_opt()
        path = ckpt.save_best(model, opt, step=1, epoch=0, val_loss=1.0)
        assert path is not None
        assert os.path.isfile(path)

    def test_best_not_saved_when_no_improvement(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3, save_best=True)
        model, opt = self._make_model_and_opt()
        ckpt.save_best(model, opt, step=1, epoch=0, val_loss=1.0)
        path = ckpt.save_best(model, opt, step=2, epoch=1, val_loss=1.5)
        assert path is None

    def test_best_checkpoint_path(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=3, save_best=True)
        assert ckpt.best_checkpoint() is None
        model, opt = self._make_model_and_opt()
        ckpt.save_best(model, opt, step=1, epoch=0, val_loss=0.5)
        assert ckpt.best_checkpoint() is not None


# --------------------------------------------------------------------------- #
# CheckpointManager — pruning                                                  #
# --------------------------------------------------------------------------- #


class TestCheckpointManagerPruning:
    def _make_model_and_opt(self):
        model = create_lgt("150k")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    def test_prunes_oldest(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=2)
        model, opt = self._make_model_and_opt()
        for i in range(1, 5):
            ckpt.save(model, opt, step=i * 100, epoch=i, loss=1.0)
        paths = ckpt.list_checkpoints()
        assert len(paths) == 2
        # Latest two should remain
        assert "step_00000400" in paths[-1]
        assert "step_00000300" in paths[-2]

    def test_list_is_sorted(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=10)
        model, opt = self._make_model_and_opt()
        for i in [3, 1, 2]:
            ckpt.save(model, opt, step=i * 100, epoch=i, loss=1.0)
        paths = ckpt.list_checkpoints()
        steps = [int(p.split("step_")[1].split(".")[0]) for p in paths]
        assert steps == sorted(steps)

    def test_latest_checkpoint(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path), keep_last_n=5)
        assert ckpt.latest_checkpoint() is None
        model, opt = self._make_model_and_opt()
        ckpt.save(model, opt, step=100, epoch=0, loss=1.0)
        ckpt.save(model, opt, step=200, epoch=1, loss=0.9)
        latest = ckpt.latest_checkpoint()
        assert "step_00000200" in latest


# --------------------------------------------------------------------------- #
# Trainer — dry run                                                            #
# --------------------------------------------------------------------------- #


class TestTrainerDryRun:
    def test_dry_run_returns_status(self, tmp_path):
        cfg = TrainingConfig(
            dry_run=True,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
        )
        trainer = Trainer(cfg)
        result = trainer.train()
        assert result["status"] == "dry_run"
        assert result["steps"] == 0

    def test_dry_run_all_variants(self, tmp_path):
        for variant in ("150k", "600k", "2m"):
            cfg = TrainingConfig(
                model_variant=variant,
                dry_run=True,
                checkpoint_dir=str(tmp_path / f"ckpts_{variant}"),
                log_dir=str(tmp_path / f"logs_{variant}"),
                export_dir=str(tmp_path / f"exports_{variant}"),
            )
            result = Trainer(cfg).train()
            assert result["status"] == "dry_run"


# --------------------------------------------------------------------------- #
# Trainer — minimal training loop (smoke test)                                 #
# --------------------------------------------------------------------------- #


class TestTrainerSmoke:
    def test_train_one_epoch(self, tmp_path):
        cfg = TrainingConfig(
            epochs=1,
            max_steps=3,
            batch_size=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
            checkpoint_every_steps=2,
            checkpoint_every_epochs=1,
        )
        trainer = Trainer(cfg)
        result = trainer.train()
        assert result["status"] == "ok"
        assert result["steps"] >= 1

    def test_progress_callback_called(self, tmp_path):
        calls = []

        def cb(info):
            calls.append(info)

        cfg = TrainingConfig(
            epochs=1,
            max_steps=2,
            batch_size=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
        )
        trainer = Trainer(cfg, progress_callback=cb)
        trainer.train()
        assert len(calls) >= 1
        assert "step" in calls[0]
        assert "loss" in calls[0]
        assert "eta_sec" in calls[0]

    def test_stop_event_halts_training(self, tmp_path):
        stop = threading.Event()
        cfg = TrainingConfig(
            epochs=100,
            batch_size=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
        )
        trainer = Trainer(cfg, stop_event=stop)
        # Set stop event in a thread shortly after training begins
        t = threading.Timer(0.2, stop.set)
        t.start()
        result = trainer.train()
        # Should have stopped well before 100 epochs
        assert result["steps"] < 10000

    def test_checkpoint_created(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        cfg = TrainingConfig(
            epochs=1,
            max_steps=5,
            batch_size=2,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
            checkpoint_every_steps=2,
            checkpoint_every_epochs=1,
        )
        Trainer(cfg).train()
        # At least one .pt file should exist
        pts = list(ckpt_dir.glob("*.pt"))
        assert len(pts) >= 1


# --------------------------------------------------------------------------- #
# Trainer — resume                                                             #
# --------------------------------------------------------------------------- #


class TestTrainerResume:
    def test_resume_from_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        cfg = TrainingConfig(
            epochs=1,
            max_steps=3,
            batch_size=2,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(tmp_path / "logs"),
            export_dir=str(tmp_path / "exports"),
            checkpoint_every_steps=2,
            checkpoint_every_epochs=0,
        )
        # First run
        trainer1 = Trainer(cfg)
        trainer1.train()
        ckpt_mgr = CheckpointManager(str(ckpt_dir), keep_last_n=5)
        latest = ckpt_mgr.latest_checkpoint()
        assert latest is not None

        # Resume
        trainer2 = Trainer(cfg)
        result = trainer2.train(resume_from=latest)
        assert result["status"] == "ok"


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #


class TestExportTrainedModel:
    def test_export_pt(self, tmp_path):
        from gravitronics.training.export import export_trained_model

        model = create_lgt("150k")
        cfg = TrainingConfig(
            export_dir=str(tmp_path / "exports"),
            export_format="pt",
            export_quantization="none",
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
        )
        result = export_trained_model(model, cfg)
        assert result["status"] == "ok"
        assert any(f.endswith(".pt") for f in result["files"])

    def test_export_creates_metadata(self, tmp_path):
        from gravitronics.training.export import export_trained_model

        model = create_lgt("150k")
        cfg = TrainingConfig(
            export_dir=str(tmp_path / "exports"),
            export_format="pt",
            export_quantization="none",
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
        )
        result = export_trained_model(model, cfg, step=42, epoch=3, val_loss=0.5)
        meta_path = result["metadata_path"]
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 42
        assert meta["epoch"] == 3

    def test_load_exported_model(self, tmp_path):
        from gravitronics.training.export import export_trained_model, load_exported_model

        model = create_lgt("150k")
        cfg = TrainingConfig(
            export_dir=str(tmp_path / "exports"),
            export_format="pt",
            export_quantization="none",
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
        )
        export_trained_model(model, cfg)
        loaded = load_exported_model(str(tmp_path / "exports"), device="cpu")
        assert loaded is not None
        # Run a forward pass
        model_cfg = loaded.get_config()
        ids = torch.randint(0, model_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _ = loaded(ids)
        assert logits.shape[-1] == model_cfg.vocab_size

    def test_load_missing_export_raises(self, tmp_path):
        from gravitronics.training.export import load_exported_model

        with pytest.raises(FileNotFoundError):
            load_exported_model(str(tmp_path / "nonexistent"))


# ===========================================================================
# File-based training additions
# ===========================================================================
# The tests below cover CodeDataset, the checkpoints helpers (checkpoints.py),
# FileTrainer, WizardSetup, and the CLI train-files sub-command.
# ===========================================================================

from pathlib import Path

from gravitronics.training.dataset import CodeDataset, _collect_files
from gravitronics.training.checkpoints import (
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from gravitronics.training.file_trainer import FileTrainer, FileTrainerConfig
from gravitronics.wizard.setup import RunConfig, WizardSetup, run_wizard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source_dir(tmp_path: Path, n: int = 5) -> Path:
    """Write *n* Python files with enough bytes for seq_len=64 examples."""
    for i in range(n):
        content = f"# Python file {i}\n" * 30  # ~420 bytes each
        (tmp_path / f"file_{i}.py").write_text(content)
    return tmp_path


def _small_dataset(tmp_path: Path) -> CodeDataset:
    src = _make_source_dir(tmp_path)
    return CodeDataset(sources=[src], seq_len=64)


# ---------------------------------------------------------------------------
# CodeDataset
# ---------------------------------------------------------------------------


class TestCodeDataset:
    def test_creates_dataset_from_directory(self, tmp_path: Path) -> None:
        src = _make_source_dir(tmp_path)
        ds = CodeDataset(sources=[src], seq_len=64)
        assert len(ds) > 0

    def test_items_have_correct_shapes(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        inp, tgt = ds[0]
        assert inp.shape == (64,)
        assert tgt.shape == (64,)

    def test_target_is_shifted_input(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        inp, tgt = ds[0]
        assert torch.equal(inp[1:], tgt[:-1])

    def test_token_dtype_is_long(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        inp, tgt = ds[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_tokens_in_valid_byte_range(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        inp, _ = ds[0]
        assert int(inp.min()) >= 0
        assert int(inp.max()) <= 255

    def test_stride_produces_more_examples(self, tmp_path: Path) -> None:
        src = _make_source_dir(tmp_path)
        ds_full = CodeDataset(sources=[src], seq_len=64, stride=64)
        ds_half = CodeDataset(sources=[src], seq_len=64, stride=32)
        assert len(ds_half) >= len(ds_full)

    def test_single_file_source(self, tmp_path: Path) -> None:
        src = _make_source_dir(tmp_path)
        py_file = next(src.glob("*.py"))
        ds = CodeDataset(sources=[py_file], seq_len=64)
        assert len(ds) > 0

    def test_num_tokens_property(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        assert ds.num_tokens > 0

    def test_source_files_property(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        assert len(ds.source_files) == 5

    def test_empty_sources_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No matching files"):
            CodeDataset(sources=[empty], seq_len=64)

    def test_too_small_corpus_raises(self, tmp_path: Path) -> None:
        tiny = tmp_path / "tiny.py"
        tiny.write_bytes(b"hi")
        with pytest.raises(ValueError, match="[Cc]orpus"):
            CodeDataset(sources=[tiny], seq_len=64)

    def test_collect_files_skips_wrong_extension(self, tmp_path: Path) -> None:
        (tmp_path / "keep.py").write_text("x = 1")
        (tmp_path / "skip.exe").write_bytes(b"\x00\x01")
        files = _collect_files([tmp_path])
        names = [f.name for f in files]
        assert "keep.py" in names
        assert "skip.exe" not in names


# ---------------------------------------------------------------------------
# Checkpoints helpers (checkpoints.py — distinct from checkpoint.py)
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        model = create_lgt("150k")
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(model, ckpt_path, epoch=2, step=100, loss=1.23)

        model2 = create_lgt("150k")
        payload = load_checkpoint(ckpt_path, model2)

        assert payload["epoch"] == 2
        assert payload["step"] == 100
        assert abs(payload["loss"] - 1.23) < 1e-6

        for (name, p1), (_, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter mismatch: {name}"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        model = create_lgt("150k")
        deep = tmp_path / "a" / "b" / "c" / "ckpt.pt"
        save_checkpoint(model, deep)
        assert deep.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        model = create_lgt("150k")
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "missing.pt", model)

    def test_list_checkpoints_finds_pt_files(self, tmp_path: Path) -> None:
        model = create_lgt("150k")
        for name in ("a.pt", "b.pt", "c.pt"):
            save_checkpoint(model, tmp_path / name)
        ckpts = list_checkpoints(tmp_path)
        assert len(ckpts) == 3
        assert all(p.suffix == ".pt" for p in ckpts)

    def test_list_checkpoints_empty_dir(self, tmp_path: Path) -> None:
        assert list_checkpoints(tmp_path) == []

    def test_list_checkpoints_nonexistent_dir(self, tmp_path: Path) -> None:
        assert list_checkpoints(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# FileTrainer
# ---------------------------------------------------------------------------


class TestFileTrainer:
    def test_creates_model_and_runs_one_step(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        cfg = FileTrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = FileTrainer(cfg, dataset=ds)
        inp = torch.randint(0, 256, (4, 64))
        tgt = torch.randint(0, 256, (4, 64))
        loss = trainer.train_step(inp, tgt)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_returns_loss_history(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        cfg = FileTrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = FileTrainer(cfg, dataset=ds)
        history = trainer.train()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_no_sources_raises(self) -> None:
        cfg = FileTrainerConfig(sources=[], model_variant="150k", device="cpu")
        with pytest.raises(ValueError, match="sources"):
            FileTrainer(cfg)

    def test_global_step_increments(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        cfg = FileTrainerConfig(
            model_variant="150k",
            batch_size=4,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
        )
        trainer = FileTrainer(cfg, dataset=ds)
        for _ in range(3):
            trainer.train_step(
                torch.randint(0, 256, (4, 64)),
                torch.randint(0, 256, (4, 64)),
            )
        assert trainer.global_step == 3

    def test_trainer_saves_checkpoint(self, tmp_path: Path) -> None:
        ds = _small_dataset(tmp_path)
        ckpt_dir = tmp_path / "ckpts"
        cfg = FileTrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,
            checkpoint_dir=str(ckpt_dir),
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        FileTrainer(cfg, dataset=ds).train()
        ckpts = list_checkpoints(ckpt_dir)
        assert len(ckpts) >= 1


# ---------------------------------------------------------------------------
# WizardSetup (non-interactive)
# ---------------------------------------------------------------------------


class TestWizardSetup:
    def test_non_interactive_uses_defaults(self) -> None:
        cfg = run_wizard(interactive=False)
        assert isinstance(cfg, RunConfig)
        assert cfg.model_variant == "150k"
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 16

    def test_non_interactive_accepts_overrides(self) -> None:
        cfg = run_wizard(
            interactive=False,
            sources=["/some/path"],
            model_variant="600k",
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            checkpoint_dir="/tmp/ckpts",
            device="cpu",
        )
        assert cfg.sources == ["/some/path"]
        assert cfg.model_variant == "600k"
        assert cfg.num_epochs == 10
        assert cfg.batch_size == 32
        assert abs(cfg.learning_rate - 1e-3) < 1e-10

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model variant"):
            run_wizard(interactive=False, model_variant="999m")

    def test_runconfig_is_dataclass(self) -> None:
        cfg = RunConfig()
        assert hasattr(cfg, "sources")
        assert hasattr(cfg, "model_variant")

    def test_all_variants_accepted(self) -> None:
        for variant in ("150k", "600k", "2m"):
            cfg = run_wizard(interactive=False, model_variant=variant)
            assert cfg.model_variant == variant


# ---------------------------------------------------------------------------
# CLI — train-files sub-command parsing
# ---------------------------------------------------------------------------


class TestCLITrainFiles:
    def _parse(self, argv):
        from gravitronics.cli import _build_parser

        return _build_parser().parse_args(argv)

    def test_train_files_subcommand_parsed(self) -> None:
        args = self._parse(
            ["train-files", "--sources", "/some/dir", "--variant", "150k", "--epochs", "2"]
        )
        assert args.command == "train-files"
        assert args.sources == ["/some/dir"]
        assert args.variant == "150k"
        assert args.epochs == 2

    def test_train_files_defaults(self) -> None:
        args = self._parse(["train-files", "--sources", "/dir"])
        assert args.batch_size == 16
        assert abs(args.lr - 3e-4) < 1e-10
        assert args.checkpoint_dir == "checkpoints"
        assert args.device == "auto"
        assert args.seed == 42

    def test_train_files_multiple_sources(self) -> None:
        args = self._parse(["train-files", "--sources", "/dir1", "/dir2", "file.py"])
        assert len(args.sources) == 3

    def test_train_files_resume_flag(self) -> None:
        args = self._parse(
            ["train-files", "--sources", "/dir", "--resume", "ckpts/model.pt"]
        )
        assert args.resume == "ckpts/model.pt"

    def test_original_train_subcommand_still_works(self) -> None:
        args = self._parse(["train", "--config", "training_config.json"])
        assert args.command == "train"
        assert args.config == "training_config.json"
