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
