"""
Tests for the Gravitronics training pipeline.

Covers: CodeDataset, Trainer (train_step), checkpoints, WizardSetup, and CLI.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.training.dataset import CodeDataset, _collect_files
from gravitronics.training.checkpoints import (
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from gravitronics.training.trainer import Trainer, TrainerConfig
from gravitronics.wizard.setup import RunConfig, WizardSetup, run_wizard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_source_dir(tmp_path: Path) -> Path:
    """A temporary directory with a handful of Python source files."""
    # Write enough bytes to form at least a few training examples.
    for i in range(5):
        content = f"# Python file {i}\n" * 30  # ~420 bytes each
        (tmp_path / f"file_{i}.py").write_text(content)
    return tmp_path


@pytest.fixture()
def small_dataset(tmp_source_dir: Path) -> CodeDataset:
    """A CodeDataset built from the tiny tmp source dir, seq_len=64."""
    return CodeDataset(sources=[tmp_source_dir], seq_len=64)


# ---------------------------------------------------------------------------
# CodeDataset
# ---------------------------------------------------------------------------


class TestCodeDataset:
    def test_creates_dataset_from_directory(self, tmp_source_dir: Path) -> None:
        ds = CodeDataset(sources=[tmp_source_dir], seq_len=64)
        assert len(ds) > 0

    def test_items_have_correct_shapes(self, small_dataset: CodeDataset) -> None:
        inp, tgt = small_dataset[0]
        assert inp.shape == (64,)
        assert tgt.shape == (64,)

    def test_target_is_shifted_input(self, small_dataset: CodeDataset) -> None:
        inp, tgt = small_dataset[0]
        # In a sliding window dataset, target == input shifted by one.
        # Both come from the same contiguous chunk.
        assert torch.equal(inp[1:], tgt[:-1])

    def test_token_dtype_is_long(self, small_dataset: CodeDataset) -> None:
        inp, tgt = small_dataset[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_tokens_in_valid_byte_range(self, small_dataset: CodeDataset) -> None:
        inp, _ = small_dataset[0]
        assert int(inp.min()) >= 0
        assert int(inp.max()) <= 255

    def test_stride_reduces_overlap(self, tmp_source_dir: Path) -> None:
        ds_full = CodeDataset(sources=[tmp_source_dir], seq_len=64, stride=64)
        ds_half = CodeDataset(sources=[tmp_source_dir], seq_len=64, stride=32)
        assert len(ds_half) >= len(ds_full)

    def test_single_file_source(self, tmp_source_dir: Path) -> None:
        py_file = next(tmp_source_dir.glob("*.py"))
        ds = CodeDataset(sources=[py_file], seq_len=64)
        assert len(ds) > 0

    def test_num_tokens_property(self, small_dataset: CodeDataset) -> None:
        assert small_dataset.num_tokens > 0

    def test_source_files_property(self, small_dataset: CodeDataset) -> None:
        assert len(small_dataset.source_files) == 5

    def test_empty_sources_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No matching files"):
            CodeDataset(sources=[empty_dir], seq_len=64)

    def test_too_small_corpus_raises(self, tmp_path: Path) -> None:
        tiny = tmp_path / "tiny.py"
        tiny.write_bytes(b"hi")  # 2 bytes — way too small for seq_len=64
        with pytest.raises(ValueError, match="[Cc]orpus"):
            CodeDataset(sources=[tiny], seq_len=64)

    def test_nonexistent_source_warns_but_raises(self) -> None:
        with pytest.raises(ValueError):
            CodeDataset(sources=["/nonexistent/path/xyz"], seq_len=64)

    def test_collect_files_skips_wrong_extension(self, tmp_path: Path) -> None:
        (tmp_path / "keep.py").write_text("x = 1")
        (tmp_path / "skip.exe").write_bytes(b"\x00\x01")
        files = _collect_files([tmp_path])
        names = [f.name for f in files]
        assert "keep.py" in names
        assert "skip.exe" not in names


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        from gravitronics.lgt.model import create_lgt

        model = create_lgt("150k")
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(model, ckpt_path, epoch=2, step=100, loss=1.23)

        model2 = create_lgt("150k")
        payload = load_checkpoint(ckpt_path, model2)

        assert payload["epoch"] == 2
        assert payload["step"] == 100
        assert abs(payload["loss"] - 1.23) < 1e-6

        # Weights should match after loading
        for (name, p1), (_, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter mismatch: {name}"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        from gravitronics.lgt.model import create_lgt

        model = create_lgt("150k")
        deep = tmp_path / "a" / "b" / "c" / "ckpt.pt"
        save_checkpoint(model, deep)
        assert deep.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        from gravitronics.lgt.model import create_lgt

        model = create_lgt("150k")
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "missing.pt", model)

    def test_optimizer_state_saved_and_restored(self, tmp_path: Path) -> None:
        from gravitronics.lgt.model import create_lgt

        model = create_lgt("150k")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Take one gradient step so optimizer state is non-trivial
        dummy_input = torch.randint(0, 256, (1, 64))
        logits, _ = model(dummy_input)
        loss = logits.sum()
        loss.backward()
        opt.step()

        ckpt_path = tmp_path / "with_opt.pt"
        save_checkpoint(model, ckpt_path, optimizer=opt, epoch=0, step=1)

        model2 = create_lgt("150k")
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(ckpt_path, model2, optimizer=opt2)

        # State groups should have been restored
        assert len(opt2.state) > 0

    def test_list_checkpoints_finds_pt_files(self, tmp_path: Path) -> None:
        from gravitronics.lgt.model import create_lgt

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
# Trainer
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_trainer_creates_model_and_runs_one_step(
        self, small_dataset: CodeDataset
    ) -> None:
        cfg = TrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,  # disable mid-run checkpoints
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = Trainer(cfg, dataset=small_dataset)
        inp = torch.randint(0, 256, (4, 64))
        tgt = torch.randint(0, 256, (4, 64))
        loss = trainer.train_step(inp, tgt)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_returns_loss_history(
        self, small_dataset: CodeDataset
    ) -> None:
        cfg = TrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = Trainer(cfg, dataset=small_dataset)
        history = trainer.train()
        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(v, float) for v in history)

    def test_trainer_loss_decreases_slightly_with_overfit(
        self, tmp_source_dir: Path
    ) -> None:
        """On a tiny repeated corpus, loss should trend down over many epochs."""
        # Use stride=1 to maximise examples, run 5 epochs.
        ds = CodeDataset(sources=[tmp_source_dir], seq_len=64, stride=16)
        cfg = TrainerConfig(
            model_variant="150k",
            num_epochs=5,
            batch_size=8,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
            seed=42,
        )
        trainer = Trainer(cfg, dataset=ds)
        history = trainer.train()
        # First loss should be higher than last loss (model is learning)
        assert history[0] > history[-1]

    def test_trainer_saves_checkpoint(
        self, small_dataset: CodeDataset, tmp_path: Path
    ) -> None:
        cfg = TrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,  # only epoch-end checkpoints
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = Trainer(cfg, dataset=small_dataset)
        trainer.train()
        ckpts = list_checkpoints(tmp_path / "ckpts")
        assert len(ckpts) >= 1

    def test_trainer_resume_from_checkpoint(
        self, small_dataset: CodeDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        cfg = TrainerConfig(
            model_variant="150k",
            num_epochs=1,
            batch_size=4,
            checkpoint_every_n_steps=0,
            checkpoint_dir=str(ckpt_dir),
            log_every_n_steps=0,
            device="cpu",
            seed=0,
        )
        trainer = Trainer(cfg, dataset=small_dataset)
        trainer.train()

        ckpts = list_checkpoints(ckpt_dir)
        assert ckpts, "No checkpoint was written — cannot test resume"

        # Create a fresh trainer and resume
        trainer2 = Trainer(cfg, dataset=small_dataset)
        trainer2.resume_from(ckpts[0])
        assert trainer2.global_step >= 0  # resumed without error

    def test_trainer_no_sources_raises(self) -> None:
        cfg = TrainerConfig(sources=[], model_variant="150k", device="cpu")
        with pytest.raises(ValueError, match="sources"):
            Trainer(cfg)  # no dataset and no sources

    def test_global_step_increments(self, small_dataset: CodeDataset) -> None:
        cfg = TrainerConfig(
            model_variant="150k",
            batch_size=4,
            checkpoint_every_n_steps=0,
            log_every_n_steps=0,
            device="cpu",
        )
        trainer = Trainer(cfg, dataset=small_dataset)
        for _ in range(3):
            trainer.train_step(
                torch.randint(0, 256, (4, 64)),
                torch.randint(0, 256, (4, 64)),
            )
        assert trainer.global_step == 3


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
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.device == "cpu"

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model variant"):
            run_wizard(interactive=False, model_variant="999m")

    def test_runconfig_is_dataclass(self) -> None:
        cfg = RunConfig()
        assert hasattr(cfg, "sources")
        assert hasattr(cfg, "model_variant")

    def test_wizard_setup_non_interactive_all_variants(self) -> None:
        for variant in ("150k", "600k", "2m"):
            cfg = run_wizard(interactive=False, model_variant=variant)
            assert cfg.model_variant == variant


# ---------------------------------------------------------------------------
# CLI argument parsing (no subprocess — just parse args directly)
# ---------------------------------------------------------------------------


class TestCLI:
    def _parse(self, argv):
        from gravitronics.cli import _build_parser

        return _build_parser().parse_args(argv)

    def test_train_subcommand_parsed(self) -> None:
        args = self._parse(
            ["train", "--sources", "/some/dir", "--variant", "150k", "--epochs", "2"]
        )
        assert args.command == "train"
        assert args.sources == ["/some/dir"]
        assert args.variant == "150k"
        assert args.epochs == 2

    def test_train_defaults(self) -> None:
        args = self._parse(["train", "--sources", "/dir"])
        assert args.batch_size == 16
        assert abs(args.lr - 3e-4) < 1e-10
        assert args.checkpoint_dir == "checkpoints"
        assert args.device == "auto"
        assert args.seed == 42

    def test_wizard_subcommand_parsed(self) -> None:
        args = self._parse(["wizard"])
        assert args.command == "wizard"

    def test_verbose_flag(self) -> None:
        args = self._parse(["-v", "train", "--sources", "/dir"])
        assert args.verbose is True

    def test_train_multiple_sources(self) -> None:
        args = self._parse(
            ["train", "--sources", "/dir1", "/dir2", "file.py"]
        )
        assert len(args.sources) == 3

    def test_train_resume_flag(self) -> None:
        args = self._parse(
            ["train", "--sources", "/dir", "--resume", "ckpts/model.pt"]
        )
        assert args.resume == "ckpts/model.pt"
