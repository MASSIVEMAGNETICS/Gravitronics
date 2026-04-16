"""
cli.py — Gravitronics command-line interface.

Sub-commands
------------
wizard          Launch the Windows graphical setup wizard.
train           Run a training job from a config file.
train-files     Train directly on source files/repos (no config file needed).
resume          Resume training from a checkpoint.
export          Export a trained model to disk.
validate-config Validate a config file without running training.

Usage examples
--------------
    python -m gravitronics.cli wizard
    python -m gravitronics.cli train --config training_config.json
    python -m gravitronics.cli train-files --sources ./my_repo ./extra.py
    python -m gravitronics.cli resume --config training_config.json \\
            --checkpoint checkpoints/checkpoint_step_00001000.pt
    python -m gravitronics.cli export --config training_config.json \\
            --checkpoint checkpoints/best_model.pt
    python -m gravitronics.cli validate-config --config training_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sub-command implementations                                                  #
# --------------------------------------------------------------------------- #


def _cmd_wizard(args: argparse.Namespace) -> int:
    """Launch the Windows graphical setup wizard."""
    try:
        from gravitronics.wizard.setup_wizard import run_wizard
    except ImportError as exc:
        print(
            f"ERROR: Could not import wizard ({exc}).\n"
            "Ensure tkinter is installed (comes with most Python distributions)."
        )
        return 1
    run_wizard()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run training from a config file."""
    from gravitronics.training.config import TrainingConfig
    from gravitronics.training.trainer import Trainer

    cfg = TrainingConfig.load(args.config)
    if args.dry_run:
        cfg.dry_run = True

    trainer = Trainer(cfg)
    result = trainer.train()
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in ("ok", "dry_run") else 1


def _cmd_resume(args: argparse.Namespace) -> int:
    """Resume training from a checkpoint."""
    from gravitronics.training.config import TrainingConfig
    from gravitronics.training.trainer import Trainer

    cfg = TrainingConfig.load(args.config)
    trainer = Trainer(cfg)
    result = trainer.train(resume_from=args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


def _cmd_export(args: argparse.Namespace) -> int:
    """Export a model from a checkpoint."""
    from gravitronics.training.config import TrainingConfig
    from gravitronics.training.export import export_trained_model
    from gravitronics.training.checkpoint import CheckpointManager
    from gravitronics.lgt.model import create_lgt
    from gravitronics.lgt.config import LGTConfig
    import torch

    cfg = TrainingConfig.load(args.config)
    model = create_lgt(cfg.model_variant)
    model_cfg = LGTConfig.from_variant(cfg.model_variant)

    if args.checkpoint:
        payload = CheckpointManager.load(args.checkpoint)
        model.load_state_dict(payload["model_state_dict"])
        step = payload.get("step", 0)
        epoch = payload.get("epoch", 0)
        val_loss = payload.get("val_loss", float("nan"))
    else:
        step, epoch, val_loss = 0, 0, float("nan")

    result = export_trained_model(
        model, cfg, model_cfg=model_cfg, step=step, epoch=epoch, val_loss=val_loss
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


def _cmd_validate_config(args: argparse.Namespace) -> int:
    """Validate a config file."""
    from gravitronics.training.config import TrainingConfig

    try:
        cfg = TrainingConfig.load(args.config)
        print(f"Config '{args.config}' is valid.")
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    except Exception as exc:
        print(f"Config validation FAILED: {exc}")
        return 1


def _cmd_train_files(args: argparse.Namespace) -> int:
    """Train directly on source files/repos without a config file."""
    from gravitronics.training.file_trainer import FileTrainer, FileTrainerConfig

    cfg = FileTrainerConfig(
        sources=args.sources,
        model_variant=args.variant,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_n_steps=args.checkpoint_steps,
        device=args.device,
        seed=args.seed,
        log_every_n_steps=args.log_steps,
    )
    trainer = FileTrainer(cfg)
    if args.resume:
        trainer.resume_from(args.resume)
    losses = trainer.train()
    if losses:
        print(f"\n[train-files] Done. Final loss: {losses[-1]:.4f}")
    else:
        print("\n[train-files] Done (no steps were executed).")
    return 0


# --------------------------------------------------------------------------- #
# Argument parser                                                              #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gravitronics",
        description="Gravitronics — Lightweight Gravitational Transformer CLI",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── wizard ── #
    sub.add_parser("wizard", help="Launch the Windows graphical setup wizard.")

    # ── train ── #
    train_p = sub.add_parser("train", help="Run training from a config file.")
    train_p.add_argument(
        "--config", required=True, metavar="PATH", help="Path to training_config.json."
    )
    train_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and build model without running the loop.",
    )

    # ── resume ── #
    resume_p = sub.add_parser("resume", help="Resume training from a checkpoint.")
    resume_p.add_argument(
        "--config", required=True, metavar="PATH", help="Path to training_config.json."
    )
    resume_p.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to a .pt checkpoint file.",
    )

    # ── export ── #
    export_p = sub.add_parser("export", help="Export a model.")
    export_p.add_argument(
        "--config", required=True, metavar="PATH", help="Path to training_config.json."
    )
    export_p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Checkpoint to export (optional; uses fresh weights if omitted).",
    )

    # ── validate-config ── #
    vc_p = sub.add_parser("validate-config", help="Validate a training config file.")
    vc_p.add_argument(
        "--config", required=True, metavar="PATH", help="Path to training_config.json."
    )

    # ── train-files ── #
    tf_p = sub.add_parser(
        "train-files",
        help="Train on source files/repos directly (no config file needed).",
    )
    tf_p.add_argument(
        "--sources",
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more directories / files to train on.",
    )
    tf_p.add_argument(
        "--variant",
        choices=("150k", "600k", "2m"),
        default="150k",
        help="Model size variant (default: 150k).",
    )
    tf_p.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Number of training epochs (default: 3).",
    )
    tf_p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Batch size (default: 16).",
    )
    tf_p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Peak learning rate for AdamW (default: 3e-4).",
    )
    tf_p.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        metavar="DIR",
        help="Directory to write checkpoints (default: checkpoints).",
    )
    tf_p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        metavar="N",
        help="Save checkpoint every N steps (default: 500).",
    )
    tf_p.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto / cpu / cuda / mps (default: auto).",
    )
    tf_p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed (default: 42).",
    )
    tf_p.add_argument(
        "--log-steps",
        type=int,
        default=50,
        metavar="N",
        help="Log progress every N steps (default: 50).",
    )
    tf_p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume from.",
    )

    return p


# --------------------------------------------------------------------------- #
# Entry-point                                                                  #
# --------------------------------------------------------------------------- #

_COMMANDS = {
    "wizard": _cmd_wizard,
    "train": _cmd_train,
    "train-files": _cmd_train_files,
    "resume": _cmd_resume,
    "export": _cmd_export,
    "validate-config": _cmd_validate_config,
}


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
