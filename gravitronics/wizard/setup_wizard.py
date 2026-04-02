"""
setup_wizard.py — Production-grade Windows graphical setup wizard for Gravitronics.

Multi-step wizard implemented with tkinter (ships with Python on Windows).
Steps
-----
1. Welcome / prerequisites check
2. Data source selection
3. Model variant + preset selection
4. Training hyper-parameters (epochs, batch-size, learning-rate, device)
5. Checkpointing + export options
6. Summary + dry-run validation + "Start Training" button

Usage
-----
    python -m gravitronics.wizard.setup_wizard
    # or via CLI:
    python -m gravitronics.cli wizard

The wizard writes ``training_config.json`` to the current working directory
(path is shown in the summary step and configurable by the user).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Colour / style palette                                                       #
# --------------------------------------------------------------------------- #

_BG = "#1e1e2e"          # dark background
_FG = "#cdd6f4"          # light text
_ACCENT = "#89b4fa"      # blue accent
_BTN_BG = "#313244"      # button background
_BTN_ACTIVE = "#45475a"  # button hover
_ENTRY_BG = "#313244"
_WARN = "#f38ba8"        # red for errors
_OK = "#a6e3a1"          # green for success
_FONT_BODY = ("Segoe UI", 10)
_FONT_HEADER = ("Segoe UI", 14, "bold")
_FONT_MONO = ("Consolas", 9)


# --------------------------------------------------------------------------- #
# Validation helpers                                                           #
# --------------------------------------------------------------------------- #


def _validate_positive_int(value: str, field: str) -> int:
    try:
        v = int(value)
        if v < 1:
            raise ValueError
        return v
    except ValueError:
        raise ValueError(f"'{field}' must be a positive integer; got '{value}'.")


def _validate_positive_float(value: str, field: str) -> float:
    try:
        v = float(value)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        raise ValueError(f"'{field}' must be a positive number; got '{value}'.")


def _validate_fraction(value: str, field: str) -> float:
    try:
        v = float(value)
        if not (0.0 < v < 1.0):
            raise ValueError
        return v
    except ValueError:
        raise ValueError(f"'{field}' must be between 0 and 1 (exclusive); got '{value}'.")


# --------------------------------------------------------------------------- #
# Wizard Application                                                           #
# --------------------------------------------------------------------------- #


class GravitronicsSetupWizard:
    """Multi-step tkinter wizard that produces a :class:`TrainingConfig`.

    Parameters
    ----------
    root:
        Tk root window to build the wizard inside.
    on_complete:
        Called with the final config dict when the user clicks "Start Training".
    """

    # total number of wizard steps
    _N_STEPS = 6

    def __init__(
        self,
        root: tk.Tk,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.root = root
        self.on_complete = on_complete
        self._step = 0
        self._config: Dict[str, Any] = {}

        # ── tkinter variables ─────────────────────────────────────────── #
        self._data_dir = tk.StringVar(value="")
        self._model_variant = tk.StringVar(value="150k")
        self._preset = tk.StringVar(value="basic")
        self._epochs = tk.StringVar(value="10")
        self._batch_size = tk.StringVar(value="32")
        self._learning_rate = tk.StringVar(value="3e-4")
        self._device = tk.StringVar(value="auto")
        self._checkpoint_dir = tk.StringVar(value="checkpoints")
        self._checkpoint_every_epochs = tk.StringVar(value="1")
        self._keep_last_n = tk.StringVar(value="3")
        self._save_best = tk.BooleanVar(value=True)
        self._export_dir = tk.StringVar(value="exports")
        self._export_format = tk.StringVar(value="pt")
        self._export_quantization = tk.StringVar(value="fp16")
        self._auto_self_train = tk.BooleanVar(value=False)
        self._self_train_policy = tk.StringVar(value="time")
        self._self_train_interval = tk.StringVar(value="3600")
        self._config_path = tk.StringVar(value="training_config.json")
        self._seed = tk.StringVar(value="42")
        self._dry_run = tk.BooleanVar(value=False)
        self._val_split = tk.StringVar(value="0.1")

        self._setup_window()
        self._build_step()

    # ------------------------------------------------------------------ #
    # Window setup                                                         #
    # ------------------------------------------------------------------ #

    def _setup_window(self) -> None:
        self.root.title("Gravitronics Setup Wizard")
        self.root.geometry("780x620")
        self.root.minsize(700, 560)
        self.root.configure(bg=_BG)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # Apply ttk style
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=_BG)
        style.configure("TLabel", background=_BG, foreground=_FG, font=_FONT_BODY)
        style.configure("Header.TLabel", background=_BG, foreground=_ACCENT, font=_FONT_HEADER)
        style.configure(
            "TButton",
            background=_BTN_BG,
            foreground=_FG,
            font=_FONT_BODY,
            relief="flat",
            padding=(10, 5),
        )
        style.map("TButton", background=[("active", _BTN_ACTIVE)])
        style.configure("TEntry", fieldbackground=_ENTRY_BG, foreground=_FG, font=_FONT_BODY)
        style.configure(
            "TCombobox",
            fieldbackground=_ENTRY_BG,
            foreground=_FG,
            background=_BTN_BG,
            font=_FONT_BODY,
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=_BTN_BG,
            background=_ACCENT,
        )
        style.configure("TCheckbutton", background=_BG, foreground=_FG, font=_FONT_BODY)
        style.configure("TRadiobutton", background=_BG, foreground=_FG, font=_FONT_BODY)
        style.configure(
            "TLabelframe",
            background=_BG,
            foreground=_FG,
            font=_FONT_BODY,
        )
        style.configure("TLabelframe.Label", background=_BG, foreground=_ACCENT, font=_FONT_BODY)

        # Main container
        self._container = ttk.Frame(self.root, padding=20)
        self._container.pack(fill="both", expand=True)

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            self._container,
            variable=self._progress_var,
            maximum=self._N_STEPS,
            style="Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill="x", pady=(0, 8))

        # Step label
        self._step_label = ttk.Label(
            self._container, text="", style="TLabel", anchor="e"
        )
        self._step_label.pack(fill="x")

        # Content frame (replaced each step)
        self._content = ttk.Frame(self._container)
        self._content.pack(fill="both", expand=True, pady=10)

        # Navigation buttons
        nav = ttk.Frame(self._container)
        nav.pack(fill="x", pady=(8, 0))
        self._btn_back = ttk.Button(nav, text="◀  Back", command=self._go_back)
        self._btn_back.pack(side="left")
        self._btn_next = ttk.Button(nav, text="Next  ▶", command=self._go_next)
        self._btn_next.pack(side="right")

    # ------------------------------------------------------------------ #
    # Step routing                                                         #
    # ------------------------------------------------------------------ #

    def _build_step(self) -> None:
        # Clear content frame
        for child in self._content.winfo_children():
            child.destroy()

        self._progress_var.set(self._step + 1)
        self._step_label.config(
            text=f"Step {self._step + 1} of {self._N_STEPS}"
        )
        self._btn_back.config(state="normal" if self._step > 0 else "disabled")
        last = self._step == self._N_STEPS - 1
        self._btn_next.config(text="Finish  ✓" if last else "Next  ▶")

        builders = [
            self._step_welcome,
            self._step_data,
            self._step_model,
            self._step_training_params,
            self._step_checkpoint_export,
            self._step_summary,
        ]
        builders[self._step]()

    def _go_next(self) -> None:
        try:
            self._validate_current_step()
        except ValueError as exc:
            messagebox.showerror("Validation Error", str(exc), parent=self.root)
            return
        if self._step < self._N_STEPS - 1:
            self._step += 1
            self._build_step()
        else:
            self._finish()

    def _go_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._build_step()

    # ------------------------------------------------------------------ #
    # Per-step validation                                                  #
    # ------------------------------------------------------------------ #

    def _validate_current_step(self) -> None:
        """Raise ValueError for the currently active step."""
        if self._step == 3:  # training params
            _validate_positive_int(self._epochs.get(), "epochs")
            _validate_positive_int(self._batch_size.get(), "batch size")
            _validate_positive_float(self._learning_rate.get(), "learning rate")
            _validate_positive_int(self._seed.get() or "1", "seed")
            _validate_fraction(self._val_split.get(), "validation split")
        elif self._step == 4:  # checkpoint/export
            _validate_positive_int(
                self._checkpoint_every_epochs.get(), "checkpoint every N epochs"
            )
            _validate_positive_int(
                self._keep_last_n.get(), "keep last N checkpoints"
            )
            if self._auto_self_train.get():
                _validate_positive_float(
                    self._self_train_interval.get(), "self-training interval (seconds)"
                )

    # ------------------------------------------------------------------ #
    # Step 1: Welcome                                                      #
    # ------------------------------------------------------------------ #

    def _step_welcome(self) -> None:
        f = self._content
        ttk.Label(f, text="🌌  Welcome to Gravitronics", style="Header.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        ttk.Label(
            f,
            text=(
                "This wizard will guide you through configuring a training job\n"
                "for the Lightweight Gravitational Transformer (LGT) model.\n\n"
                "At the end you will be able to:\n"
                "  •  Save a training config to disk\n"
                "  •  Run a dry-run validation\n"
                "  •  Start live training directly from the wizard\n\n"
                "Prerequisites check:"
            ),
            justify="left",
        ).pack(anchor="w")

        prereq_frame = ttk.LabelFrame(f, text="Prerequisites", padding=10)
        prereq_frame.pack(fill="x", pady=10)

        checks = self._check_prerequisites()
        for label, ok, detail in checks:
            row = ttk.Frame(prereq_frame)
            row.pack(fill="x", pady=2)
            icon = "✔" if ok else "✘"
            colour = _OK if ok else _WARN
            tk.Label(
                row, text=f"{icon} {label}", fg=colour, bg=_BG,
                font=_FONT_BODY, anchor="w",
            ).pack(side="left")
            if detail:
                tk.Label(
                    row, text=f"  ({detail})", fg=_FG, bg=_BG,
                    font=("Segoe UI", 9, "italic"), anchor="w",
                ).pack(side="left")

    @staticmethod
    def _check_prerequisites() -> List[tuple]:
        checks = []
        # Python version
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ok = sys.version_info >= (3, 9)
        checks.append(("Python ≥ 3.9", ok, py_ver))

        # PyTorch
        try:
            import torch
            checks.append(("PyTorch", True, torch.__version__))
        except ImportError:
            checks.append(("PyTorch", False, "not installed"))

        # CUDA
        try:
            import torch
            if torch.cuda.is_available():
                checks.append(("CUDA GPU", True, torch.cuda.get_device_name(0)))
            else:
                checks.append(("CUDA GPU", False, "not available (CPU will be used)"))
        except Exception:
            checks.append(("CUDA GPU", False, "unknown"))

        # ONNX (optional)
        try:
            import onnx  # noqa: F401
            checks.append(("ONNX (optional)", True, "available"))
        except ImportError:
            checks.append(("ONNX (optional)", False, "install: pip install onnx onnxruntime"))

        return checks

    # ------------------------------------------------------------------ #
    # Step 2: Data                                                         #
    # ------------------------------------------------------------------ #

    def _step_data(self) -> None:
        f = self._content
        ttk.Label(f, text="📂  Data Source", style="Header.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        ttk.Label(
            f,
            text=(
                "Select the directory containing your training data.\n"
                "Leave blank to use synthetic data (good for smoke tests)."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Data directory:", width=20).pack(side="left")
        ttk.Entry(row, textvariable=self._data_dir, width=40).pack(
            side="left", padx=4
        )
        ttk.Button(
            row, text="Browse…", command=self._browse_data_dir
        ).pack(side="left")

        ttk.Label(
            f,
            text=(
                "\nNote: the real data loader is a placeholder in this release.\n"
                "Synthetic random token sequences are generated automatically\n"
                "when no real data is supplied."
            ),
            foreground="#a6adc8",
            justify="left",
        ).pack(anchor="w")

        val_row = ttk.Frame(f)
        val_row.pack(fill="x", pady=8)
        ttk.Label(val_row, text="Validation split:", width=20).pack(side="left")
        ttk.Entry(val_row, textvariable=self._val_split, width=10).pack(
            side="left", padx=4
        )
        ttk.Label(val_row, text="(0 < val_split < 1)").pack(side="left")

    def _browse_data_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select data directory", parent=self.root
        )
        if path:
            self._data_dir.set(path)

    # ------------------------------------------------------------------ #
    # Step 3: Model                                                        #
    # ------------------------------------------------------------------ #

    def _step_model(self) -> None:
        f = self._content
        ttk.Label(f, text="🤖  Model Configuration", style="Header.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        variant_frame = ttk.LabelFrame(f, text="Model variant", padding=10)
        variant_frame.pack(fill="x", pady=4)
        variants = [
            ("150k  — ~150K params  (edge / Raspberry Pi)", "150k"),
            ("600k  — ~600K params  (laptop / Jetson Nano)", "600k"),
            ("2m    — ~2M params    (server / swarm coordinator)", "2m"),
        ]
        for text, val in variants:
            ttk.Radiobutton(
                variant_frame,
                text=text,
                variable=self._model_variant,
                value=val,
            ).pack(anchor="w", pady=2)

        preset_frame = ttk.LabelFrame(f, text="Training preset", padding=10)
        preset_frame.pack(fill="x", pady=8)
        for text, val in [
            ("Basic   — recommended defaults", "basic"),
            ("Advanced — exposes all hyper-parameters (next step)", "advanced"),
        ]:
            ttk.Radiobutton(
                preset_frame, text=text, variable=self._preset, value=val
            ).pack(anchor="w", pady=2)

        seed_row = ttk.Frame(f)
        seed_row.pack(fill="x", pady=6)
        ttk.Label(seed_row, text="Random seed:", width=20).pack(side="left")
        ttk.Entry(seed_row, textvariable=self._seed, width=10).pack(
            side="left", padx=4
        )
        ttk.Label(seed_row, text="(-1 = non-deterministic)").pack(side="left")

    # ------------------------------------------------------------------ #
    # Step 4: Training hyper-parameters                                    #
    # ------------------------------------------------------------------ #

    def _step_training_params(self) -> None:
        f = self._content
        ttk.Label(f, text="⚙️  Training Parameters", style="Header.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        fields = [
            ("Epochs:", self._epochs, "Number of full passes over the data."),
            ("Batch size:", self._batch_size, "Samples per gradient step."),
            ("Learning rate:", self._learning_rate, "AdamW initial learning rate."),
            ("Validation split:", self._val_split, "Fraction held out for validation."),
        ]
        for label, var, hint in fields:
            row = ttk.Frame(f)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=20).pack(side="left")
            ttk.Entry(row, textvariable=var, width=14).pack(side="left", padx=4)
            ttk.Label(row, text=hint, foreground="#a6adc8").pack(side="left")

        # Device
        device_frame = ttk.LabelFrame(f, text="Compute device", padding=8)
        device_frame.pack(fill="x", pady=8)
        devices = [
            ("auto  — best available (CUDA > MPS > CPU)", "auto"),
            ("cpu", "cpu"),
            ("cuda  — NVIDIA GPU (requires CUDA)", "cuda"),
            ("mps   — Apple Silicon GPU", "mps"),
        ]
        for text, val in devices:
            ttk.Radiobutton(
                device_frame, text=text, variable=self._device, value=val
            ).pack(anchor="w", pady=1)

        ttk.Checkbutton(
            f,
            text="Dry run (validate config + build model, do not train)",
            variable=self._dry_run,
        ).pack(anchor="w", pady=6)

    # ------------------------------------------------------------------ #
    # Step 5: Checkpoints + export                                         #
    # ------------------------------------------------------------------ #

    def _step_checkpoint_export(self) -> None:
        f = self._content
        ttk.Label(f, text="💾  Checkpoints & Export", style="Header.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        ckpt_frame = ttk.LabelFrame(f, text="Checkpointing", padding=10)
        ckpt_frame.pack(fill="x", pady=4)

        for label, var, hint in [
            ("Checkpoint dir:", self._checkpoint_dir, ""),
            ("Every N epochs:", self._checkpoint_every_epochs, "0 = disable epoch-based"),
            ("Keep last N:", self._keep_last_n, "Older checkpoints are deleted"),
        ]:
            row = ttk.Frame(ckpt_frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=20).pack(side="left", padx=4)
            if hint:
                ttk.Label(row, text=hint, foreground="#a6adc8").pack(side="left")

        ckpt_browse = ttk.Frame(ckpt_frame)
        ckpt_browse.pack(fill="x", pady=2)
        ttk.Button(
            ckpt_browse,
            text="Browse checkpoint dir…",
            command=lambda: self._browse_dir(self._checkpoint_dir),
        ).pack(side="left")

        ttk.Checkbutton(
            ckpt_frame,
            text="Save best-model checkpoint (lowest validation loss)",
            variable=self._save_best,
        ).pack(anchor="w", pady=4)

        export_frame = ttk.LabelFrame(f, text="Export", padding=10)
        export_frame.pack(fill="x", pady=8)

        for label, var, options in [
            ("Export dir:", self._export_dir, None),
            ("Format:", self._export_format, ["pt", "onnx", "both"]),
            ("Quantization:", self._export_quantization, ["fp16", "int8", "none"]),
        ]:
            row = ttk.Frame(export_frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=16).pack(side="left")
            if options:
                ttk.Combobox(
                    row, textvariable=var, values=options, state="readonly", width=10
                ).pack(side="left", padx=4)
            else:
                ttk.Entry(row, textvariable=var, width=28).pack(side="left", padx=4)

        ttk.Button(
            export_frame,
            text="Browse export dir…",
            command=lambda: self._browse_dir(self._export_dir),
        ).pack(anchor="w", pady=2)

        # Self-training
        st_frame = ttk.LabelFrame(f, text="Auto self-training", padding=10)
        st_frame.pack(fill="x", pady=4)

        ttk.Checkbutton(
            st_frame,
            text="Enable auto self-training (periodic fine-tuning)",
            variable=self._auto_self_train,
        ).pack(anchor="w")

        for text, val in [
            ("time  — every N seconds", "time"),
            ("data_threshold  — when dataset grows by N samples", "data_threshold"),
            ("metric_threshold  — when val loss drifts", "metric_threshold"),
            ("disabled", "disabled"),
        ]:
            ttk.Radiobutton(
                st_frame,
                text=text,
                variable=self._self_train_policy,
                value=val,
            ).pack(anchor="w", pady=1)

        interval_row = ttk.Frame(st_frame)
        interval_row.pack(fill="x", pady=3)
        ttk.Label(interval_row, text="Interval (sec):", width=16).pack(side="left")
        ttk.Entry(interval_row, textvariable=self._self_train_interval, width=10).pack(
            side="left", padx=4
        )

    def _browse_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(parent=self.root)
        if path:
            var.set(path)

    # ------------------------------------------------------------------ #
    # Step 6: Summary                                                      #
    # ------------------------------------------------------------------ #

    def _step_summary(self) -> None:
        f = self._content
        ttk.Label(f, text="📋  Summary", style="Header.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        cfg = self._build_config()

        # Config-path row
        cfg_row = ttk.Frame(f)
        cfg_row.pack(fill="x", pady=4)
        ttk.Label(cfg_row, text="Save config to:", width=20).pack(side="left")
        ttk.Entry(cfg_row, textvariable=self._config_path, width=35).pack(
            side="left", padx=4
        )
        ttk.Button(
            cfg_row,
            text="Browse…",
            command=self._browse_config_path,
        ).pack(side="left")

        # JSON preview
        text_box = scrolledtext.ScrolledText(
            f,
            height=12,
            bg=_ENTRY_BG,
            fg=_FG,
            font=_FONT_MONO,
            relief="flat",
            insertbackground=_FG,
        )
        text_box.pack(fill="both", expand=True, pady=6)
        text_box.insert("end", json.dumps(cfg, indent=2))
        text_box.config(state="disabled")

        btn_row = ttk.Frame(f)
        btn_row.pack(fill="x", pady=4)

        self._status_label = ttk.Label(btn_row, text="", foreground=_OK)
        self._status_label.pack(side="bottom", fill="x")

        ttk.Button(
            btn_row,
            text="💾  Save Config",
            command=self._save_config,
        ).pack(side="left", padx=4)

        ttk.Button(
            btn_row,
            text="🔍  Dry Run",
            command=self._run_dry_run,
        ).pack(side="left", padx=4)

    def _browse_config_path(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self._config_path.get(),
            parent=self.root,
        )
        if path:
            self._config_path.set(path)

    def _build_config(self) -> Dict[str, Any]:
        return {
            "model_variant": self._model_variant.get(),
            "preset": self._preset.get(),
            "data_dir": self._data_dir.get(),
            "val_split": float(self._val_split.get() or "0.1"),
            "epochs": int(self._epochs.get() or "10"),
            "batch_size": int(self._batch_size.get() or "32"),
            "learning_rate": float(self._learning_rate.get() or "3e-4"),
            "seed": int(self._seed.get() or "42"),
            "device": self._device.get(),
            "checkpoint_dir": self._checkpoint_dir.get(),
            "checkpoint_every_epochs": int(
                self._checkpoint_every_epochs.get() or "1"
            ),
            "keep_last_n_checkpoints": int(self._keep_last_n.get() or "3"),
            "save_best_checkpoint": self._save_best.get(),
            "export_dir": self._export_dir.get(),
            "export_format": self._export_format.get(),
            "export_quantization": self._export_quantization.get(),
            "auto_self_train": self._auto_self_train.get(),
            "self_train_policy": self._self_train_policy.get(),
            "self_train_interval_sec": float(
                self._self_train_interval.get() or "3600"
            ),
            "dry_run": self._dry_run.get(),
        }

    def _save_config(self) -> None:
        """Validate then save the config JSON."""
        try:
            from gravitronics.training.config import TrainingConfig

            cfg_dict = self._build_config()
            cfg = TrainingConfig.from_dict(cfg_dict)
            cfg.validate()
            path = self._config_path.get() or "training_config.json"
            cfg.save(path)
            self._set_status(f"Config saved to '{path}'.", ok=True)
        except Exception as exc:
            self._set_status(f"Error: {exc}", ok=False)
            messagebox.showerror("Save Error", str(exc), parent=self.root)

    def _run_dry_run(self) -> None:
        """Save config and run a dry-run in a background thread."""
        self._save_config()
        path = self._config_path.get() or "training_config.json"
        if not os.path.isfile(path):
            messagebox.showerror(
                "Dry Run Error",
                "Please save the config first.",
                parent=self.root,
            )
            return
        self._set_status("Running dry-run validation…", ok=True)
        t = threading.Thread(target=self._dry_run_thread, args=(path,), daemon=True)
        t.start()

    def _dry_run_thread(self, config_path: str) -> None:
        try:
            from gravitronics.training.config import TrainingConfig
            from gravitronics.training.trainer import Trainer

            cfg = TrainingConfig.load(config_path)
            cfg.dry_run = True
            trainer = Trainer(cfg)
            result = trainer.train()
            msg = (
                f"Dry run OK — model built successfully.\n"
                f"Variant: {cfg.model_variant}, device: {cfg.resolve_device()}"
            )
            self.root.after(0, lambda: self._set_status(msg, ok=True))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Dry Run", msg, parent=self.root
                ),
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda: self._set_status(f"Dry run FAILED: {exc}", ok=False),
            )
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Dry Run Error", str(exc), parent=self.root
                ),
            )

    def _set_status(self, msg: str, ok: bool = True) -> None:
        colour = _OK if ok else _WARN
        if hasattr(self, "_status_label"):
            self._status_label.config(text=msg, foreground=colour)

    # ------------------------------------------------------------------ #
    # Finish                                                               #
    # ------------------------------------------------------------------ #

    def _finish(self) -> None:
        """Final 'Finish' action — save config and optionally start training."""
        self._save_config()
        cfg_dict = self._build_config()
        self._config = cfg_dict

        answer = messagebox.askyesnocancel(
            "Start Training?",
            "Configuration saved.\n\nStart training now?",
            parent=self.root,
        )
        if answer is True:
            self._launch_training(cfg_dict)
        elif answer is False:
            self.root.destroy()
        # None (Cancel) = stay in wizard

    def _launch_training(self, cfg_dict: Dict[str, Any]) -> None:
        """Open a training progress window and start training in a background thread."""
        if self.on_complete:
            self.on_complete(cfg_dict)
            return
        _TrainingProgressWindow(self.root, cfg_dict)


# --------------------------------------------------------------------------- #
# Training progress window                                                     #
# --------------------------------------------------------------------------- #


class _TrainingProgressWindow:
    """A secondary window showing live training progress."""

    def __init__(self, parent: tk.Tk, cfg_dict: Dict[str, Any]) -> None:
        self._win = tk.Toplevel(parent)
        self._win.title("Gravitronics — Training")
        self._win.geometry("600x400")
        self._win.configure(bg=_BG)

        ttk.Label(
            self._win,
            text="🌌  Training in progress…",
            style="Header.TLabel",
        ).pack(pady=10, padx=20, anchor="w")

        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            self._win,
            variable=self._progress_var,
            maximum=1.0,
            style="Horizontal.TProgressbar",
        ).pack(fill="x", padx=20, pady=4)

        self._status_var = tk.StringVar(value="Initialising…")
        ttk.Label(self._win, textvariable=self._status_var).pack(
            anchor="w", padx=20
        )

        self._log = scrolledtext.ScrolledText(
            self._win,
            height=12,
            bg=_ENTRY_BG,
            fg=_FG,
            font=_FONT_MONO,
            state="disabled",
        )
        self._log.pack(fill="both", expand=True, padx=20, pady=8)

        stop_btn = ttk.Button(
            self._win, text="⏹  Stop", command=self._stop
        )
        stop_btn.pack(pady=6)

        self._stop_event = __import__("threading").Event()
        self._thread = __import__("threading").Thread(
            target=self._run, args=(cfg_dict,), daemon=True
        )
        self._thread.start()

    def _run(self, cfg_dict: Dict[str, Any]) -> None:
        try:
            from gravitronics.training.config import TrainingConfig
            from gravitronics.training.trainer import Trainer

            cfg = TrainingConfig.from_dict(cfg_dict)
            trainer = Trainer(
                cfg,
                progress_callback=self._on_progress,
                stop_event=self._stop_event,
            )
            result = trainer.train()
            self._win.after(
                0,
                lambda: self._append_log(
                    f"\nTraining finished: {result}\n"
                ),
            )
        except Exception as exc:
            self._win.after(0, lambda: self._append_log(f"\nERROR: {exc}\n"))

    def _on_progress(self, info: Dict[str, Any]) -> None:
        msg = (
            f"Step {info['step']} | Epoch {info['epoch']} | "
            f"Loss {info['loss']:.4f} | ETA {info['eta_sec']:.0f}s"
        )
        pct = info.get("progress", 0.0)
        self._win.after(0, lambda: self._progress_var.set(pct))
        self._win.after(0, lambda: self._status_var.set(msg))
        self._win.after(0, lambda: self._append_log(msg + "\n"))

    def _append_log(self, text: str) -> None:
        self._log.config(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _stop(self) -> None:
        self._stop_event.set()
        self._status_var.set("Stop requested…")


# --------------------------------------------------------------------------- #
# Entry-point                                                                  #
# --------------------------------------------------------------------------- #


def run_wizard(on_complete: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    """Create and run the Gravitronics setup wizard.

    Parameters
    ----------
    on_complete:
        Optional callback invoked with the final config dict when the user
        clicks "Finish / Start Training".
    """
    root = tk.Tk()
    wizard = GravitronicsSetupWizard(root, on_complete=on_complete)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    run_wizard()
