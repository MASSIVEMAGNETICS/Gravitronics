"""Gravitronics wizard package — setup utilities for training runs.

This package provides two wizard implementations:

* :mod:`gravitronics.wizard.setup_wizard` — graphical (tkinter) wizard for
  Windows / desktop environments.
* :mod:`gravitronics.wizard.setup` — text-based interactive wizard that works
  in any terminal, and can also be invoked non-interactively from scripts.

Exported symbols (text wizard)
------------------------------
RunConfig    Dataclass describing a complete file-based training run.
WizardSetup  Interactive assistant that populates a :class:`RunConfig`.
run_wizard   Convenience function: create wizard + run in one call.
"""

from gravitronics.wizard.setup import RunConfig, WizardSetup, run_wizard

__all__ = ["RunConfig", "WizardSetup", "run_wizard"]
