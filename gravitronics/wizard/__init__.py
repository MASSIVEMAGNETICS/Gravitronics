"""
gravitronics.wizard — interactive training-setup wizard.

Exported symbols
----------------
RunConfig    Dataclass describing a complete training run configuration.
WizardSetup  Interactive assistant that populates a RunConfig.
run_wizard   Convenience function: create wizard + run in one call.
"""

from gravitronics.wizard.setup import RunConfig, WizardSetup, run_wizard

__all__ = ["RunConfig", "WizardSetup", "run_wizard"]
