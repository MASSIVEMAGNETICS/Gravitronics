"""
LGTConfig — Hyperparameter configuration for Lightweight Gravitational Transformer.

All model hyperparameters live here so that they can be serialised, logged, and
passed between components without tight coupling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LGTConfig:
    """Configuration for the Lightweight Gravitational Transformer.

    Attributes
    ----------
    vocab_size:
        Number of tokens in the vocabulary.
    hidden_dim:
        Dimension of hidden states throughout the model.
    num_heads:
        Number of attention heads.  Must divide ``hidden_dim`` evenly.
    num_layers:
        Number of stacked transformer blocks.
    max_seq_len:
        Maximum sequence length accepted by the model.
    dropout:
        Dropout probability applied throughout the model.
    gravitational_constant:
        Scaled gravitational constant *G* used in the attention force law.
    hawking_temperature:
        Temperature parameter for Hawking-radiation regularisation.
    bekenstein_limit:
        Maximum absolute value of attention logits (information capacity bound).
    curvature_scale:
        Global scale factor applied to the curvature tensor.
    mass_init_scale:
        Scale factor for initialising per-token mass projections.
    sparse_top_k:
        Number of top-k keys retained per query in sparse attention.
        Set to a value ≥ ``max_seq_len`` to disable sparsity.
    use_hawking_radiation:
        Whether to apply Hawking-radiation regularisation to attention.
    use_curved_positions:
        Whether to use curvature-modulated position embeddings.
    use_mirror_layer:
        Whether to wrap each attention module with a MirrorLayer.
    model_variant:
        Human-readable size tag — ``"150k"``, ``"600k"``, or ``"2m"``.
    """

    # Core architecture
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512

    # Regularisation
    dropout: float = 0.1

    # Gravitational physics parameters
    gravitational_constant: float = 6.674e-3
    hawking_temperature: float = 0.1
    bekenstein_limit: float = 50.0
    curvature_scale: float = 1.0
    mass_init_scale: float = 1.0

    # Sparse attention
    sparse_top_k: int = 32

    # Feature flags
    use_hawking_radiation: bool = True
    use_curved_positions: bool = True
    use_mirror_layer: bool = True

    # Variant tag
    model_variant: str = "150k"

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Validate field consistency after initialisation."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.sparse_top_k < 1:
            raise ValueError(f"sparse_top_k must be ≥ 1, got {self.sparse_top_k}.")
        logger.debug("LGTConfig validated: variant=%s", self.model_variant)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_variant(cls, variant: str) -> "LGTConfig":
        """Return a pre-set :class:`LGTConfig` for the requested size variant.

        Parameters
        ----------
        variant:
            One of ``"150k"``, ``"600k"``, or ``"2m"``.

        Returns
        -------
        LGTConfig
            A fully validated configuration object.

        Raises
        ------
        ValueError
            If *variant* is not one of the supported strings.
        """
        # Vocab sizes are kept small so the total parameter counts land near
        # their variant targets (150 k / 600 k / 2 M).  With vocab_size=32000
        # the embedding layer alone would overwhelm the transformer parameters.
        presets: dict[str, dict] = {
            "150k": {
                "vocab_size": 256,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "max_seq_len": 64,
                "sparse_top_k": 16,
                "model_variant": "150k",
            },
            "600k": {
                "vocab_size": 512,
                "hidden_dim": 128,
                "num_heads": 4,
                "num_layers": 3,
                "max_seq_len": 128,
                "sparse_top_k": 32,
                "model_variant": "600k",
            },
            "2m": {
                "vocab_size": 1024,
                "hidden_dim": 256,
                "num_heads": 8,
                "num_layers": 2,
                "max_seq_len": 256,
                "sparse_top_k": 64,
                "model_variant": "2m",
            },
        }
        if variant not in presets:
            raise ValueError(
                f"Unknown variant '{variant}'.  Choose from: {list(presets)}."
            )
        logger.info("Creating LGTConfig from variant '%s'.", variant)
        return cls(**presets[variant])

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of all configuration fields."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LGTConfig":
        """Construct an :class:`LGTConfig` from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
