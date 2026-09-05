"""train — losses + per-fold trainer for the learned aggregators (Phase 3).

* ``features`` — SpotSet → token tensors (invariant shape ⊕ log-area ⊕ rel-position) + augment
* ``losses``   — supervised contrastive
* ``trainer``  — train one encoder per CV fold; embed sets / tokens at inference
"""
from . import crops, features, losses, trainer
from .trainer import (
    TrainConfig,
    embed_sets,
    embed_sets_cnn,
    embed_tokens,
    train_encoder,
    train_encoder_cnn,
)

__all__ = ["crops", "features", "losses", "trainer", "TrainConfig", "train_encoder",
           "train_encoder_cnn", "embed_sets", "embed_sets_cnn", "embed_tokens"]
