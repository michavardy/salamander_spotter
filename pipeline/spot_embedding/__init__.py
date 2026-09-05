"""spot_embedding — salamander identification matcher (pure logic).

Importable package; the CLIs that configure and call it live in ``scripts/embedding/emb_*.py``
(``pixi run emb-prepare | emb-eval | emb-selfcheck | emb-train | emb-identify | emb-bakeoff``).

This package builds the matcher designed in
``docs/per_spot_embedding_aggregation.md`` and follows the phased plan in
``docs/spot_embedding_aggregation_plan.md``. **Phase 0** (this commit) is the shared
evaluation harness only: data access, CV splits, metrics, and a harness exercised by a
dummy (random) and an oracle (label-cheating) matcher.

Sub-packages
------------
* ``data``   — read ``contours.db`` into ``SpotSet``; make leakage-safe CV splits
* ``eval``   — retrieval / verification / open-set metrics + the harness + reports
* ``models`` — the ``Matcher`` interface and the Phase-0 dummy / oracle
* ``augment`` / ``encoders`` / ``train`` / ``match`` — placeholders for later phases
"""
from ._common import reconfigure_utf8

__all__ = ["reconfigure_utf8"]
