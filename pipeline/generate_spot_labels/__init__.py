"""generate_spot_labels — spot-label generation pipeline (pure logic).

Importable package; the CLI that configures and calls it is ``scripts/dataset/extract_spot_labels.py``
(``pixi run extract-spot-labels <all|segment|contours|masks>``). Typical programmatic use::

    from generate_spot_labels import runner
    runner.run("all_sasa_norm", max_bleed=0.35)

Submodules
----------
* ``llm_animal_count``      — stage 0: how many salamanders per frame -> animals/ (standalone)
* ``llm_spot_segmentation`` — stage 1: Gemini inpaint -> purple/ (judge + escalation)
* ``extract_spot_contours`` — stage 2: purple/ -> per-spot contours + masks in DuckDB
* ``runner``                — full pipeline over one input dir
* ``_common``               — shared path/env helpers (stdlib only)
"""
from ._common import reconfigure_utf8

__all__ = ["reconfigure_utf8"]
