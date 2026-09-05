"""sweeps — runnable experiment drivers that train the models and write results tables.

NOTE: each script here is normally run directly (``python .../train_all13.py``), which never
triggers this ``__init__.py`` — a direct script run only executes the file itself, not its
parent packages. Every script must do its own ``pipeline.utils`` bootstrap + import (see
``train_all13.py``); this file only helps the rare case of a proper dotted-package import
(e.g. ``import pipeline.spot_transformer.sweeps``).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # sweeps -> spot_transformer -> pipeline -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.utils.encoding_utils import reconfigure_stream_to_utf8
reconfigure_stream_to_utf8()