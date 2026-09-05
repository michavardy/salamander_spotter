"""Shared logging setup for every pipeline script.

``get_logger(name)`` returns a ``logging.Logger`` that always writes to stdout (timestamp +
level + module name + message) and, when the app has configured it, also writes to a file.
File logging is controlled from the app's Settings page (``app/settings_store.py``'s
``log_dir``/``log_level`` keys): ``app/pipeline_bridge/training.py`` passes those through to
the training subprocess as ``SPOTTER_LOG_DIR``/``SPOTTER_LOG_LEVEL``. Both env vars can also be
exported by hand for a standalone run, the same way ``SOURCE``/``EMB_TABLE`` are (see
``pipeline/spot_transformer/core/data.py``). Default (both unset) is stdout only.

    from pathlib import Path
    from pipeline.utils.logger_utils import get_logger
    logger = get_logger(Path(__file__).stem)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _configured:
        return logger
    _configured.add(name)

    level_name = os.environ.get("SPOTTER_LOG_LEVEL", "").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = os.environ.get("SPOTTER_LOG_DIR", "").strip()
    if log_dir and level_name:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name.replace('.', '_')}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
