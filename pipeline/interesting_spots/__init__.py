"""interesting_spots — a tiny web app to hand-label the *interesting* spots per image.

Page through the images of one folder, click inside a spot to toggle it green, and the
selection (salamander_id -> [spot_id, ...]) is written to a JSON file after every click.
The click is resolved to a real spot by looking up the per-spot masks already stored in
``images/<folder>/contours/contours.db`` — so the whole spot turns green, not just a dot.

The CLI is ``scripts/tools/interesting_spot_selector.py`` (``pixi run interesting-spot-selector``).
"""
from .app import SpotSelectorApp
from .server import run

__all__ = ["SpotSelectorApp", "run"]
