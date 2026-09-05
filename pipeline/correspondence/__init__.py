"""Human spot-correspondence labelling — the label the matcher has never had.

Image-level labels say "these two photos are the same animal". A correspondence says
"spot 7 in photo A is the same physical spot as spot 3 in photo B". That distinction is what
separates the four failure modes the pipeline currently cannot tell apart: the extractor missed
the spot, shattered it, encoded it inconsistently, or the matcher simply picked the wrong
neighbour. See ``analyze.py`` for what the labels buy.
"""
