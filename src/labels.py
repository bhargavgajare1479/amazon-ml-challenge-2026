"""
Ground truth label parsing. Kept as its own module (rather than folded into
normalize.py) to match the team repo's existing labels.py naming.
"""
from .normalize import parse_ground_truth  # noqa: F401  (re-exported for convenience)
