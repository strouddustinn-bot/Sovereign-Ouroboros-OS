"""ChronoWeave – Counterfactual Timeline Engine layer.

The Temporal Simulator of Project Ouroboros. It replaces linear planning with a
multiverse: for each imagined prototype it spawns parallel hypothetical futures,
simulates their outcomes via lightweight deterministic causal heuristics, and
collapses the timelines into the single highest-value probabilistic path.
"""

from ouroboros.chronoweave.timeline_engine import ChronoWeave

__all__ = ["ChronoWeave"]
