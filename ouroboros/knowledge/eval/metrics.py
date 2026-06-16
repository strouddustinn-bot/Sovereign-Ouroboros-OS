"""Retrieval evaluation metrics: Recall@k and MRR.

The architecture doc says: "A golden query set is the only way to know if a
change helped or hurt. Build it before tuning anything." This module runs the
golden set against the live KnowledgeBase and reports whether changes to the
retrieval pipeline improved or degraded quality.

Usage::

    from ouroboros.knowledge.eval.metrics import evaluate
    from ouroboros.knowledge import KnowledgeBase

    kb = KnowledgeBase()
    results = evaluate(kb, golden_set_path="knowledge/eval/golden_set.jsonl")
    print(results)   # {"recall@1": 0.8, "recall@5": 1.0, "mrr": 0.9}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Fraction of relevant chunks found in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant result; 0.0 if none found."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rr_values: list[float]) -> float:
    """Mean of reciprocal ranks across all queries."""
    return sum(rr_values) / len(rr_values) if rr_values else 0.0


def evaluate(
    kb: Any,  # KnowledgeBase — loose type to avoid circular import
    golden_set_path: str | Path = "ouroboros/knowledge/eval/golden_set.jsonl",
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Run the golden set through *kb* and return retrieval metrics.

    Each line in *golden_set_path* is a JSON object::

        {"query": "how do refunds work?",
         "relevant_source_uris": ["s3://docs/billing.md"],
         "filters": {"domain": "billing"}}

    Parameters
    ----------
    kb:
        A ``KnowledgeBase`` instance (already has documents ingested).
    golden_set_path:
        Path to the JSONL golden set file.
    k_values:
        List of k values for Recall@k.  Defaults to [1, 3, 5].

    Returns
    -------
    dict[str, float]
        Metrics: ``recall@1``, ``recall@3``, ``recall@5``, ``mrr``.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    path = Path(golden_set_path)
    if not path.exists():
        return {"error": f"golden set not found: {path}"}

    queries: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    if not queries:
        return {"error": "golden set is empty"}

    recall_buckets: dict[int, list[float]] = {k: [] for k in k_values}
    rr_values: list[float] = []

    for item in queries:
        query_text: str = item.get("query", "")
        relevant_uris: set[str] = set(item.get("relevant_source_uris", []))
        filters: dict[str, Any] = item.get("filters", {})

        hits = kb.query(query_text, **filters)
        retrieved_ids = [h.chunk.id for h in hits]

        # Map source URIs → chunk ids via hits
        relevant_ids: set[str] = {
            h.chunk.id
            for h in hits
            if h.chunk.metadata.source_uri in relevant_uris
        }

        for k in k_values:
            recall_buckets[k].append(recall_at_k(retrieved_ids, relevant_ids, k))
        rr_values.append(reciprocal_rank(retrieved_ids, relevant_ids))

    metrics: dict[str, float] = {}
    for k in k_values:
        vals = recall_buckets[k]
        metrics[f"recall@{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    metrics["mrr"] = round(mean_reciprocal_rank(rr_values), 4)
    metrics["n_queries"] = float(len(queries))
    return metrics
