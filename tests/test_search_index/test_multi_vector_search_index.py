"""Tests for the multi-vector (late-interaction) search backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mteb.models.model_meta import ScoringFunction
from mteb.models.search_encoder_index import MultiVectorSearchIndex
from mteb.similarity_functions import max_sim


class _FakeMaxSimModel:
    """Minimal stand-in exposing the ``mteb_model_meta.similarity_fn_name`` seam."""

    class _Meta:
        similarity_fn_name = ScoringFunction.MAX_SIM

    mteb_model_meta = _Meta()


def _random_multi_vectors(
    n: int, n_tokens: int, dim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_tokens, dim)).astype(np.float32)


def _brute_force_max_sim(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Reference MaxSim scores using the existing `mteb.similarity_functions`."""
    return max_sim(torch.from_numpy(queries), torch.from_numpy(docs)).numpy()


def test_for_model_gates_on_max_sim():
    assert MultiVectorSearchIndex.for_model(_FakeMaxSimModel()) is not None

    class _CosineModel:
        class _Meta:
            similarity_fn_name = ScoringFunction.COSINE

        mteb_model_meta = _Meta()

    assert MultiVectorSearchIndex.for_model(_CosineModel()) is None


def test_rejects_non_max_sim_model():
    class _CosineModel:
        class _Meta:
            similarity_fn_name = ScoringFunction.COSINE

        mteb_model_meta = _Meta()

    with pytest.raises(ValueError, match="MaxSim"):
        MultiVectorSearchIndex(_CosineModel())


def test_exact_search_matches_reference_max_sim():
    """The default (exact) backend must reproduce brute-force MaxSim ranking."""
    docs = _random_multi_vectors(n=12, n_tokens=5, dim=8, seed=0)
    queries = _random_multi_vectors(n=4, n_tokens=3, dim=8, seed=1)
    doc_ids = [f"d{i}" for i in range(len(docs))]

    index = MultiVectorSearchIndex(_FakeMaxSimModel())
    index.add_documents(docs, doc_ids)

    top_k = 5
    scores, idxs = index.search(queries, top_k, similarity_fn=max_sim)

    reference = _brute_force_max_sim(queries, docs)
    for query_i in range(len(queries)):
        expected_order = np.argsort(-reference[query_i])[:top_k]
        assert idxs[query_i] == expected_order.tolist()
        np.testing.assert_allclose(
            scores[query_i], reference[query_i][expected_order], rtol=1e-5, atol=1e-5
        )


def test_candidate_pruning_recovers_top_result():
    """Pruned two-stage retrieval keeps the true best document as rank 1."""
    docs = _random_multi_vectors(n=40, n_tokens=6, dim=16, seed=2)
    queries = _random_multi_vectors(n=3, n_tokens=4, dim=16, seed=3)
    doc_ids = [f"d{i}" for i in range(len(docs))]

    pruned = MultiVectorSearchIndex(
        _FakeMaxSimModel(), n_candidates=10, tokens_per_query=10
    )
    pruned.add_documents(docs, doc_ids)
    _, pruned_idxs = pruned.search(queries, top_k=5, similarity_fn=max_sim)

    reference = _brute_force_max_sim(queries, docs)
    for query_i in range(len(queries)):
        assert len(pruned_idxs[query_i]) <= 10
        assert pruned_idxs[query_i][0] == int(np.argmax(reference[query_i]))


def test_reranking_mode_restricts_to_candidates():
    docs = _random_multi_vectors(n=8, n_tokens=4, dim=8, seed=4)
    queries = _random_multi_vectors(n=2, n_tokens=3, dim=8, seed=5)
    doc_ids = [f"d{i}" for i in range(len(docs))]

    index = MultiVectorSearchIndex(_FakeMaxSimModel())
    index.add_documents(docs, doc_ids)

    top_ranked = {"q0": ["d3", "d1", "d5"], "q1": ["d0", "d7"]}
    query_idx_to_id = {0: "q0", 1: "q1"}

    scores, idxs = index.search(
        queries,
        top_k=2,
        similarity_fn=max_sim,
        top_ranked=top_ranked,
        query_idx_to_id=query_idx_to_id,
    )

    # Reranking returns indices local to each query's candidate list.
    assert all(i < len(top_ranked["q0"]) for i in idxs[0])
    assert all(i < len(top_ranked["q1"]) for i in idxs[1])
    # Scores are sorted descending.
    for query_scores in scores:
        assert query_scores == sorted(query_scores, reverse=True)


def test_clear_resets_state():
    docs = _random_multi_vectors(n=3, n_tokens=4, dim=8, seed=6)
    index = MultiVectorSearchIndex(_FakeMaxSimModel())
    index.add_documents(docs, ["a", "b", "c"])
    index.clear()

    assert index.idxs == []
    with pytest.raises(ValueError, match="No documents indexed"):
        index.search(docs, top_k=1, similarity_fn=max_sim)
