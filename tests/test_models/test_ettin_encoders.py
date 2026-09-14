"""Tests for the base Ettin encoder entries wired into the model registry.

These exercise the registry integration (``mteb.get_model_meta`` /
``mteb.get_model_metas``), not the new metadata objects in isolation, to prove the
``jhu-clsp/ettin-encoder-*`` entries are discoverable through MTEB's public API.
"""

import pytest

import mteb
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

# (name, embed_dim, n_embedding_parameters). n_embedding_parameters == 50368 (vocab) *
# embed_dim, matching the paired ettin-reranker-* entries that share the vocabulary.
ETTIN_ENCODERS = [
    ("jhu-clsp/ettin-encoder-17m", 256, 12_894_208),
    ("jhu-clsp/ettin-encoder-32m", 384, 19_341_312),
    ("jhu-clsp/ettin-encoder-68m", 512, 25_788_416),
    ("jhu-clsp/ettin-encoder-150m", 768, 38_682_624),
    ("jhu-clsp/ettin-encoder-400m", 1024, 51_576_832),
    ("jhu-clsp/ettin-encoder-1b", 1792, 90_259_456),
]


@pytest.mark.parametrize(("name", "embed_dim", "n_embedding"), ETTIN_ENCODERS)
def test_ettin_encoder_registered_as_dense(name, embed_dim, n_embedding):
    meta = mteb.get_model_meta(name)

    assert meta.model_type == ["dense"]
    assert not meta.is_cross_encoder
    assert meta.loader is SentenceTransformerEncoderWrapper
    assert meta.embed_dim == embed_dim
    assert meta.n_embedding_parameters == n_embedding
    assert meta.open_weights is True
    # 50368-token ModernBERT vocabulary shared with the reranker siblings.
    assert meta.n_embedding_parameters == 50368 * embed_dim


def test_ettin_encoders_filtered_via_get_model_metas():
    """The encoders should be reachable through the registry-wide dense filter."""
    dense = {m.name for m in mteb.get_model_metas(model_types=["dense"])}
    for name, _embed_dim, _n_embedding in ETTIN_ENCODERS:
        assert name in dense


def test_ettin_encoders_are_zero_shot_and_distinct_from_rerankers():
    """Base encoders carry no MTEB training data and are separate from the rerankers."""
    encoder = mteb.get_model_meta("jhu-clsp/ettin-encoder-17m")
    reranker = mteb.get_model_meta("cross-encoder/ettin-reranker-17m-v1")

    # Encoder itself declares no MTEB task training data -> zero-shot everywhere.
    assert encoder.training_datasets is None
    assert encoder.is_zero_shot_on(["MSMARCO"]) is None

    # The reranker is adapted from this encoder; they must stay distinct registrations.
    assert reranker.model_type == ["cross-encoder"]
    assert encoder.name != reranker.name
    assert encoder.name in reranker.adapted_from
