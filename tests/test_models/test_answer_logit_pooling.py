"""Tests for InBedder-style answer-logit pooling and its registry wiring."""

import torch

import mteb
from mteb.models import InstructAnswerEncoderWrapper, pool_answer_logits

_INBEDDER_MODEL = "BrandonZYW/roberta-large-InBedder"


def test_inbedder_model_registered():
    """The InBedder checkpoint is discoverable through the public model registry."""
    meta = mteb.get_model_meta(_INBEDDER_MODEL)

    assert meta.loader is InstructAnswerEncoderWrapper
    assert meta.use_instructions is True
    # Embedding is the standardized answer distribution over the vocabulary.
    assert meta.embed_dim == 50265
    assert meta.loader_kwargs["n_answer_tokens"] == 3


def test_inbedder_present_in_get_model_metas():
    """The wiring surfaces the model to the eval-path helper get_model_metas."""
    names = {m.name for m in mteb.get_model_metas()}
    assert _INBEDDER_MODEL in names


def test_pool_answer_logits_averages_over_mask_positions():
    """Only the answer (mask) positions contribute to the pooled representation."""
    # batch=1, seq_len=3, vocab=2. Answer slots are positions 1 and 2.
    logits = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [3.0, 5.0]]])
    answer_mask = torch.tensor([[False, True, True]])

    pooled = pool_answer_logits(logits, answer_mask)

    # Mean over the two answer slots ([2.0, 4.0]), ignoring the first position.
    mean_answer = torch.tensor([[2.0, 4.0]])
    expected = (mean_answer - mean_answer.mean()) / mean_answer.std()
    assert pooled.shape == (1, 2)
    torch.testing.assert_close(pooled, expected)


def test_pool_answer_logits_standardizes_per_sample():
    """Each row is standardized independently (zero mean, scale invariant)."""
    logits = torch.tensor([[[0.0, 0.0], [1.0, 5.0]], [[0.0, 0.0], [10.0, 20.0]]])
    answer_mask = torch.tensor([[False, True], [False, True]])

    pooled = pool_answer_logits(logits, answer_mask)

    torch.testing.assert_close(pooled.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    # Both rows collapse to the same standardized direction despite different scales.
    torch.testing.assert_close(pooled[0], pooled[1])
