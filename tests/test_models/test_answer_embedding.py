"""Tests for the InBedder answer-embedding model integration.

Covers (1) that the model auto-registers into the existing mteb registry with
the expected instruction-following metadata, and (2) the InBedder answer-position
pooling mechanism, exercised with lightweight stubs so no weights are downloaded.
"""

from types import SimpleNamespace

import numpy as np
import torch

import mteb
from mteb.models.model_implementations.answer_embedding_models import (
    InstructionAnswerEncoder,
)
from mteb.models.model_meta import ScoringFunction

MODEL_NAME = "KomeijiForce/inbedder-roberta-large"


def test_inbedder_registered_in_mteb():
    """The new module is picked up by the existing auto-registry in mteb."""
    meta = mteb.get_model_meta(MODEL_NAME)

    assert meta.name == MODEL_NAME
    assert meta.loader is InstructionAnswerEncoder
    assert meta.use_instructions is True
    assert meta.embed_dim == 1024
    assert meta.similarity_fn_name == ScoringFunction.COSINE

    all_names = {m.name for m in mteb.get_model_metas()}
    assert MODEL_NAME in all_names


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    mask_token = "<mask>"
    mask_token_id = 50264

    def __init__(self, n_mask):
        self.n_mask = n_mask

    def __call__(self, texts, prompts, **kwargs):
        # one content token followed by exactly n_mask answer slots per example
        row = [1] + [self.mask_token_id] * self.n_mask
        input_ids = torch.tensor([row for _ in texts])
        return _FakeBatch(input_ids=input_ids)


class _FakeEncoder:
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, input_ids=None, **kwargs):
        bsz, seq = input_ids.shape
        torch.manual_seed(0)
        hidden_states = torch.randn(bsz, seq, self.hidden)
        return SimpleNamespace(last_hidden_state=hidden_states)


def _make_encoder(n_mask=3, hidden=8):
    enc = InstructionAnswerEncoder.__new__(InstructionAnswerEncoder)
    enc.n_mask = n_mask
    enc.max_length = 512
    enc.device = "cpu"
    enc.dense = torch.nn.Identity()
    enc.layer_norm = torch.nn.Identity()
    enc.tokenizer = _FakeTokenizer(n_mask)
    enc.model = _FakeEncoder(hidden)
    return enc


def test_answer_pooling_shape_and_standardization():
    enc = _make_encoder(n_mask=3, hidden=8)

    out = enc._answer_embeddings(["I love cats", "I love dogs"], "what is the topic?")

    # one embedding per text, sized to the model hidden dimension
    assert out.shape == (2, 8)
    # InBedder standardizes each embedding across the hidden dimension
    assert torch.allclose(out.mean(1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(out.std(1), torch.ones(2), atol=1e-4)


def test_encode_returns_numpy_embeddings():
    enc = _make_encoder(n_mask=3, hidden=8)
    task_metadata = SimpleNamespace(
        prompt="what is the topic?", name="dummy", type="Retrieval"
    )
    inputs = [{"text": ["I love cats", "I love dogs"]}]

    emb = enc.encode(
        inputs,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
    )

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 8)
