"""Tests for the RoBERTa InBedder instruction-following embedder.

Exercises the wiring into MTEB's model registry (via the public ``mteb`` API)
and the paper's answer-pooling mechanism, without downloading any weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import mteb
from mteb.models.model_implementations.inbedder_models import InBedderRobertaModel

MODEL_NAME = "KomeijiForce/inbedder-roberta-large"


def _collate(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {"text": [row["text"] for row in batch]}


def test_inbedder_registered_with_custom_loader():
    """The new module is auto-discovered and wired into the model registry."""
    meta = mteb.get_model_meta(MODEL_NAME)

    assert meta.loader is InBedderRobertaModel
    assert meta.use_instructions is True
    assert meta.embed_dim == 1024
    assert meta.license == "mit"
    assert meta.similarity_fn_name.value == "cosine"
    assert meta.name in {m.name for m in mteb.get_model_metas()}


def test_answer_pooling_mean_over_mask_tokens():
    """Answer tokens are mean-pooled per example from the raw hidden states."""
    model = object.__new__(InBedderRobertaModel)

    # 2 examples x 4 tokens x 4 dims; the last 3 tokens per example are masks.
    hidden_states = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    answer_mask = torch.tensor(
        [[False, True, True, True], [False, True, True, True]]
    )
    pooled = model._pool_answers(hidden_states, answer_mask)

    assert pooled.shape == (2, 4)
    # Each row is the mean of its three mask-token vectors, with no projection
    # or z-score normalization applied.
    assert torch.allclose(pooled, hidden_states[:, 1:, :].mean(1))


def test_encode_resolves_instruction_and_batches():
    """encode() feeds the task instruction as the question and concatenates batches."""
    model = object.__new__(InBedderRobertaModel)
    seen: dict[str, object] = {}

    def fake_embed_batch(texts, instruction):
        seen["instruction"] = instruction
        seen.setdefault("n_texts", 0)
        seen["n_texts"] += len(texts)
        return np.zeros((len(texts), 8), dtype=np.float32)

    model._embed_batch = fake_embed_batch  # type: ignore[method-assign]

    inputs = DataLoader(
        [{"text": "cats sit on mats"}, {"text": "dogs run in parks"}, {"text": "birds"}],
        batch_size=2,
        collate_fn=_collate,
    )
    task_metadata = SimpleNamespace(
        name="MockInBedderTask",
        type="Clustering",
        prompt="What is the topic of this text?",
    )

    embeddings = model.encode(
        inputs,
        task_metadata=task_metadata,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        prompt_type=None,
    )

    assert embeddings.shape == (3, 8)
    assert seen["n_texts"] == 3
    assert seen["instruction"] == "What is the topic of this text?"
