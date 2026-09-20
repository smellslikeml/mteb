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


def test_answer_pooling_mean_and_zscore():
    """Answer tokens are mean-pooled per example then z-score normalized."""
    model = object.__new__(InBedderRobertaModel)
    model.n_mask = 3

    # 2 examples x 3 answer tokens x 4 dims.
    answer_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    pooled = model._pool_answers(answer_states, n_texts=2)

    assert pooled.shape == (2, 4)
    # z-scored rows are mean-centered with unit std.
    assert torch.allclose(pooled.mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(pooled.std(1), torch.ones(2), atol=1e-6)


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
