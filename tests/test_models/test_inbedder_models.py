"""Tests for the InBedder instruction-following embedder.

These exercise the wiring into MTEB's existing model registry (the model is
auto-discovered and resolvable through ``mteb.get_model_meta``) plus the two
pure pieces of the paper's recipe (answer-slot prompt construction and the
final per-embedding standardisation), without downloading the checkpoint.
"""

import numpy as np
import torch

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.inbedder_models import (
    DEFAULT_QUESTION,
    InBedderModel,
    build_answer_prompts,
    standardize_answer_embeddings,
)
from mteb.types import PromptType
from tests.mock_tasks import MockRetrievalTask

MODEL_NAME = "KomeijiForce/inbedder-roberta-large"


def test_inbedder_registered_in_registry():
    """The new module is auto-discovered and resolvable via the public API."""
    meta = mteb.get_model_meta(MODEL_NAME)
    assert meta.loader is InBedderModel
    assert meta.loader_kwargs["n_mask"] == 3
    assert meta.use_instructions is True
    # Included in the registry-wide listing used across MTEB.
    assert MODEL_NAME in {m.name for m in mteb.get_model_metas()}


def test_build_answer_prompts_appends_mask_slot():
    prompts = build_answer_prompts(
        "Represent the topic", batch_size=2, mask_token="<mask>", n_mask=3
    )
    assert prompts == ["Represent the topic<mask><mask><mask>"] * 2


def test_build_answer_prompts_falls_back_to_default_question():
    prompts = build_answer_prompts("", batch_size=1, mask_token="<mask>", n_mask=1)
    assert prompts == [f"{DEFAULT_QUESTION}<mask>"]


def test_standardize_answer_embeddings_is_zero_mean_unit_std():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = standardize_answer_embeddings(x)
    assert torch.allclose(out.mean(dim=1), torch.zeros(1), atol=1e-6)
    # torch.std uses the unbiased estimator; the standardised row matches it.
    assert torch.allclose(out.std(dim=1), torch.ones(1), atol=1e-5)


def test_standardize_answer_embeddings_handles_constant_vector():
    """A degenerate answer must not produce NaNs that poison cosine similarity."""
    out = standardize_answer_embeddings(torch.full((1, 4), 5.0))
    assert not torch.isnan(out).any()
    assert np.allclose(out.numpy(), 0.0)


def test_instruction_is_used_verbatim_as_question():
    """InBedder answers the raw task instruction (no instruction template)."""
    # Build an instance without loading weights; get_task_instruction comes from
    # the shared AbsEncoder base class exactly as it does at inference time.
    encoder = InBedderModel.__new__(InBedderModel)
    assert isinstance(encoder, AbsEncoder)
    assert encoder.instruction_template is None

    meta = MockRetrievalTask.metadata
    meta.prompt = {"query": "Given a question, retrieve relevant passages"}
    instruction = encoder.get_task_instruction(meta, PromptType.query)
    assert instruction == "Given a question, retrieve relevant passages"
