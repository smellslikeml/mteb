"""Tests for the InBedder instruction-following embedder integration."""

import numpy as np
import torch

import mteb
from mteb.models.model_implementations.inbedder_models import (
    InBedderModel,
    build_answer_prompt,
    mean_pool_answer_tokens,
)

MODEL_NAME = "KomeijiForce/inbedder-roberta-large"


def test_inbedder_registered_and_wired():
    """The auto-discovery registry exposes the InBedder model via the public API."""
    meta = mteb.get_model_meta(MODEL_NAME)

    assert meta.name == MODEL_NAME
    assert meta.use_instructions is True
    # The registry wires our custom encoder as the loader (the call site).
    assert meta.loader is InBedderModel
    assert meta.loader_kwargs["num_answer_tokens"] == 3


def test_inbedder_present_in_model_metas():
    """The model shows up in the general model listing used by the leaderboard."""
    names = {m.name for m in mteb.get_model_metas()}
    assert MODEL_NAME in names


def test_build_answer_prompt_appends_answer_slot():
    prompt = build_answer_prompt(
        "The cat sat on the mat.",
        "What animal is described?",
        mask_token="<mask>",
        num_answer_tokens=3,
    )
    assert prompt.count("<mask>") == 3
    assert "What animal is described?" in prompt
    assert prompt.startswith("The cat sat on the mat.")


def test_build_answer_prompt_without_instruction():
    prompt = build_answer_prompt(
        "some text", "", mask_token="[M]", num_answer_tokens=2
    )
    assert prompt == "some text [M] [M]"


def test_mean_pool_answer_tokens_pools_only_mask_positions():
    mask_id = 7
    # batch of 1, seq len 3, hidden 2. Only positions 0 and 2 are answer tokens.
    hidden = torch.tensor([[[1.0, 1.0], [100.0, 100.0], [3.0, 3.0]]])
    input_ids = torch.tensor([[mask_id, 0, mask_id]])

    pooled = mean_pool_answer_tokens(hidden, input_ids, mask_id)

    # position 1 (the non-answer token) must not contribute to the mean.
    assert pooled.shape == (1, 2)
    np.testing.assert_allclose(pooled.numpy(), np.array([[2.0, 2.0]]))


def test_mean_pool_answer_tokens_no_answer_returns_zero():
    hidden = torch.tensor([[[5.0, 5.0], [6.0, 6.0]]])
    input_ids = torch.tensor([[0, 1]])

    pooled = mean_pool_answer_tokens(hidden, input_ids, mask_token_id=7)

    np.testing.assert_allclose(pooled.numpy(), np.zeros((1, 2)))
