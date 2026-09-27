"""Tests for the InBedder instruction-following embedder.

These exercise the wiring into MTEB's existing model registry (the model is
auto-discovered and resolvable through ``mteb.get_model_meta``) plus the pure
piece of the reference's recipe (templated answer-slot prompt construction),
without downloading the checkpoint.
"""

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.inbedder_models import (
    DEFAULT_QUESTION,
    INBEDDER_PATTERN,
    InBedderModel,
    build_answer_prompts,
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


def test_build_answer_prompts_templates_text_and_appends_mask_slot():
    prompts = build_answer_prompts(
        ["the cat sat"], "Represent the topic", mask_token="<mask>", n_mask=3
    )
    expected = (
        INBEDDER_PATTERN.replace("{input}", "the cat sat").replace(
            "{instruction}", "Represent the topic"
        )
        + "<mask><mask><mask>"
    )
    assert prompts == [expected]


def test_build_answer_prompts_falls_back_to_default_question():
    prompts = build_answer_prompts([""], "", mask_token="<mask>", n_mask=1)
    assert DEFAULT_QUESTION in prompts[0]
    assert prompts[0].endswith("<mask>")


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
