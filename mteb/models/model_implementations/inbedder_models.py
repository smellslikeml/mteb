"""InBedder: instruction-following embeddings via answering the question.

Adapted from "Answer is All You Need: Instruction-following Text Embedding via
Answering the Question" (Peng et al., 2024, https://arxiv.org/abs/2402.09642).

The paper's core idea is to treat the user instruction as a *question* about the
input text and to represent the text by the (implicit) *answer* the model would
give. For the RoBERTa checkpoint this is realised without any generation: the
text and instruction are laid out with the reference's ``### Input / ###
Instruction / ### Response`` template, a run of ``<mask>`` tokens (the answer
slot) is appended after ``### Response:``, and the embedding is the mean of the
backbone's last-layer hidden states at those mask positions (the reference's
``avg_gen`` aggregation). Texts that would elicit the same answer to a given
instruction land close together.

This mirrors the pipeline that produced the paper's MTEB numbers
(``configs/maskedlm_roberta-large-qa.json`` +
``lm_encoders_hf/maskedlm_encoder_hf.py`` in the reference), which uses the raw
mask-position hidden states -- no masked-LM head transform and no per-embedding
standardisation (those belong to the ``UseCase.ipynb`` model-card demo, a
different code path).

Implementation reference: https://github.com/zhang-yu-wei/InBedder and the
``KomeijiForce/inbedder-roberta-large`` model card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# Used when a task provides no instruction so InBedder still has a question to
# answer (the paper always conditions the representation on an instruction).
DEFAULT_QUESTION = "What is the meaning of this text?"

# The reference's MTEB prompt (configs/maskedlm_roberta-large-qa.json): the text
# and instruction are laid out with this template and the answer-slot masks are
# appended after "### Response:".
INBEDDER_PATTERN = (
    "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
)


def build_answer_prompts(
    texts: list[str], instruction: str, mask_token: str, n_mask: int
) -> list[str]:
    """Build the InBedder templated prompt with an answer slot for each text.

    Mirrors the reference pipeline: each text is placed in the ``### Input /
    ### Instruction / ### Response`` template with the instruction as the
    question, then ``n_mask`` mask tokens (whose hidden states carry the answer)
    are appended after ``### Response:``.

    Args:
        texts: The batch of input texts (one prompt per text).
        instruction: The task instruction, treated as a question. Falls back to
            ``DEFAULT_QUESTION`` when empty.
        mask_token: The tokenizer's mask token string.
        n_mask: How many mask tokens form the answer slot.

    Returns:
        A list of prompt strings, one per input text.
    """
    if n_mask < 1:
        raise ValueError("n_mask must be at least 1")
    question = instruction or DEFAULT_QUESTION
    answer_slot = mask_token * n_mask
    return [
        INBEDDER_PATTERN.replace("{input}", text).replace(
            "{instruction}", question
        )
        + answer_slot
        for text in texts
    ]


class InBedderModel(AbsEncoder):
    """Instruction-following encoder that answers the instruction with mask tokens."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        n_mask: int = 3,
        max_input_length: int = 512,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.n_mask = n_mask
        self.max_input_length = max_input_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        masked_lm = AutoModelForMaskedLM.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        # Truncate from the left (matching the reference) so the trailing answer
        # slot and instruction survive when the templated text is too long.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, truncation_side="left", **kwargs
        )
        # InBedder's MTEB (avg_gen) aggregation reads the raw backbone hidden
        # states at the mask positions, so we only need the encoder backbone --
        # not the masked-LM head.
        self.model = masked_lm.roberta.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        # The task instruction is the question InBedder answers. With no
        # instruction_template configured, this returns the raw task prompt.
        instruction = self.get_task_instruction(task_metadata, prompt_type)
        texts = [text for batch in inputs for text in batch["text"]]

        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            prompts = build_answer_prompts(
                batch, instruction, self.tokenizer.mask_token, self.n_mask
            )
            # A single templated string per text; left-truncation keeps the
            # trailing answer slot intact.
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)

            mask = encoded.input_ids.eq(self.tokenizer.mask_token_id)
            # avg_gen: mean of the raw last-layer hidden states over the mask
            # positions (no MLM-head transform, no standardisation).
            hidden = self.model(**encoded).last_hidden_state[mask]
            hidden = hidden.reshape(len(batch), self.n_mask, -1).mean(1)
            embeddings.append(hidden.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


_INBEDDER_CITATION = """@article{peng2024answer,
  title={Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
  author={Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
  journal={arXiv preprint arXiv:2402.09642},
  year={2024}
}"""

inbedder_roberta_large = ModelMeta(
    loader=InBedderModel,
    loader_kwargs=dict(n_mask=3),
    name="KomeijiForce/inbedder-roberta-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="51efa8290f751aaef1bc219f40f0b0bbec27cdab",
    release_date="2024-02-29",
    n_parameters=355_000_000,
    n_embedding_parameters=51_471_360,
    memory_usage_mb=1355,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/KomeijiForce/inbedder-roberta-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    adapted_from="FacebookAI/roberta-large",
    superseded_by=None,
    public_training_code="https://github.com/zhang-yu-wei/InBedder",
    public_training_data=None,
    training_datasets=None,
    citation=_INBEDDER_CITATION,
)
