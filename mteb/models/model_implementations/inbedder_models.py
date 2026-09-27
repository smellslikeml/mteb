"""InBedder: instruction-following embeddings via answering the question.

Adapted from "Answer is All You Need: Instruction-following Text Embedding via
Answering the Question" (Peng et al., 2024, https://arxiv.org/abs/2402.09642).

The paper's core idea is to treat the user instruction as a *question* about the
input text and to represent the text by the (implicit) *answer* the model would
give. For the RoBERTa checkpoint this is realised without any generation: the
instruction is appended to the text followed by a run of ``<mask>`` tokens (the
answer slot), the masked-LM head transform (dense -> gelu -> layer_norm) is
applied to the hidden states at those mask positions, and the per-mask vectors
are mean-pooled and standardised to form the embedding. Texts that would elicit
the same answer to a given instruction land close together.

Implementation reference: https://github.com/zhang-yu-wei/InBedder and the
``KomeijiForce/inbedder-roberta-large`` model card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn.functional import gelu

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# Used when a task provides no instruction so InBedder still has a question to
# answer (the paper always conditions the representation on an instruction).
DEFAULT_QUESTION = "What is the meaning of this text?"


def build_answer_prompts(
    instruction: str, batch_size: int, mask_token: str, n_mask: int
) -> list[str]:
    """Build the InBedder "answer slot" prompt for each item in a batch.

    Following the paper's recipe, the prompt is the instruction (the question)
    followed by ``n_mask`` mask tokens whose hidden states carry the answer.

    Args:
        instruction: The task instruction, treated as a question. Falls back to
            ``DEFAULT_QUESTION`` when empty.
        batch_size: Number of texts in the batch (one prompt per text).
        mask_token: The tokenizer's mask token string.
        n_mask: How many mask tokens form the answer slot.

    Returns:
        A list of ``batch_size`` identical prompt strings.
    """
    if n_mask < 1:
        raise ValueError("n_mask must be at least 1")
    question = instruction or DEFAULT_QUESTION
    answer_slot = mask_token * n_mask
    return [f"{question}{answer_slot}"] * batch_size


def standardize_answer_embeddings(
    embeddings: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Standardise each embedding across its dimensions (InBedder's final step).

    Mirrors the model card's ``(x - x.mean()) / x.std()`` per-embedding
    normalisation, with a small ``eps`` clamp on the standard deviation so a
    degenerate (constant) answer vector yields zeros instead of NaNs that would
    poison downstream cosine similarity.
    """
    mean = embeddings.mean(dim=1, keepdim=True)
    std = embeddings.std(dim=1, keepdim=True).clamp_min(eps)
    return (embeddings - mean) / std


class InBedderModel(AbsEncoder):
    """Instruction-following encoder that answers the instruction with mask tokens."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        n_mask: int = 3,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.n_mask = n_mask
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        masked_lm = AutoModelForMaskedLM.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        # InBedder reads the answer from the backbone hidden states passed through
        # the MLM head's transform (dense -> gelu -> layer_norm), not the vocab
        # projection, so we keep those three pieces and drop the decoder.
        self.model = masked_lm.roberta.to(self.device)
        self.dense = masked_lm.lm_head.dense.to(self.device)
        self.layer_norm = masked_lm.lm_head.layer_norm.to(self.device)
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
                instruction, len(batch), self.tokenizer.mask_token, self.n_mask
            )
            # text is segment A, the question+answer-slot is segment B. Truncate
            # only the text so the trailing mask tokens are never cut off.
            encoded = self.tokenizer(
                batch,
                prompts,
                padding=True,
                truncation="only_first",
                return_tensors="pt",
            ).to(self.device)

            mask = encoded.input_ids.eq(self.tokenizer.mask_token_id)
            hidden = self.model(**encoded).last_hidden_state[mask]
            hidden = self.layer_norm(gelu(self.dense(hidden)))
            hidden = hidden.reshape(len(batch), self.n_mask, -1).mean(1)
            hidden = standardize_answer_embeddings(hidden)
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
