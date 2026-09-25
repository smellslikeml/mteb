"""Instruction-following embeddings obtained by answering the instruction.

Adapted from "Answer is All You Need: Instruction-following Text Embedding via
Answering the Question" (Peng et al., ACL 2024, https://arxiv.org/abs/2402.09642),
reference implementation at https://github.com/zhang-yu-wei/InBedder.

The core idea (InBedder): treat the user instruction as a *question* about the
input text, append answer (mask) slots to it, run a masked-language-model
encoder, and pool the hidden states at the answer positions after the LM head's
projection. Two texts that imply the same answer to the instruction end up with
similar representations, which is what makes the embedding instruction-aware.

This module ports the RoBERTa (encoder / fill-mask) variant of InBedder at full
fidelity. Only inference is implemented -- the fine-tuned weights are loaded from
the Hub, so no trainer, optimizer, or pretraining data pipeline is required here.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn.functional import gelu

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


def _batched(iterable: Sequence[str], n: int) -> Generator[list[str], None, None]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


class InstructionAnswerEncoder(AbsEncoder):
    """InBedder encoder: embed a text by answering the instruction about it.

    The instruction is framed as a question and followed by ``n_mask`` answer
    slots. The masked-language-model encoder fills those slots; the pooled,
    LM-head-projected hidden states at the answer positions form the embedding.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        n_mask: int = 3,
        max_length: int = 512,
        **kwargs: Any,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.n_mask = n_mask
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        masked_lm = AutoModelForMaskedLM.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        # InBedder pools the *answer* representation produced by the LM head's
        # projection (dense -> gelu -> layer_norm), stopping short of the vocab
        # decoder. We therefore keep the base encoder and the head's projection
        # layers rather than the full masked-LM forward.
        self.model = masked_lm.roberta.to(self.device).eval()
        self.dense = masked_lm.lm_head.dense.to(self.device)
        self.layer_norm = masked_lm.lm_head.layer_norm.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    def _answer_embeddings(self, texts: list[str], instruction: str) -> torch.Tensor:
        prompt = instruction + self.tokenizer.mask_token * self.n_mask
        inputs = self.tokenizer(
            texts,
            [prompt] * len(texts),
            padding=True,
            # Only truncate the input text so the appended answer slots always
            # survive -- this keeps exactly ``n_mask`` masks per example.
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        mask = inputs["input_ids"].eq(self.tokenizer.mask_token_id)
        with torch.no_grad():
            hidden_states = self.model(**inputs).last_hidden_state

        answers = self.layer_norm(gelu(self.dense(hidden_states[mask])))
        answers = answers.reshape(len(texts), self.n_mask, -1).mean(1)
        # Per-embedding standardization, following the reference implementation.
        answers = (answers - answers.mean(1, keepdim=True)) / answers.std(
            1, keepdim=True
        )
        return answers

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
        instruction = self.get_task_instruction(task_metadata, prompt_type) or ""
        sentences = [text for batch in inputs for text in batch["text"]]

        embeddings = []
        for batch in _batched(sentences, batch_size):
            embeddings.append(self._answer_embeddings(batch, instruction).cpu().numpy())
        return np.concatenate(embeddings, axis=0)


inbedder_roberta_large = ModelMeta(
    loader=InstructionAnswerEncoder,
    name="KomeijiForce/inbedder-roberta-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="51efa8290f751aaef1bc219f40f0b0bbec27cdab",
    release_date="2024-02-15",  # arXiv v1 submission
    n_parameters=355_000_000,
    n_embedding_parameters=51_471_360,  # vocab_size (50265) * hidden_size (1024)
    memory_usage_mb=1355,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/KomeijiForce/inbedder-roberta-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    adapted_from="FacebookAI/roberta-large",
    superseded_by=None,
    public_training_code="https://github.com/zhang-yu-wei/InBedder",
    public_training_data="https://huggingface.co/datasets/KomeijiForce/Inbedder-Pretrain-Data",
    training_datasets=None,
    citation="""@inproceedings{peng2024answer,
      title={Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
      author={Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
      year={2024}
    }""",
)
