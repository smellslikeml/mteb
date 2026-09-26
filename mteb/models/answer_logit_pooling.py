"""Instruction-following embeddings via answer-logit pooling.

Adapted from *Answer is All You Need: Instruction-following Text Embedding via
Answering the Question* (Peng et al., 2024, https://arxiv.org/abs/2402.09642),
the method released as **InBedder**.

The core idea implemented here: an instruction is treated as a *question* about
the input text. The model is asked to "answer" the question at a run of appended
``[MASK]`` slots, and the answer is read out as the mean of the masked-language-
model vocabulary logits over those slots (standardized per sample). Texts that
share the same answer to the instruction land close together, which is what makes
the embedding instruction-following.

Only the encoder-based (masked-LM) variant of the paper is implemented; the
decoder/generative variant and the paper's separate instruction-tuning and
evaluation harnesses are intentionally out of scope — this wraps a released
checkpoint so it can be evaluated through the standard mteb encode path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from .abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

logger = logging.getLogger(__name__)


def pool_answer_logits(
    logits: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pool masked-language-model logits over the answer (mask) positions.

    Implements the InBedder read-out: the model answers the instruction at the
    appended ``[MASK]`` slots, and the answer is the mean of the vocabulary
    logits over those slots, standardized per sample.

    Args:
        logits: ``(batch, seq_len, vocab)`` masked-LM logits from the model.
        answer_mask: ``(batch, seq_len)`` boolean tensor that is ``True`` at the
            appended answer/mask positions to pool over.
        eps: Numerical floor for the per-sample standard deviation.

    Returns:
        ``(batch, vocab)`` answer-distribution embeddings.
    """
    mask = answer_mask.unsqueeze(-1).to(logits.dtype)
    summed = (logits * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / counts
    mean = pooled.mean(dim=-1, keepdim=True)
    std = pooled.std(dim=-1, keepdim=True).clamp(min=eps)
    return (pooled - mean) / std


class InstructAnswerEncoderWrapper(AbsEncoder):
    """Encode text by pooling a masked-LM's answer to an instruction.

    The instruction (a question about the text) is concatenated with a run of
    ``[MASK]`` tokens, the masked-LM logits at those positions are pooled with
    :func:`pool_answer_logits`, and the result is the embedding. Suitable for
    ``use_instructions=True`` checkpoints trained the InBedder way (an
    ``AutoModelForMaskedLM`` head over an encoder such as RoBERTa).
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        *,
        n_answer_tokens: int = 3,
        max_seq_length: int = 512,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the masked-LM checkpoint and its tokenizer.

        Args:
            model_name: HuggingFace model id of an InBedder-style masked-LM.
            revision: The revision of the model to load.
            device: Device to load the model on. Defaults to CUDA when available.
            n_answer_tokens: Number of ``[MASK]`` answer slots to append.
            max_seq_length: Max tokenized length; the input text is truncated,
                the instruction and mask slots are preserved.
            model_kwargs: Extra keyword arguments for ``from_pretrained``.
            tokenizer_kwargs: Extra keyword arguments for the tokenizer.
            **kwargs: Absorbed for compatibility with the loader contract.
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **(tokenizer_kwargs or {})
        )
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"Tokenizer for '{model_name}' has no mask token; "
                "InstructAnswerEncoderWrapper requires a masked-LM checkpoint."
            )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, revision=revision, **(model_kwargs or {})
        )
        self.model.to(self.device)
        self.model.eval()
        self.n_answer_tokens = n_answer_tokens
        self.max_seq_length = max_seq_length

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode a batch of texts as answers to the task instruction.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: Metadata of the current task, used to resolve the
                instruction (the question) for these texts.
            hf_split: Split of the current task.
            hf_subset: Subset of the current task.
            prompt_type: The prompt type (query or document).
            **kwargs: Additional arguments; ``batch_size`` is honoured.

        Returns:
            ``(num_texts, vocab)`` array of answer-distribution embeddings.
        """
        instruction = self.get_instruction(task_metadata, prompt_type)
        texts = [text for batch in inputs for text in batch["text"]]
        logger.debug("Encoding %d texts with instruction: %r", len(texts), instruction)

        mask = self.tokenizer.mask_token
        answer_slots = mask * self.n_answer_tokens
        batch_size = int(kwargs.get("batch_size", 32) or 32)

        embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            prompts = [instruction + answer_slots for _ in chunk]
            encoded = self.tokenizer(
                chunk,
                prompts,
                padding=True,
                truncation="only_first",
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            answer_mask = encoded["input_ids"] == self.tokenizer.mask_token_id
            pooled = pool_answer_logits(logits, answer_mask)
            embeddings.append(pooled.cpu().float())

        return cast("Array", np.asarray(torch.cat(embeddings, dim=0)))
