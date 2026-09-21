from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

#: Supported dimensionality-reduction methods.
#:
#: - ``"truncate"``: keep the first ``n_components`` dimensions. Cheap and
#:   parameter-free; ideal for Matryoshka-trained models whose leading
#:   dimensions carry the most information.
#: - ``"gaussian"``: project onto ``n_components`` random directions drawn
#:   from a fixed-seed Gaussian (a Johnson-Lindenstrauss random projection).
#:   The projection matrix is a pure function of ``(input_dim, n_components,
#:   seed)``, so queries and corpus are always mapped with the same basis.
REDUCTION_METHODS = ("truncate", "gaussian")


class DimensionalityReductionWrapper:
    """Wraps a model to reduce the dimensionality of its embeddings before scoring.

    This is the dimensionality-reduction counterpart to
    [`CompressionWrapper`][mteb.models.compression_wrappers.compression_wrapper.CompressionWrapper]:
    where the compression wrapper lowers the per-dimension bit-width, this
    wrapper lowers the *number* of dimensions. Both trade retrieval quality for
    embedding-storage savings and can be evaluated directly through
    ``mteb.evaluate`` on any task.

    Only data-independent projections are used so that queries and corpus are
    always reduced with an identical, order-independent basis (the per-call
    ``encode`` contract does not expose a shared fitting corpus). ``"truncate"``
    keeps the leading dimensions; ``"gaussian"`` applies a fixed-seed
    Johnson-Lindenstrauss random projection.

    Examples:
        >>> import mteb
        >>> from mteb.models import DimensionalityReductionWrapper
        >>> model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> reduced_model = DimensionalityReductionWrapper(model, n_components=128)
        >>> task = mteb.get_task("NanoArguAnaRetrieval")
        >>> mteb.evaluate(reduced_model, task)
    """

    def __init__(
        self,
        model: EncoderProtocol,
        n_components: int,
        method: str = "truncate",
        seed: int = 42,
    ) -> None:
        """Instantiates the wrapper with an embedding model and a target dimensionality.

        Args:
            model: The model whose embeddings should be reduced.
            n_components: The number of dimensions to keep. Must be positive.
            method: The reduction method, one of ``"truncate"`` or ``"gaussian"``.
            seed: Seed for the Gaussian projection matrix (ignored for ``"truncate"``).
        """
        if n_components <= 0:
            raise ValueError(
                f"n_components must be a positive integer, but got {n_components}."
            )
        if method not in REDUCTION_METHODS:
            raise ValueError(
                f"Unknown reduction method '{method}'. Supported methods are {REDUCTION_METHODS}."
            )
        self.model = model
        self.n_components = n_components
        self.method = method
        self.seed = seed
        # Lazily built and cached on the first encode call; a pure function of
        # (input_dim, n_components, seed), so reuse across query/corpus is safe.
        self._projection: torch.Tensor | None = None

        meta = model.mteb_model_meta
        exp_kwargs = dict(meta.experiment_kwargs) if meta.experiment_kwargs else {}
        exp_kwargs["reduction_method"] = method
        exp_kwargs["n_components"] = n_components
        model.mteb_model_meta = meta.model_copy(  # type: ignore[misc]
            update={
                "embed_dim": n_components,
                "experiment_kwargs": exp_kwargs,
            }
        )
        logger.info(
            f"Initialized DimensionalityReductionWrapper (method={method}, n_components={n_components})."
        )

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """Return wrapped model meta data."""
        return self.model.mteb_model_meta

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
        """Encodes the given inputs, then reduces the embedding dimensionality.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task.
            hf_split: Split of current task.
            hf_subset: Subset of current task.
            prompt_type: The name type of prompt. (query or passage)
            batch_size: Batch size.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input reduced to ``n_components`` dimensions, an array of
            shape (Number of sentences) x (n_components).
        """
        embeddings = self.model.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings).float()

        logger.info(
            f"Reducing embeddings from {embeddings.shape[-1]} to {self.n_components} dimensions "
            f"using '{self.method}'."
        )
        return self._reduce_embeddings(embeddings)

    def _reduce_embeddings(self, embeddings: torch.Tensor) -> Array:
        """Projects full-dimensional embeddings down to ``n_components`` dimensions.

        Args:
            embeddings: The embeddings to reduce, shape (n_sentences, input_dim).

        Returns:
            The reduced embeddings, shape (n_sentences, n_components).
        """
        input_dim = embeddings.shape[-1]
        if input_dim < self.n_components:
            raise ValueError(
                f"Cannot reduce embeddings of dimension {input_dim} to {self.n_components} dimensions; "
                f"n_components must not exceed the embedding dimension."
            )

        if self.method == "truncate":
            return embeddings[..., : self.n_components].contiguous()

        projection = self._get_projection(input_dim, embeddings.dtype)
        return embeddings.to(projection.dtype) @ projection

    def _get_projection(self, input_dim: int, dtype: torch.dtype) -> torch.Tensor:
        """Builds (or returns the cached) Gaussian random-projection matrix.

        The matrix is deterministic given ``(input_dim, n_components, seed)`` and
        is scaled by ``1 / sqrt(n_components)`` so that expected squared distances
        are preserved (Johnson-Lindenstrauss).

        Args:
            input_dim: Dimensionality of the incoming embeddings.
            dtype: Floating dtype to build the matrix in.

        Returns:
            The projection matrix of shape (input_dim, n_components).
        """
        if self._projection is None:
            generator = torch.Generator().manual_seed(self.seed)
            matrix = torch.randn(
                input_dim, self.n_components, generator=generator, dtype=torch.float32
            )
            matrix /= self.n_components**0.5
            self._projection = matrix.to(dtype)
        return self._projection

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity] for more details."""
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity_pairwise] for more details."""
        return self.model.similarity_pairwise(embeddings1, embeddings2)
