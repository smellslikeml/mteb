from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, TopRankedDocumentsType


logger = logging.getLogger(__name__)


def _as_document_list(embeddings: Array) -> list[NDArray[np.float32]]:
    """Coerce an encoder's multi-vector output into per-document token matrices.

    Late-interaction encoders emit one embedding per token, so a batch of
    documents is either a padded ``(n_docs, n_tokens, dim)`` array or an iterable
    of ragged ``(n_tokens, dim)`` arrays. Both are normalised here to a list of
    2-D ``float32`` arrays, one per document.

    Args:
        embeddings: Multi-vector embeddings produced by the encoder.

    Returns:
        A list of ``(n_tokens, dim)`` float32 arrays, one per document.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if isinstance(embeddings, np.ndarray) and embeddings.dtype != object:
        if embeddings.ndim == 3:
            return [np.asarray(doc, dtype=np.float32) for doc in embeddings]
        if embeddings.ndim == 2:
            return [embeddings.astype(np.float32)]
        raise ValueError(
            f"Expected 2-D or 3-D document embeddings, got shape {embeddings.shape}."
        )

    documents: list[NDArray[np.float32]] = []
    for doc in embeddings:
        if isinstance(doc, torch.Tensor):
            doc = doc.detach().cpu().numpy()  # noqa: PLW2901
        doc = np.asarray(doc, dtype=np.float32)  # noqa: PLW2901
        if doc.ndim == 1:
            doc = doc.reshape(1, -1)  # noqa: PLW2901
        documents.append(doc)
    return documents


def _max_sim_scores(
    query: NDArray[np.float32], documents: list[NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Compute MaxSim between one query and each document's token matrix.

    MaxSim (the ColBERT late-interaction score) is, for every query token, the
    maximum similarity to any document token, summed over query tokens. Working
    per document keeps ragged token counts exact without padding artefacts.

    Args:
        query: Query token matrix of shape ``(n_query_tokens, dim)``.
        documents: List of ``(n_doc_tokens, dim)`` document token matrices.

    Returns:
        A ``float32`` array of length ``len(documents)`` with the MaxSim score
        of the query against each document.
    """
    query_tensor = torch.as_tensor(query, dtype=torch.float32)
    scores = np.empty(len(documents), dtype=np.float32)
    for i, document in enumerate(documents):
        doc_tensor = torch.as_tensor(document, dtype=torch.float32)
        token_similarity = query_tensor @ doc_tensor.T
        scores[i] = token_similarity.max(dim=1).values.sum().item()
    return scores


class MultiVectorSearchIndex:
    """In-memory late-interaction (multi-vector / ColBERT-style) search backend.

    Implements `IndexEncoderSearchProtocol` for models scored by MaxSim
    (colpali/colqwen/pylate and friends), which the single-vector
    `FaissSearchIndex` cannot represent because it stores one vector per
    document rather than one per token.

    Retrieval is two-stage, following the shape of learned multi-vector indexes
    such as LEMUR (Learned Multi-Vector Retrieval, arXiv:2601.21853): a cheap
    token nearest-neighbour stage nominates candidate documents, then exact
    MaxSim scores only those candidates. LEMUR *learns* the candidate index; here
    that component is replaced by a parameter-free flat token-nearest-neighbour
    proxy so the backend has no training dependency. With ``n_candidates=None``
    (the default) the candidate stage is skipped and scoring is exact over the
    whole corpus, preserving MTEB result fidelity; setting ``n_candidates`` trades
    recall for latency the way the learned index does.

    Notes:
        - Stores every document token embedding in memory.
        - No FAISS dependency; scoring uses torch/numpy directly.
    """

    def __init__(
        self,
        model: EncoderProtocol,
        *,
        n_candidates: int | None = None,
        tokens_per_query: int | None = None,
    ) -> None:
        if model.mteb_model_meta.similarity_fn_name is not ScoringFunction.MAX_SIM:
            raise ValueError(
                "MultiVectorSearchIndex only supports MaxSim (late-interaction) "
                f"models, got {model.mteb_model_meta.similarity_fn_name}."
            )
        self.n_candidates = n_candidates
        self.tokens_per_query = tokens_per_query
        self.idxs: list[str] = []
        self._documents: list[NDArray[np.float32]] = []
        self._token_matrix: NDArray[np.float32] | None = None
        self._token_doc_ids: NDArray[np.int64] | None = None

    @classmethod
    def for_model(cls, model: EncoderProtocol) -> MultiVectorSearchIndex | None:
        """Build a backend when the model is scored by MaxSim, else return None.

        Lets a caller opt a late-interaction model into this backend without
        having to inspect the scoring function itself.

        Args:
            model: The encoder whose scoring function decides applicability.

        Returns:
            A `MultiVectorSearchIndex` for MaxSim models, otherwise ``None``.
        """
        if model.mteb_model_meta.similarity_fn_name is not ScoringFunction.MAX_SIM:
            return None
        return cls(model)

    def add_documents(self, embeddings: Array, idxs: list[str]) -> None:
        """Store per-token document embeddings and their IDs."""
        documents = _as_document_list(embeddings)
        self._documents.extend(documents)
        self.idxs.extend(idxs)
        # Invalidate the flat token index; it is rebuilt lazily when pruning.
        self._token_matrix = None
        self._token_doc_ids = None
        new_tokens = sum(doc.shape[0] for doc in documents)
        logger.info(
            f"Multi-vector index holds {len(self._documents)} documents "
            f"({new_tokens} new token vectors)."
        )

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Retrieve or rerank documents by MaxSim.

        Args:
            embeddings: Query embeddings; a padded ``(n_queries, n_tokens, dim)``
                array or an iterable of ragged per-query token matrices.
            top_k: Number of results to return per query.
            similarity_fn: Unused; MaxSim is computed internally so the backend
                stays faithful to late-interaction scoring. Kept for protocol
                compatibility with `IndexEncoderSearchProtocol`.
            top_ranked: Optional mapping of query_id -> candidate doc_ids for
                reranking mode.
            query_idx_to_id: Mapping of query index -> query_id, required when
                reranking.

        Returns:
            A tuple ``(top_k_values, top_k_indices)`` per query.
        """
        if not self._documents:
            raise ValueError("No documents indexed. Call add_documents() first.")

        queries = _as_document_list(embeddings)

        if top_ranked is not None:
            if query_idx_to_id is None:
                raise ValueError("query_idx_to_id must be provided when reranking.")
            return self._rerank(queries, top_k, top_ranked, query_idx_to_id)
        return self._full_search(queries, top_k)

    def _candidate_doc_indices(self, query: NDArray[np.float32]) -> NDArray[np.int64]:
        """Nominate candidate documents for a query via token nearest neighbours.

        Returns all documents when pruning is disabled (``n_candidates`` is None
        or covers the corpus), so exact scoring is the default. Otherwise each
        query token votes for its closest document tokens and the top scoring
        documents are kept -- the parameter-free stand-in for LEMUR's learned
        candidate index.

        Args:
            query: Query token matrix of shape ``(n_query_tokens, dim)``.

        Returns:
            An array of candidate document indices into ``self.idxs``.
        """
        n_docs = len(self._documents)
        n_candidates = self.n_candidates
        if n_candidates is None or n_candidates >= n_docs:
            return np.arange(n_docs, dtype=np.int64)

        if self._token_matrix is None or self._token_doc_ids is None:
            token_matrix, token_doc_ids = self._build_token_index()
        else:
            token_matrix, token_doc_ids = self._token_matrix, self._token_doc_ids

        token_similarity = np.asarray(query, dtype=np.float32) @ token_matrix.T
        probe_target = (
            self.tokens_per_query if self.tokens_per_query is not None else n_candidates
        )
        probe = min(probe_target, token_similarity.shape[1])

        approx_score: dict[int, float] = {}
        for query_token_sim in token_similarity:
            nearest = np.argpartition(-query_token_sim, probe - 1)[:probe]
            best_per_doc: dict[int, float] = {}
            for token_idx in nearest:
                doc = int(token_doc_ids[token_idx])
                sim = float(query_token_sim[token_idx])
                if sim > best_per_doc.get(doc, -np.inf):
                    best_per_doc[doc] = sim
            for doc, sim in best_per_doc.items():
                approx_score[doc] = approx_score.get(doc, 0.0) + sim

        ranked = sorted(approx_score, key=lambda doc: approx_score[doc], reverse=True)
        return np.asarray(ranked[:n_candidates], dtype=np.int64)

    def _build_token_index(self) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Flatten stored documents into a token matrix with per-token owners."""
        owners = [
            np.full(doc.shape[0], doc_idx, dtype=np.int64)
            for doc_idx, doc in enumerate(self._documents)
        ]
        token_matrix = np.vstack(self._documents).astype(np.float32)
        token_doc_ids = np.concatenate(owners).astype(np.int64)
        self._token_matrix = token_matrix
        self._token_doc_ids = token_doc_ids
        return token_matrix, token_doc_ids

    def _full_search(
        self, queries: list[NDArray[np.float32]], top_k: int
    ) -> tuple[list[list[float]], list[list[int]]]:
        scores_all: list[list[float]] = []
        idxs_all: list[list[int]] = []
        for query in queries:
            candidates = self._candidate_doc_indices(query)
            scores = _max_sim_scores(query, [self._documents[i] for i in candidates])
            order = np.argsort(-scores)[:top_k]
            scores_all.append(scores[order].tolist())
            idxs_all.append(candidates[order].tolist())
        return scores_all, idxs_all

    def _rerank(
        self,
        queries: list[NDArray[np.float32]],
        top_k: int,
        top_ranked: TopRankedDocumentsType,
        query_idx_to_id: dict[int, str],
    ) -> tuple[list[list[float]], list[list[int]]]:
        doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.idxs)}
        scores_all: list[list[float]] = []
        idxs_all: list[list[int]] = []
        for query_idx, query in enumerate(queries):
            query_id = query_idx_to_id[query_idx]
            ranked_ids = top_ranked.get(query_id)
            if not ranked_ids:
                msg = f"No top-ranked documents for query {query_id}"
                logger.warning(msg)
                warnings.warn(msg, stacklevel=2)
                scores_all.append([])
                idxs_all.append([])
                continue

            candidates = [self._documents[doc_id_to_idx[d]] for d in ranked_ids]
            scores = _max_sim_scores(query, candidates)
            order = np.argsort(-scores)[: min(top_k, len(ranked_ids))]
            scores_all.append(scores[order].tolist())
            idxs_all.append(order.tolist())
        return scores_all, idxs_all

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
        self.idxs = []
        self._documents = []
        self._token_matrix = None
        self._token_doc_ids = None
