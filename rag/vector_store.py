# =============================================================================
# rag/vector_store.py
# FAISS vector store — stores, persists, and searches embedded chunks
# =============================================================================

import os
import json
import numpy as np
from typing import Optional
from backend.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    TOP_K_RETRIEVAL,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# VectorStore
# =============================================================================

class VectorStore:
    """
    FAISS-backed vector store for chunk retrieval.

    Stores:
        - FAISS index (binary)     → fast ANN search
        - Metadata JSON            → text + source for each vector

    Usage:
        store = VectorStore()
        store.add_chunks(embedded_chunks)   # chunks with "embedding" key
        results = store.search(query_vector, top_k=5)
        store.save()
        store.load()
    """

    def __init__(
        self,
        index_path:    str = FAISS_INDEX_PATH,
        metadata_path: str = FAISS_METADATA_PATH,
        dimension:     int = 384,
    ):
        self.index_path    = index_path
        self.metadata_path = metadata_path
        self.dimension     = dimension

        self._index    = None    # faiss.IndexFlatIP  (inner product = cosine on normalised vecs)
        self._metadata: list[dict] = []   # parallel list to index rows

        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.index_path),    exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

    # -------------------------------------------------------------------------
    # Index management
    # -------------------------------------------------------------------------

    def _get_or_create_index(self):
        """Lazy-create a FAISS IndexFlatIP on first use."""
        if self._index is not None:
            return self._index
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"FAISS index created | dim={self.dimension}")
            return self._index
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

    # -------------------------------------------------------------------------
    # Add chunks
    # -------------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Add embedded chunks to the FAISS index.

        Args:
            chunks: List of chunk dicts with "embedding" key (np.ndarray).
                    Chunks without "embedding" are skipped.

        Returns:
            Number of chunks successfully added.
        """
        index = self._get_or_create_index()

        valid_chunks = [c for c in chunks if "embedding" in c]
        if not valid_chunks:
            logger.warning("add_chunks: no embedded chunks to add.")
            return 0

        # Stack into matrix [N, dim]
        vectors = np.stack(
            [c["embedding"].astype(np.float32) for c in valid_chunks]
        )

        index.add(vectors)

        # Store metadata (everything except the embedding array)
        for chunk in valid_chunks:
            meta = {k: v for k, v in chunk.items() if k != "embedding"}
            self._metadata.append(meta)

        added = len(valid_chunks)
        logger.info(
            f"Added {added} chunks | total index size={index.ntotal}"
        )
        return added

    def clear(self):
        """Remove all vectors from the index and metadata."""
        import faiss
        self._index    = faiss.IndexFlatIP(self.dimension)
        self._metadata = []
        logger.info("VectorStore cleared.")

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict]:
        """
        Find the top-k most similar chunks to query_vector.

        Args:
            query_vector: np.ndarray [384], L2-normalised.
            top_k:        Number of results to return.

        Returns:
            List of chunk dicts sorted by similarity (descending),
            each with an added "score" key (cosine similarity 0-1).
        """
        index = self._get_or_create_index()

        if index.ntotal == 0:
            logger.warning("VectorStore is empty — no results.")
            return []

        query = query_vector.astype(np.float32).reshape(1, -1)
        top_k = min(top_k, index.ntotal)

        try:
            scores, indices = index.search(query, top_k)
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            result = dict(self._metadata[idx])
            result["score"] = float(score)
            results.append(result)

        logger.info(f"Search done | top_k={top_k} | results={len(results)}")
        return results

    def is_empty(self) -> bool:
        if self._index is None:
            return True
        return self._index.ntotal == 0

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self):
        """Persist FAISS index + metadata to disk."""
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Nothing to save — index is empty.")
            return

        try:
            import faiss
            faiss.write_index(self._index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            logger.info(
                f"VectorStore saved | "
                f"vectors={self._index.ntotal} | "
                f"path={self.index_path}"
            )
        except Exception as e:
            logger.error(f"VectorStore save error: {e}")

    def load(self) -> bool:
        """
        Load FAISS index + metadata from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(self.index_path):
            logger.info("No saved index found — starting fresh.")
            return False

        try:
            import faiss
            self._index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                f"VectorStore loaded | vectors={self._index.ntotal}"
            )
            return True
        except Exception as e:
            logger.error(f"VectorStore load error: {e}")
            return False