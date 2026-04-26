# =============================================================================
# rag/embedder.py
# Generates vector embeddings for text chunks using SentenceTransformers
# Free, local, no API key needed
# =============================================================================

from typing import Optional
import numpy as np
from backend.config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Embedder
# =============================================================================

class Embedder:
    """
    Wraps SentenceTransformers to generate dense vector embeddings.

    Model: all-MiniLM-L6-v2
        - 384-dimensional embeddings
        - Fast, lightweight, free
        - Downloads once (~90MB), cached locally by HuggingFace

    Usage:
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)   # List of chunk dicts
        query_vec = embedder.embed_query("What is the risk?")
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model     = None   # Lazy load — don't load until first call

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunk dicts in-place.
        Adds an "embedding" key (np.ndarray, shape [384]) to each chunk.

        Args:
            chunks: List of {"text": ..., "source": ..., "chunk": ...}

        Returns:
            Same list with "embedding" added to each dict.
        """
        if not chunks:
            return chunks

        self._load_model()

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")

        try:
            # batch encoding — much faster than one-by-one
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-normalise for cosine similarity
            )

            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding

            logger.info(f"Embedding done | dim={embeddings.shape[1]}")
            return chunks

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            # Return chunks without embeddings — vector store will handle gracefully
            return chunks

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Embed a single query string for similarity search.

        Returns:
            np.ndarray of shape [384], L2-normalised.
            None if embedding fails.
        """
        if not query or not query.strip():
            return None

        self._load_model()

        try:
            vector = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vector[0]    # shape [384]
        except Exception as e:
            logger.error(f"Query embedding error: {e}")
            return None

    def embed_texts(self, texts: list[str]) -> Optional[np.ndarray]:
        """
        Embed a list of raw strings.
        Returns np.ndarray of shape [N, 384] or None on failure.
        Used internally by the vector store for batch operations.
        """
        if not texts:
            return None

        self._load_model()

        try:
            return self._model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return None

    @property
    def dimension(self) -> int:
        """Embedding dimension — 384 for all-MiniLM-L6-v2."""
        return 384