# =============================================================================
# rag/retriever.py
# Full RAG pipeline — loads → embeds → indexes → retrieves → formats context
# Single class the debate engine calls: retriever.get_context(query)
# =============================================================================

from typing import Optional
from rag.loader      import DocumentLoader
from rag.embedder    import Embedder
from rag.vector_store import VectorStore
from backend.config  import TOP_K_RETRIEVAL
from utils.logger    import get_logger

logger = get_logger(__name__)


# =============================================================================
# RAGRetriever
# =============================================================================

class RAGRetriever:
    """
    Coordinates the full retrieval pipeline:

        Documents / text
              ↓
        DocumentLoader  (chunk)
              ↓
        Embedder        (embed each chunk)
              ↓
        VectorStore     (FAISS index)
              ↓
        Query embed → similarity search → top-k chunks
              ↓
        Formatted context string → injected into agent prompts

    Usage:
        retriever = RAGRetriever()

        # Index documents once
        retriever.index_text("market report content...", source="market_report")
        retriever.index_pdf("/uploads/report.pdf")

        # Retrieve at debate time
        context = retriever.get_context("What is the ROI outlook?")
    """

    def __init__(self):
        self.loader  = DocumentLoader()
        self.embedder = Embedder()
        self.store   = VectorStore()
        self._indexed_sources: list[str] = []

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def index_text(self, text: str, source: str = "web_search") -> int:
        """
        Chunk, embed, and index a raw text string.

        Args:
            text:   The document content.
            source: Label (e.g. "web_search", "user_input", "wiki")

        Returns:
            Number of chunks added.
        """
        if not text or not text.strip():
            logger.warning(f"index_text: empty text for source={source}")
            return 0

        chunks = self.loader.load_text(text, source=source)
        return self._embed_and_add(chunks, source)

    def index_pdf(self, file_path: str) -> int:
        """
        Load, chunk, embed, and index a PDF file.

        Args:
            file_path: Path to PDF on disk.

        Returns:
            Number of chunks added.
        """
        chunks = self.loader.load_pdf(file_path)
        return self._embed_and_add(chunks, source=file_path)

    def index_uploaded_file(self, uploaded_file) -> int:
        """
        Index a Streamlit UploadedFile object directly.

        Returns:
            Number of chunks added.
        """
        chunks = self.loader.load_uploaded_file(uploaded_file)
        return self._embed_and_add(chunks, source=uploaded_file.name)

    def index_texts(self, texts: list[dict]) -> int:
        """
        Batch index multiple texts at once.

        Args:
            texts: List of {"text": str, "source": str}

        Returns:
            Total chunks added.
        """
        total = 0
        for item in texts:
            total += self.index_text(
                item.get("text",   ""),
                item.get("source", "batch"),
            )
        return total

    def _embed_and_add(self, chunks: list[dict], source: str) -> int:
        """Internal: embed chunks and add to vector store."""
        if not chunks:
            return 0

        embedded = self.embedder.embed_chunks(chunks)
        added    = self.store.add_chunks(embedded)

        if added > 0 and source not in self._indexed_sources:
            self._indexed_sources.append(source)

        logger.info(f"Indexed | source={source} | chunks={added}")
        return added

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def get_context(
        self,
        query:  str,
        top_k:  int = TOP_K_RETRIEVAL,
        min_score: float = 0.2,
    ) -> str:
        """
        Core retrieval method — called by the debate engine.

        Embeds the query, searches FAISS, formats top-k results
        into a single context string ready for agent prompt injection.

        Args:
            query:     The debate topic or agent's current question.
            top_k:     Max number of chunks to retrieve.
            min_score: Minimum cosine similarity to include a chunk.

        Returns:
            Formatted context string, or "" if nothing relevant found.

        Example output:
            [Source: market_report.pdf | Relevance: 0.87]
            "Renewable energy investment surged 25% in Q3 2024..."

            [Source: web_search | Relevance: 0.81]
            "Solar panel costs have dropped 89% over the past decade..."
        """
        if self.store.is_empty():
            logger.info("RAG: store is empty — no context retrieved.")
            return ""

        query_vector = self.embedder.embed_query(query)
        if query_vector is None:
            logger.warning("RAG: query embedding failed.")
            return ""

        results = self.store.search(query_vector, top_k=top_k)

        # Filter by minimum relevance score
        results = [r for r in results if r.get("score", 0) >= min_score]

        if not results:
            logger.info(f"RAG: no results above min_score={min_score}")
            return ""

        logger.info(f"RAG: retrieved {len(results)} chunks for query='{query[:60]}'")
        return self._format_context(results)

    def get_raw_results(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict]:
        """
        Return raw search results (list of chunk dicts with scores).
        Used by the Fact-Checker for precise claim verification.
        """
        if self.store.is_empty():
            return []

        query_vector = self.embedder.embed_query(query)
        if query_vector is None:
            return []

        return self.store.search(query_vector, top_k=top_k)

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def clear(self):
        """Clear the vector store and source list. Call between debates."""
        self.store.clear()
        self._indexed_sources = []
        logger.info("RAGRetriever cleared.")

    def save_index(self):
        """Persist FAISS index to disk for reuse across sessions."""
        self.store.save()

    def load_index(self) -> bool:
        """Load previously saved FAISS index from disk."""
        return self.store.load()

    @property
    def is_ready(self) -> bool:
        """True if the retriever has indexed content and is ready to search."""
        return not self.store.is_empty()

    @property
    def indexed_sources(self) -> list[str]:
        return list(self._indexed_sources)

    @property
    def chunk_count(self) -> int:
        return self.store.size

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    def _format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a readable context block
        for injection into agent system/user prompts.
        """
        lines = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown")
            score  = result.get("score",  0.0)
            text   = result.get("text",   "").strip()

            # Truncate very long chunks for prompt efficiency
            if len(text) > 600:
                text = text[:600] + "..."

            lines.append(
                f"[{i}] Source: {source} | Relevance: {score:.2f}\n{text}"
            )

        return "\n\n".join(lines)