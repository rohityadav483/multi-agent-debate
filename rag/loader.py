# =============================================================================
# rag/loader.py
# Loads documents from PDFs, plain text, and raw strings into chunks
# =============================================================================

import os
import re
from typing import Union
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, ALLOWED_FILE_TYPES
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DocumentLoader
# =============================================================================

class DocumentLoader:
    """
    Loads and chunks documents from multiple sources:
        - Uploaded PDF files  (via pypdf)
        - Plain .txt files
        - Raw text strings    (web search results, pasted content)

    All sources produce the same output: list of chunk dicts.

    Chunk dict schema:
        {
            "text":   str,          # The chunk content
            "source": str,          # Filename or "raw_text" or "web"
            "chunk":  int,          # Chunk index within this document
        }
    """

    def __init__(
        self,
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------------------------------------------------------------
    # Public loaders
    # -------------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> list[dict]:
        """
        Extract text from a PDF and split into chunks.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of chunk dicts.
        """
        try:
            import pypdf
        except ImportError:
            logger.error("pypdf not installed. Run: pip install pypdf")
            return []

        if not os.path.exists(file_path):
            logger.error(f"PDF not found: {file_path}")
            return []

        logger.info(f"Loading PDF: {file_path}")

        try:
            reader   = pypdf.PdfReader(file_path)
            raw_text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                raw_text += f"\n[Page {page_num + 1}]\n{page_text}"

            source = os.path.basename(file_path)
            chunks = self._split_text(raw_text, source=source)
            logger.info(f"PDF loaded | pages={len(reader.pages)} | chunks={len(chunks)}")
            return chunks

        except Exception as e:
            logger.error(f"PDF load error: {e}")
            return []

    def load_txt(self, file_path: str) -> list[dict]:
        """Load a plain text file and split into chunks."""
        if not os.path.exists(file_path):
            logger.error(f"TXT not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            source = os.path.basename(file_path)
            chunks = self._split_text(raw_text, source=source)
            logger.info(f"TXT loaded | chars={len(raw_text)} | chunks={len(chunks)}")
            return chunks
        except Exception as e:
            logger.error(f"TXT load error: {e}")
            return []

    def load_text(self, text: str, source: str = "raw_text") -> list[dict]:
        """
        Load a raw string directly — used for web search results,
        pasted content, or any in-memory text.

        Args:
            text:   The raw string to chunk.
            source: Label for this content (e.g. "web_search", "user_paste")
        """
        if not text or not text.strip():
            return []
        chunks = self._split_text(text, source=source)
        logger.info(f"Text loaded | source={source} | chunks={len(chunks)}")
        return chunks

    def load_file(self, file_path: str) -> list[dict]:
        """
        Auto-detect file type and load accordingly.
        Supported: .pdf, .txt
        """
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        if ext == "pdf":
            return self.load_pdf(file_path)
        elif ext == "txt":
            return self.load_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: .{ext}")
            return []

    def load_uploaded_file(self, uploaded_file) -> list[dict]:
        """
        Load a Streamlit UploadedFile object directly.
        Saves to a temp path then loads.

        Args:
            uploaded_file: st.file_uploader result object.
        """
        import tempfile

        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks = self.load_file(tmp_path)

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return chunks

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def _split_text(self, text: str, source: str) -> list[dict]:
        """
        Splits text into overlapping chunks using a sliding window.

        Strategy:
            1. Clean whitespace
            2. Split on sentence boundaries where possible
            3. Slide window of `chunk_size` chars with `chunk_overlap` overlap

        Returns:
            List of chunk dicts with text, source, chunk index.
        """
        # Clean
        text = self._clean_text(text)

        if len(text) <= self.chunk_size:
            return [{"text": text, "source": source, "chunk": 0}]

        chunks  = []
        start   = 0
        idx     = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for ". " or "\n" near the end of the window
                boundary = self._find_boundary(text, end)
                end = boundary if boundary else end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text":   chunk_text,
                    "source": source,
                    "chunk":  idx,
                })
                idx += 1

            # Slide forward with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        logger.debug(f"Split | source={source} | total_chars={len(text)} | chunks={len(chunks)}")
        return chunks

    def _find_boundary(self, text: str, pos: int, search_window: int = 100) -> int:
        """
        Search backwards from `pos` for a sentence/paragraph boundary.
        Returns the boundary position, or the original pos if none found.
        """
        search_start = max(0, pos - search_window)
        segment      = text[search_start:pos]

        # Look for paragraph break first, then sentence end
        for pattern in [r"\n\n", r"\. ", r"\.\n", r"! ", r"\? "]:
            match = None
            for m in re.finditer(pattern, segment):
                match = m
            if match:
                return search_start + match.end()

        return pos

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove non-printable chars except newlines and tabs
        text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
        return text.strip()

    def merge_chunks(self, *chunk_lists: list[dict]) -> list[dict]:
        """
        Merge chunks from multiple sources into a single list.
        Used to combine PDF + web search results before embedding.
        """
        merged = []
        for chunk_list in chunk_lists:
            merged.extend(chunk_list)
        logger.info(f"Merged {len(chunk_lists)} sources | total chunks={len(merged)}")
        return merged