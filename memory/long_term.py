# =============================================================================
# memory/long_term.py
# Long-term memory — persists debate history to JSON + FAISS for recall
# =============================================================================

import os
import json
import glob
from typing import Optional
from memory.memory_schema import new_debate_record
from backend.config import (
    DEBATE_HISTORY_PATH,
    MAX_LONG_TERM_DEBATES,
    MEMORY_SIMILARITY_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# LongTermMemory
# =============================================================================

class LongTermMemory:
    """
    Persists completed debates to disk and supports semantic recall.

    Storage:
        data/debate_history/<debate_id>.json   — full debate record
        data/vector_index/memory.index         — FAISS index of summaries
        data/vector_index/memory_meta.json     — parallel metadata

    Key operations:
        save(debate_result)     — persist a completed debate
        search(query, top_k)    — find similar past debates by topic
        get_all()               — list all stored debates (most recent first)
        get_by_id(debate_id)    — retrieve a specific debate
        delete(debate_id)       — remove a debate record
    """

    def __init__(self):
        self._history_path  = DEBATE_HISTORY_PATH
        self._index_path    = "data/vector_index/memory.index"
        self._meta_path     = "data/vector_index/memory_meta.json"
        self._embedder      = None   # Lazy-loaded
        self._mem_store     = None   # Separate FAISS store for memory summaries
        os.makedirs(self._history_path, exist_ok=True)
        os.makedirs("data/vector_index", exist_ok=True)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save(
        self,
        topic:        str,
        domain:       str,
        agenda:       str,
        rounds:       list,
        scores:       dict,
        verdict:      dict,
        all_flags:    list,
        duration_sec: float = 0.0,
        total_tokens: int   = 0,
    ) -> str:
        """
        Save a completed debate to disk and index its summary.

        Returns:
            debate_id (str) — UUID of the saved record.
        """
        record = new_debate_record(
            topic=topic,
            domain=domain,
            agenda=agenda,
            rounds=rounds,
            scores=scores,
            verdict=verdict,
            all_flags=all_flags,
            duration_sec=duration_sec,
            total_tokens=total_tokens,
        )

        debate_id = record["debate_id"]
        file_path = os.path.join(self._history_path, f"{debate_id}.json")

        # Write JSON record
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.info(f"Debate saved | id={debate_id} | topic='{topic[:60]}'")
        except Exception as e:
            logger.error(f"Failed to save debate: {e}")
            return ""

        # Index the summary for semantic search
        self._index_summary(debate_id, record["summary"])

        # Rotate if over limit
        self._rotate_if_needed()

        return debate_id

    def save_from_result(self, result, total_tokens: int = 0) -> str:
        """
        Convenience — save directly from a DebateResult object.
        """
        return self.save(
            topic=result.topic,
            domain=result.domain,
            agenda=result.agenda,
            rounds=result.rounds,
            scores=result.scores,
            verdict=result.verdict,
            all_flags=result.all_flags,
            duration_sec=result.duration_sec,
            total_tokens=result.total_tokens or total_tokens,
        )

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def get_all(self, limit: int = 20) -> list[dict]:
        """
        Return all saved debates, most recent first.
        Returns lightweight summaries (not full round data).
        """
        files = sorted(
            glob.glob(os.path.join(self._history_path, "*.json")),
            key=os.path.getmtime,
            reverse=True,
        )[:limit]

        records = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Return lightweight summary — not full round data
                records.append({
                    "debate_id":    data.get("debate_id"),
                    "timestamp":    data.get("timestamp"),
                    "topic":        data.get("topic"),
                    "domain":       data.get("domain"),
                    "summary":      data.get("summary"),
                    "duration_sec": data.get("duration_sec"),
                    "total_tokens": data.get("total_tokens"),
                    "verdict":      data.get("verdict", {}),
                    "scores":       data.get("scores", {}),
                })
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")

        return records

    def get_by_id(self, debate_id: str) -> Optional[dict]:
        """Retrieve the full debate record by ID."""
        file_path = os.path.join(self._history_path, f"{debate_id}.json")
        if not os.path.exists(file_path):
            logger.warning(f"Debate not found: {debate_id}")
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading debate {debate_id}: {e}")
            return None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find past debates semantically similar to query.
        Uses FAISS index of debate summaries.

        Returns:
            List of lightweight debate summary dicts with similarity scores.
        """
        self._load_mem_store()

        if self._mem_store is None or self._mem_store.is_empty():
            logger.info("Memory search: no index available.")
            return []

        embedder     = self._get_embedder()
        query_vector = embedder.embed_query(query)
        if query_vector is None:
            return []

        raw_results = self._mem_store.search(
            query_vector,
            top_k=top_k,
        )

        # Filter by similarity threshold
        filtered = [
            r for r in raw_results
            if r.get("score", 0) >= MEMORY_SIMILARITY_THRESHOLD
        ]

        # Enrich with full lightweight record
        results = []
        for r in filtered:
            debate_id = r.get("debate_id")
            if not debate_id:
                continue
            record = self.get_by_id(debate_id)
            if record:
                results.append({
                    "debate_id":    debate_id,
                    "topic":        record.get("topic"),
                    "summary":      record.get("summary"),
                    "timestamp":    record.get("timestamp"),
                    "verdict":      record.get("verdict", {}),
                    "similarity":   round(r.get("score", 0), 3),
                })

        logger.info(f"Memory search | query='{query[:50]}' | results={len(results)}")
        return results

    def delete(self, debate_id: str) -> bool:
        """Delete a debate record by ID."""
        file_path = os.path.join(self._history_path, f"{debate_id}.json")
        if not os.path.exists(file_path):
            return False
        try:
            os.remove(file_path)
            logger.info(f"Deleted debate: {debate_id}")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def count(self) -> int:
        """Return total number of stored debates."""
        return len(glob.glob(os.path.join(self._history_path, "*.json")))

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from rag.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_mem_store(self):
        if self._mem_store is None:
            from rag.vector_store import VectorStore
            self._mem_store = VectorStore(
                index_path=self._index_path,
                metadata_path=self._meta_path,
            )
        return self._mem_store

    def _load_mem_store(self):
        store = self._get_mem_store()
        if store.is_empty():
            store.load()

    def _index_summary(self, debate_id: str, summary: str):
        """Embed and index a debate summary for future search."""
        try:
            embedder = self._get_embedder()
            store    = self._get_mem_store()

            vector   = embedder.embed_query(summary)
            if vector is None:
                return

            import numpy as np
            chunk = {
                "text":      summary,
                "source":    "memory",
                "chunk":     0,
                "debate_id": debate_id,
                "embedding": vector,
            }
            store.add_chunks([chunk])
            store.save()
            logger.info(f"Memory indexed | debate_id={debate_id}")
        except Exception as e:
            logger.warning(f"Memory index error (non-critical): {e}")

    def _rotate_if_needed(self):
        """Delete oldest debates if over MAX_LONG_TERM_DEBATES limit."""
        files = sorted(
            glob.glob(os.path.join(self._history_path, "*.json")),
            key=os.path.getmtime,
        )
        if len(files) > MAX_LONG_TERM_DEBATES:
            excess = files[:len(files) - MAX_LONG_TERM_DEBATES]
            for f in excess:
                try:
                    os.remove(f)
                    logger.info(f"Rotated old debate: {f}")
                except Exception:
                    pass