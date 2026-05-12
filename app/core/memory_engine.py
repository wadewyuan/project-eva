import asyncio
import math
import struct
from datetime import datetime

import aiosqlite
import numpy as np

from app.core.embeddings import embedding_model
from config.settings import settings


INIT_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '新会话',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    fact_text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '其他',
    importance_score INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    embedding BLOB,
    source_context TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score);

CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_type TEXT NOT NULL,
    fact_key TEXT NOT NULL UNIQUE,
    fact_value TEXT NOT NULL,
    confidence INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_context TEXT
);

CREATE INDEX IF NOT EXISTS idx_profile_type ON user_profile(fact_type);
"""

FTS5_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    fact_text,
    content='memories',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, fact_text) VALUES (new.id, new.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact_text)
    VALUES ('delete', old.id, old.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact_text)
    VALUES ('delete', old.id, old.fact_text);
    INSERT INTO memories_fts(rowid, fact_text) VALUES (new.id, new.fact_text);
END;
"""


class MemoryEngine:
    def __init__(self, db_path: str = settings.db_path):
        self.db_path = db_path

    async def _connect(self):
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def init_db(self) -> None:
        conn = await self._connect()
        try:
            # 1. Base tables
            await conn.executescript(INIT_SQL)

            # 2. Migrate: add embedding column if missing
            # Drop FTS5 triggers first to avoid corruption during ALTER TABLE.
            cursor = await conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "embedding" not in columns:
                await conn.execute("DROP TRIGGER IF EXISTS memories_ai")
                await conn.execute("DROP TRIGGER IF EXISTS memories_ad")
                await conn.execute("DROP TRIGGER IF EXISTS memories_au")
                await conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
                await conn.commit()

            # 3. FTS5 virtual table and triggers
            await conn.executescript(FTS5_SQL)

            # 4. Backfill FTS index from existing memories if empty
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories_fts"
            )
            count = (await cursor.fetchone())[0]
            if count == 0:
                await conn.execute(
                    "INSERT INTO memories_fts(rowid, fact_text) "
                    "SELECT id, fact_text FROM memories"
                )
                await conn.commit()

            # 5. Verify FTS5 is in sync with memories; rebuild if counts mismatch
            cursor = await conn.execute("SELECT COUNT(*) FROM memories")
            mem_count = (await cursor.fetchone())[0]
            cursor = await conn.execute("SELECT COUNT(*) FROM memories_fts")
            fts_count = (await cursor.fetchone())[0]
            if mem_count != fts_count:
                await conn.execute("DROP TABLE IF EXISTS memories_fts")
                await conn.executescript(FTS5_SQL)
                await conn.execute(
                    "INSERT INTO memories_fts(rowid, fact_text) "
                    "SELECT id, fact_text FROM memories"
                )
                await conn.commit()

            # 6. Migrate: add source_context column if missing
            cursor = await conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "source_context" not in columns:
                await conn.execute("ALTER TABLE memories ADD COLUMN source_context TEXT")
                await conn.commit()

            # 7. Backfill embeddings for memories that don't have them
            cursor = await conn.execute(
                "SELECT id, fact_text FROM memories WHERE embedding IS NULL"
            )
            rows = await cursor.fetchall()
            for row in rows:
                emb = await asyncio.to_thread(embedding_model.encode_one, row[1])
                blob = embedding_model.vector_to_blob(emb)
                await conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (blob, row[0]),
                )
            if rows:
                await conn.commit()
        finally:
            await conn.close()

    # ---------- Sessions ----------

    async def create_session(self, session_id: str, title: str = "新会话") -> None:
        now = datetime.utcnow().isoformat()
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_session_time(self, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        conn = await self._connect()
        try:
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_sessions(self):
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) as message_count "
                "FROM sessions s LEFT JOIN messages m ON s.id = m.session_id "
                "GROUP BY s.id ORDER BY s.updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def get_session_messages(self, session_id: str, limit: int | None = None):
        sql = (
            "SELECT id, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id"
        )
        params = (session_id,)
        if limit:
            sql += " DESC LIMIT ?"
            params = (session_id, limit)
            # Need to re-order after DESC LIMIT
            # Easier: just fetch all then slice in Python for MVP
            sql = (
                "SELECT id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id"
            )
            params = (session_id,)

        conn = await self._connect()
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                result = [dict(row) for row in rows]
                if limit and len(result) > limit:
                    result = result[-limit:]
                return result
        finally:
            await conn.close()

    # ---------- Messages ----------

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.utcnow().isoformat()
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def clear_session_messages(self, session_id: str) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
            # NOTE: user_profile is global, not tied to a single session
            await conn.commit()
        finally:
            await conn.close()

    # ---------- User Profile ----------

    async def upsert_profile(
        self,
        fact_type: str,
        fact_key: str,
        fact_value: str,
        confidence: int = 5,
        source_context: str | None = None,
    ) -> None:
        """Insert a user profile fact, or update if the same fact_key exists."""
        now = datetime.utcnow().isoformat()
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO user_profile (fact_type, fact_key, fact_value, confidence, created_at, updated_at, source_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at,
                    source_context = excluded.source_context
                """,
                (fact_type, fact_key, fact_value, confidence, now, now, source_context),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_user_profile(self) -> list[dict]:
        """Return all user profile facts ordered by fact_type."""
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT * FROM user_profile ORDER BY fact_type, updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def delete_profile(self, profile_id: int) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM user_profile WHERE id = ?", (profile_id,))
            await conn.commit()
        finally:
            await conn.close()

    async def clear_all_profiles(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM user_profile")
            await conn.commit()
        finally:
            await conn.close()

    # ---------- Memories ----------

    async def _is_duplicate(
        self,
        session_id: str | None,
        new_embedding: np.ndarray,
        threshold: float = 0.88,
    ) -> bool:
        """Check if a very similar memory already exists in the session."""
        conn = await self._connect()
        try:
            if session_id:
                async with conn.execute(
                    "SELECT embedding FROM memories WHERE session_id = ? ORDER BY created_at DESC LIMIT 50",
                    (session_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with conn.execute(
                    "SELECT embedding FROM memories ORDER BY created_at DESC LIMIT 50",
                ) as cursor:
                    rows = await cursor.fetchall()

            new_vec = np.array(new_embedding, dtype=np.float32)
            new_norm = np.linalg.norm(new_vec)
            if new_norm == 0:
                return False

            for row in rows:
                if row[0] is None:
                    continue
                mem_vec = np.frombuffer(row[0], dtype=np.float32)
                if len(mem_vec) != len(new_vec):
                    continue
                mem_norm = np.linalg.norm(mem_vec)
                if mem_norm == 0:
                    continue
                cosine = float(np.dot(new_vec, mem_vec) / (new_norm * mem_norm))
                if cosine > threshold:
                    return True
            return False
        finally:
            await conn.close()

    async def add_memory(
        self,
        session_id: str | None,
        fact_text: str,
        category: str = "其他",
        importance_score: int = 5,
        source_context: str | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        # Compute embedding in a thread pool to avoid blocking the event loop
        embedding = await asyncio.to_thread(embedding_model.encode_one, fact_text)

        # Skip if a nearly identical memory already exists (echo chamber guard)
        if await self._is_duplicate(session_id, embedding):
            return

        blob = embedding_model.vector_to_blob(embedding)

        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO memories (session_id, fact_text, category, importance_score, created_at, embedding, source_context) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, fact_text, category, importance_score, now, blob, source_context),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_memories(self, session_id: str | None = None):
        conn = await self._connect()
        try:
            if session_id:
                async with conn.execute(
                    "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC"
                ) as cursor:
                    rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def delete_memory(self, memory_id: int) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await conn.commit()
        finally:
            await conn.close()

    async def clear_all_memories(self) -> None:
        conn = await self._connect()
        try:
            # Drop triggers first to avoid FTS5 bulk-delete corruption,
            # then delete memories, then recreate FTS5 + triggers.
            await conn.execute("DROP TRIGGER IF EXISTS memories_ai")
            await conn.execute("DROP TRIGGER IF EXISTS memories_ad")
            await conn.execute("DROP TRIGGER IF EXISTS memories_au")
            await conn.execute("DROP TABLE IF EXISTS memories_fts")
            await conn.execute("DELETE FROM memories")
            await conn.executescript(FTS5_SQL)
            await conn.commit()
        finally:
            await conn.close()

    @staticmethod
    def _keyword_overlap(query: str, text: str) -> float:
        """Simple character-level overlap for Chinese text fallback."""
        query_chars = set(query)
        text_chars = set(text)
        if not query_chars:
            return 0.0
        overlap = len(query_chars & text_chars)
        return overlap / len(query_chars)

    def _score_memories(
        self,
        memories: list[dict],
        query: str,
        fts_ranks: dict[int, float],
        query_embedding: list[float] | None,
        current_session_id: str,
    ) -> list[tuple[float, dict]]:
        """Score memories by relevance (embedding > FTS > keyword), importance, and recency."""
        now = datetime.utcnow()
        scored: list[tuple[float, dict]] = []
        query_chars = set(query) if query else set()
        query_embedding_np = np.array(query_embedding, dtype=np.float32) if query_embedding is not None else None

        for mem in memories:
            mem_id = mem["id"]

            # 1. Relevance (0-1): embedding > FTS > keyword overlap
            relevance = 0.0
            if query_embedding_np is not None and mem.get("embedding") is not None:
                mem_vec = np.frombuffer(mem["embedding"], dtype=np.float32)
                if len(mem_vec) == len(query_embedding_np):
                    cosine = float(np.dot(query_embedding_np, mem_vec))
                    # Normalize from [-1, 1] to [0, 1]
                    relevance = (cosine + 1.0) / 2.0

            if relevance == 0.0:
                if mem_id in fts_ranks:
                    rank = fts_ranks[mem_id]
                    # bm25 rank: smaller is better; normalize via inverse
                    relevance = min(1.0, max(0.0, 1.0 / (1.0 + abs(rank))))
                else:
                    text = mem.get("fact_text", "")
                    relevance = self._keyword_overlap(query, text)

            # 2. Importance (0-1)
            importance = mem.get("importance_score", 5) / 10.0

            # 3. Recency (0-1), half-life of 30 days
            try:
                created = datetime.fromisoformat(mem["created_at"])
                age_days = (now - created).total_seconds() / 86400.0
                recency = math.exp(-age_days / 30.0)
            except (ValueError, TypeError):
                recency = 0.5

            # 4. Session boost
            session_boost = 1.2 if mem.get("session_id") == current_session_id else 1.0

            score = (relevance * 0.4 + importance * 0.3 + recency * 0.3) * session_boost
            scored.append((score, mem))

        return scored

    @staticmethod
    def _format_memory(mem: dict) -> str:
        """Format a memory for the LLM context, including its timestamp and source turn."""
        fact = mem.get("fact_text", "")
        created = mem.get("created_at", "")
        source = mem.get("source_context", "")
        if source:
            source_short = source.replace("\n", " | ")[:100]
            if len(source.replace("\n", " | ")) > 100:
                source_short += "..."
            fact = f"{fact}（来源：{source_short}）"
        if created:
            try:
                dt = datetime.fromisoformat(created)
                date_str = dt.strftime("%Y-%m-%d")
                return f"[{date_str}] {fact}"
            except (ValueError, TypeError):
                pass
        return fact

    async def get_relevant_memories(
        self, session_id: str, query: str, limit: int | None = None
    ) -> list[str]:
        """Return the most relevant memories for a query using hybrid scoring."""
        if limit is None:
            limit = settings.max_memories_in_context

        conn = await self._connect()
        try:
            fts_ids: set[int] = set()
            fts_ranks: dict[int, float] = {}

            # Step 1: Compute query embedding (CPU-bound, run in thread pool)
            cleaned_query = query.strip() if query else ""
            query_embedding: list[float] | None = None
            if cleaned_query:
                try:
                    query_embedding = await asyncio.to_thread(
                        embedding_model.encode_one, cleaned_query
                    )
                except Exception:
                    pass

            # Step 2: FTS keyword search as supplemental signal
            if cleaned_query:
                try:
                    escaped = cleaned_query.replace('"', '""')
                    async with conn.execute(
                        "SELECT rowid, rank FROM memories_fts WHERE fact_text MATCH ? ORDER BY rank LIMIT ?",
                        (escaped, settings.memory_candidate_pool),
                    ) as cursor:
                        for row in await cursor.fetchall():
                            fts_ids.add(row[0])
                            fts_ranks[row[0]] = row[1]
                except Exception:
                    pass

            # Step 3: Build candidate pool = FTS matches + recent memories
            all_memories: dict[int, dict] = {}

            if fts_ids:
                placeholders = ",".join("?" * len(fts_ids))
                async with conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})",
                    tuple(fts_ids),
                ) as cursor:
                    for row in await cursor.fetchall():
                        all_memories[row["id"]] = dict(row)

            needed = settings.memory_candidate_pool - len(all_memories)
            if needed > 0:
                existing_ids = tuple(all_memories.keys()) if all_memories else (-1,)
                placeholders = ",".join("?" * len(existing_ids))
                async with conn.execute(
                    f"SELECT * FROM memories WHERE id NOT IN ({placeholders}) ORDER BY created_at DESC LIMIT ?",
                    (*existing_ids, needed),
                ) as cursor:
                    for row in await cursor.fetchall():
                        all_memories[row["id"]] = dict(row)

            candidates = list(all_memories.values())

            # Step 4: Score and rank
            scored = self._score_memories(
                candidates, cleaned_query, fts_ranks, query_embedding, session_id
            )
            scored.sort(key=lambda x: x[0], reverse=True)

            return [self._format_memory(mem) for _, mem in scored[:limit]]
        finally:
            await conn.close()


memory_engine = MemoryEngine()
