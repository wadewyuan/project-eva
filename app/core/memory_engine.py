import aiosqlite
from datetime import datetime

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
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
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
            await conn.executescript(INIT_SQL)
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
            await conn.commit()
        finally:
            await conn.close()

    # ---------- Memories ----------

    async def add_memory(self, session_id: str | None, fact_text: str, category: str = "其他", importance_score: int = 5) -> None:
        now = datetime.utcnow().isoformat()
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO memories (session_id, fact_text, category, importance_score, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, fact_text, category, importance_score, now),
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
            await conn.execute("DELETE FROM memories")
            await conn.commit()
        finally:
            await conn.close()

    async def get_relevant_memories(self, session_id: str, query: str) -> list[str]:
        """MVP: return all memories for the session. Later upgrade to vector search."""
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT fact_text FROM memories WHERE session_id = ? OR session_id IS NULL ORDER BY created_at DESC LIMIT 10",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        finally:
            await conn.close()


memory_engine = MemoryEngine()
