"""
L3 Long-term Memory Manager — SQLite-based persistent memory layer
for FinGent multi-agent stock analysis system.

Tables:
- analysis_history: Record of every analysis run (votes, decisions, feedback)
- agent_stats: Per-agent performance statistics and dynamic weights
- user_rules: Rules extracted from user feedback, injected into agent prompts
"""

import os
import re
import json
import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # fallback to sync

DB_DIR = Path(__file__).parent.parent / "memory"
DB_PATH = os.getenv("FINGENT_MEMORY_DB", str(DB_DIR / "l3_memory.db"))

# Weight adjustment constants
WEIGHT_AGREE_DELTA = 0.05
WEIGHT_DISAGREE_DELTA = -0.10
WEIGHT_SELF_CORRECT_DELTA = -0.15
WEIGHT_MAX = 1.5
WEIGHT_MIN = 0.3
WEIGHT_RULE_PROMOTE_THRESHOLD = 3

# 结果验证（前向检验）相关
PRIMARY_HORIZON = 20          # 用于判定"已验证"与更新客观权重的交易日数
MIN_VERIFIED_SAMPLES = 30     # 每个 Agent 至少这么多已验证样本才更新客观权重

# 存储版本（PRAGMA user_version）。只追加新表/列，不改已有字段语义。
SCHEMA_VERSION = 4

# 待验证记录的最大验证尝试次数：数据源失败/退市等永久性失败不应阻塞后续记录
# （review P2：oldest-200 无上限跳过会造成队头阻塞）。达到上限后不再重试。
MAX_VERIFY_ATTEMPTS = 10

_V2_SQL = """
CREATE TABLE IF NOT EXISTS agent_votes (
    analysis_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    vote TEXT,
    target_pct REAL,
    confidence REAL,
    PRIMARY KEY (analysis_id, agent_name)
);
CREATE INDEX IF NOT EXISTS idx_av_agent ON agent_votes(agent_name);

CREATE TABLE IF NOT EXISTS agent_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    vote TEXT,
    horizon_days INTEGER NOT NULL,
    entry_date TEXT,
    exit_date TEXT,
    entry_price REAL,
    exit_price REAL,
    stock_return REAL,
    bench_return REAL,
    excess_return REAL,
    correct INTEGER,
    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (analysis_id, agent_name, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_ao_agent ON agent_outcomes(agent_name);
"""


# Agent 标准名（与 LLM/registry.py 的注册名一致）。历史数据里存在类名/旧写法（TECHNICAL_NERD、RiskManager 等），
# 必须统一，否则同一个 Agent 的权重和命中率会被拆成多份。
CANONICAL_AGENT_NAMES = ("technical-nerd", "morefit", "risk-manager", "sentiment-analyzer", "tiktok-auntie")
_CANON_KEY = {re.sub(r"[^a-z0-9]", "", n): n for n in CANONICAL_AGENT_NAMES}


def canonical_agent_name(name: str) -> str:
    """忽略大小写、连字符与下划线后映射到标准名；未知名字原样返回。"""
    key = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    return _CANON_KEY.get(key, name)


def canonical_votes(votes: Dict[str, str]) -> Dict[str, str]:
    return {canonical_agent_name(k): v for k, v in (votes or {}).items()}


def _utc_cutoff(days: int) -> str:
    """与 SQLite CURRENT_TIMESTAMP 同格式（UTC，空格分隔）的截止时间字符串。"""
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _canonicalize_legacy_names(conn: sqlite3.Connection) -> None:
    """把历史 agent_votes / agent_outcomes / agent_stats / analysis_history.agent_votes 改写为标准名。"""
    for table in ("agent_votes", "agent_outcomes"):
        names = [r[0] for r in conn.execute(f"SELECT DISTINCT agent_name FROM {table}").fetchall()]
        for old in names:
            new = canonical_agent_name(old)
            if new != old:
                conn.execute(f"UPDATE OR IGNORE {table} SET agent_name = ? WHERE agent_name = ?", (new, old))
                conn.execute(f"DELETE FROM {table} WHERE agent_name = ?", (old,))  # 与已有标准名冲突的残留
    for rid, raw in conn.execute("SELECT id, agent_votes FROM analysis_history WHERE agent_votes IS NOT NULL").fetchall():
        try:
            votes = json.loads(raw or "{}")
        except json.JSONDecodeError:
            continue
        canon = canonical_votes(votes)
        if canon != votes:
            conn.execute("UPDATE analysis_history SET agent_votes = ? WHERE id = ?",
                         (json.dumps(canon, ensure_ascii=False), rid))


def _vote_rows(analysis_id: int, agent_votes: Dict[str, str],
               agent_details: Optional[Dict[str, Dict[str, Any]]]) -> List[tuple]:
    details = {canonical_agent_name(k): v for k, v in (agent_details or {}).items()}
    rows = []
    for agent, vote in canonical_votes(agent_votes).items():
        d = details.get(agent) or {}
        rows.append((analysis_id, agent, vote, d.get("target_position_pct"), d.get("confidence")))
    return rows


@dataclass
class AnalysisRecord:
    id: int
    user_id: str
    session_id: str
    stock_symbol: str
    query: str
    agent_votes: Dict[str, str]
    final_decision: str
    reasoning_summary: str
    user_feedback: Optional[str]
    market_outcome: Optional[str]
    created_at: str


@dataclass
class AgentStat:
    agent_name: str
    user_id: str
    total_calls: int
    agrees: int
    disagrees: int
    self_corrections: int
    current_weight: float
    updated_at: str


@dataclass
class UserRule:
    id: int
    user_id: str
    agent_name: str
    rule_text: str
    trigger_count: int
    source: str
    active: bool
    created_at: str


class MemoryManager:
    """
    Manages L3 long-term memory (SQLite).
    All public methods are async when aiosqlite is available,
    otherwise sync (for compatibility).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Initialize synchronously on creation
        self._init_db()

    def _connection(self):
        """Synchronous connection (for init and sync ops)."""
        return sqlite3.connect(self.db_path)

    def _aconnection(self):
        """Asynchronous connection factory (returns a context manager)."""
        if aiosqlite is None:
            raise RuntimeError("aiosqlite not installed; cannot use async mode")
        return aiosqlite.connect(self.db_path)

    def _init_db(self):
        """Create tables if not exist."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    session_id TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    query TEXT,
                    agent_votes TEXT,          -- JSON: {"TECHNICAL_NERD": "BUY", ...}
                    final_decision TEXT,       -- BUY | HOLD | SELL | STRONG_BUY | ...
                    reasoning_summary TEXT,
                    user_feedback TEXT,        -- agree | disagree | correction | null
                    market_outcome TEXT,       -- verified | failed | pending
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ah_user_stock ON analysis_history(user_id, stock_symbol);
                CREATE INDEX IF NOT EXISTS idx_ah_session ON analysis_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_ah_created ON analysis_history(created_at);

                CREATE TABLE IF NOT EXISTS agent_stats (
                    agent_name TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'global',
                    total_calls INTEGER DEFAULT 0,
                    agrees INTEGER DEFAULT 0,
                    disagrees INTEGER DEFAULT 0,
                    self_corrections INTEGER DEFAULT 0,
                    current_weight REAL DEFAULT 1.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_name, user_id)
                );

                CREATE TABLE IF NOT EXISTS user_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    agent_name TEXT NOT NULL,  -- "ALL" for global
                    rule_text TEXT NOT NULL,
                    trigger_count INTEGER DEFAULT 1,
                    source TEXT DEFAULT 'explicit_feedback',  -- explicit_feedback | self_mined
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ur_user_agent ON user_rules(user_id, agent_name);
                CREATE INDEX IF NOT EXISTS idx_ur_active ON user_rules(active);
                """
            )
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """按 PRAGMA user_version 逐版迁移，幂等。"""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            conn.executescript(_V2_SQL)
            # 回填：把历史 JSON 投票拆成行表，便于按 Agent 查询
            rows = conn.execute(
                "SELECT id, agent_votes FROM analysis_history WHERE agent_votes IS NOT NULL"
            ).fetchall()
            for rid, raw in rows:
                try:
                    votes = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    continue
                conn.executemany(
                    "INSERT OR IGNORE INTO agent_votes (analysis_id, agent_name, vote) VALUES (?, ?, ?)",
                    [(rid, a, v) for a, v in votes.items()],
                )
            conn.execute("PRAGMA user_version = 2")
        if version < 3:
            _canonicalize_legacy_names(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if version < 4:
            # 验证尝试计数：防止永久无法验证的记录（退市/格式异常/数据源长期失败）
            # 在 oldest-拉取中无限占据名额，阻塞新记录（review P2 已修）
            cols = {r[1] for r in conn.execute("PRAGMA table_info(analysis_history)").fetchall()}
            if "verify_attempts" not in cols:
                conn.execute(
                    "ALTER TABLE analysis_history ADD COLUMN verify_attempts INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _sync_conn(self):
        """用完即关的同步连接（给 asyncio.to_thread 调用的内部方法使用）。"""
        return closing(sqlite3.connect(self.db_path))

    # ------------------------------------------------------------------
    # Analysis History
    # ------------------------------------------------------------------

    async def record_analysis(
        self,
        session_id: str,
        stock_symbol: str,
        query: str,
        agent_votes: Dict[str, str],
        final_decision: str,
        reasoning_summary: str = "",
        user_id: str = "anonymous",
        agent_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> int:
        """Record a new analysis entry. Returns the row id."""
        if aiosqlite is None:
            return self._sync_record_analysis(
                session_id, stock_symbol, query, agent_votes,
                final_decision, reasoning_summary, user_id, agent_details,
            )
        async with self._aconnection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO analysis_history
                (user_id, session_id, stock_symbol, query, agent_votes,
                 final_decision, reasoning_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    stock_symbol.upper(),
                    query,
                    json.dumps(canonical_votes(agent_votes), ensure_ascii=False),
                    final_decision,
                    reasoning_summary,
                ),
            )
            rid = cursor.lastrowid
            await conn.executemany(
                "INSERT OR REPLACE INTO agent_votes (analysis_id, agent_name, vote, target_pct, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                _vote_rows(rid, agent_votes, agent_details),
            )
            await conn.commit()
            return rid

    def _sync_record_analysis(
        self,
        session_id: str,
        stock_symbol: str,
        query: str,
        agent_votes: Dict[str, str],
        final_decision: str,
        reasoning_summary: str = "",
        user_id: str = "anonymous",
        agent_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history
                (user_id, session_id, stock_symbol, query, agent_votes,
                 final_decision, reasoning_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    stock_symbol.upper(),
                    query,
                    json.dumps(canonical_votes(agent_votes), ensure_ascii=False),
                    final_decision,
                    reasoning_summary,
                ),
            )
            rid = cursor.lastrowid
            conn.executemany(
                "INSERT OR REPLACE INTO agent_votes (analysis_id, agent_name, vote, target_pct, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                _vote_rows(rid, agent_votes, agent_details),
            )
            conn.commit()
            return rid

    async def update_feedback(
        self,
        analysis_id: int,
        feedback: str,  # agree | disagree | correction
    ):
        """Update user feedback for a specific analysis record."""
        if aiosqlite is None:
            return self._sync_update_feedback(analysis_id, feedback)
        async with self._aconnection() as conn:
            await conn.execute(
                "UPDATE analysis_history SET user_feedback = ? WHERE id = ?",
                (feedback, analysis_id),
            )
            await conn.commit()

    def _sync_update_feedback(self, analysis_id: int, feedback: str):
        with self._connection() as conn:
            conn.execute(
                "UPDATE analysis_history SET user_feedback = ? WHERE id = ?",
                (feedback, analysis_id),
            )
            conn.commit()

    async def get_analysis_record(
        self,
        session_id: str,
        stock_symbol: Optional[str] = None,
    ) -> Optional[AnalysisRecord]:
        """Get the most recent analysis record for a session and optional stock."""
        if aiosqlite is None:
            return self._sync_get_analysis_record(session_id, stock_symbol)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            if stock_symbol:
                cursor = await conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE session_id = ? AND stock_symbol = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id, stock_symbol.upper()),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,)
                )
            row = await cursor.fetchone()
            return self._row_to_analysis(row) if row else None

    def _sync_get_analysis_record(
        self,
        session_id: str,
        stock_symbol: Optional[str] = None,
    ) -> Optional[AnalysisRecord]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if stock_symbol:
                cursor = conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE session_id = ? AND stock_symbol = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id, stock_symbol.upper()),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,)
                )
            row = cursor.fetchone()
            return self._row_to_analysis(row) if row else None

    async def record_feedback(
        self,
        session_id: str,
        stock_symbol: str,
        feedback: str,
    ) -> Optional[int]:
        """Find the matching history entry and attach user feedback to it."""
        record = await self.get_analysis_record(session_id, stock_symbol)
        if not record:
            return None
        await self.update_feedback(record.id, feedback)
        return record.id

    def _sync_record_feedback(
        self,
        session_id: str,
        stock_symbol: str,
        feedback: str,
    ) -> Optional[int]:
        record = self._sync_get_analysis_record(session_id, stock_symbol)
        if not record:
            return None
        self._sync_update_feedback(record.id, feedback)
        return record.id

    async def get_recent_analyses(
        self,
        user_id: str,
        stock_symbol: Optional[str] = None,
        limit: int = 5,
    ) -> List[AnalysisRecord]:
        """Get recent analysis records for a user (optionally filtered by stock)."""
        if aiosqlite is None:
            return self._sync_get_recent_analyses(user_id, stock_symbol, limit)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            if stock_symbol:
                cursor = await conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE user_id = ? AND stock_symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, stock_symbol.upper(), limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
            rows = await cursor.fetchall()
            return [self._row_to_analysis(r) for r in rows]

    def _sync_get_recent_analyses(
        self, user_id: str, stock_symbol: Optional[str] = None, limit: int = 5
    ) -> List[AnalysisRecord]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if stock_symbol:
                cursor = conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE user_id = ? AND stock_symbol = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, stock_symbol.upper(), limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM analysis_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, limit),
                )
            rows = cursor.fetchall()
            return [self._row_to_analysis(r) for r in rows]

    def _row_to_analysis(self, row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            stock_symbol=row["stock_symbol"],
            query=row["query"] or "",
            agent_votes=json.loads(row["agent_votes"] or "{}"),
            final_decision=row["final_decision"] or "",
            reasoning_summary=row["reasoning_summary"] or "",
            user_feedback=row["user_feedback"],
            market_outcome=row["market_outcome"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Agent Stats & Weights
    # ------------------------------------------------------------------

    async def get_agent_weights(
        self,
        user_id: str = "global",
        agent_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Return current weight for each agent. Missing agents default to 1.0."""
        if agent_names:
            # 有名字被改写成标准名时，才按标准名转发一次（转发后的名字不会再改写，避免递归）
            canon = {n: canonical_agent_name(n) for n in agent_names}
            if any(k != v for k, v in canon.items()):
                loaded = await self.get_agent_weights(user_id=user_id, agent_names=sorted(set(canon.values())))
                return {orig: loaded.get(c, 1.0) for orig, c in canon.items()}
        if aiosqlite is None:
            return self._sync_get_agent_weights(user_id, agent_names)
        async with self._aconnection() as conn:
            if agent_names:
                placeholders = ",".join("?" * len(agent_names))
                cursor = await conn.execute(
                    f"""
                    SELECT agent_name, current_weight FROM agent_stats
                    WHERE user_id = ? AND agent_name IN ({placeholders})
                    """,
                    (user_id, *agent_names),
                )
            else:
                cursor = await conn.execute(
                    "SELECT agent_name, current_weight FROM agent_stats WHERE user_id = ?",
                    (user_id,),
                )
            rows = await cursor.fetchall()
            weights = {r[0]: r[1] for r in rows}
            # Ensure all requested agents have a weight
            if agent_names:
                for name in agent_names:
                    if name not in weights:
                        weights[name] = 1.0
            return weights

    def _sync_get_agent_weights(
        self, user_id: str = "global", agent_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        with self._connection() as conn:
            if agent_names:
                placeholders = ",".join("?" * len(agent_names))
                cursor = conn.execute(
                    f"""
                    SELECT agent_name, current_weight FROM agent_stats
                    WHERE user_id = ? AND agent_name IN ({placeholders})
                    """,
                    (user_id, *agent_names),
                )
            else:
                cursor = conn.execute(
                    "SELECT agent_name, current_weight FROM agent_stats WHERE user_id = ?",
                    (user_id,),
                )
            rows = cursor.fetchall()
            weights = {r[0]: r[1] for r in rows}
            if agent_names:
                for name in agent_names:
                    if name not in weights:
                        weights[name] = 1.0
            return weights

    async def adjust_weight(
        self,
        agent_name: str,
        delta: float,
        user_id: str = "global",
    ) -> float:
        """Adjust an agent's weight by delta. Returns new weight."""
        agent_name = canonical_agent_name(agent_name)
        if aiosqlite is None:
            return self._sync_adjust_weight(agent_name, delta, user_id)
        async with self._aconnection() as conn:
            # Ensure row exists
            await conn.execute(
                """
                INSERT OR IGNORE INTO agent_stats (agent_name, user_id, current_weight)
                VALUES (?, ?, 1.0)
                """,
                (agent_name, user_id),
            )
            # Update
            await conn.execute(
                """
                UPDATE agent_stats
                SET current_weight = MIN(?, MAX(?, current_weight + ?)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND user_id = ?
                """,
                (WEIGHT_MAX, WEIGHT_MIN, delta, agent_name, user_id),
            )
            await conn.commit()
            cursor = await conn.execute(
                "SELECT current_weight FROM agent_stats WHERE agent_name = ? AND user_id = ?",
                (agent_name, user_id),
            )
            row = await cursor.fetchone()
            return row[0] if row else 1.0

    def _sync_adjust_weight(self, agent_name: str, delta: float, user_id: str = "global") -> float:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_stats (agent_name, user_id, current_weight)
                VALUES (?, ?, 1.0)
                """,
                (agent_name, user_id),
            )
            conn.execute(
                """
                UPDATE agent_stats
                SET current_weight = MIN(?, MAX(?, current_weight + ?)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND user_id = ?
                """,
                (WEIGHT_MAX, WEIGHT_MIN, delta, agent_name, user_id),
            )
            conn.commit()
            cursor = conn.execute(
                "SELECT current_weight FROM agent_stats WHERE agent_name = ? AND user_id = ?",
                (agent_name, user_id),
            )
            row = cursor.fetchone()
            return row[0] if row else 1.0

    async def record_agent_outcome(
        self,
        agent_name: str,
        user_feedback: Optional[str],  # agree | disagree | correction
        self_corrected: bool = False,
        user_id: str = "global",
    ) -> float:
        """Record the outcome of an agent run and update stats/weights accordingly.

        Returns 调整后的 current_weight（调用方无需再另调 adjust_weight，
        否则同一 delta 会被叠加两次——见 review P1 修复记录）。
        """
        agent_name = canonical_agent_name(agent_name)
        delta = 0.0
        agree_inc = 0
        disagree_inc = 0
        self_corr_inc = 0

        if user_feedback == "agree":
            delta = WEIGHT_AGREE_DELTA
            agree_inc = 1
        elif user_feedback in ("disagree", "correction"):
            delta = WEIGHT_DISAGREE_DELTA
            disagree_inc = 1

        if self_corrected:
            delta += WEIGHT_SELF_CORRECT_DELTA
            self_corr_inc = 1

        if aiosqlite is None:
            return self._sync_record_agent_outcome(
                agent_name, delta, agree_inc, disagree_inc, self_corr_inc, user_id
            )
        async with self._aconnection() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO agent_stats
                (agent_name, user_id, total_calls, agrees, disagrees, self_corrections, current_weight)
                VALUES (?, ?, 0, 0, 0, 0, 1.0)
                """,
                (agent_name, user_id),
            )
            await conn.execute(
                """
                UPDATE agent_stats
                SET total_calls = total_calls + 1,
                    agrees = agrees + ?,
                    disagrees = disagrees + ?,
                    self_corrections = self_corrections + ?,
                    current_weight = MIN(?, MAX(?, current_weight + ?)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND user_id = ?
                """,
                (agree_inc, disagree_inc, self_corr_inc,
                 WEIGHT_MAX, WEIGHT_MIN, delta, agent_name, user_id),
            )
            await conn.commit()
            cursor = await conn.execute(
                "SELECT current_weight FROM agent_stats WHERE agent_name = ? AND user_id = ?",
                (agent_name, user_id),
            )
            row = await cursor.fetchone()
            return row[0] if row else 1.0

    def _sync_record_agent_outcome(
        self,
        agent_name: str,
        delta: float,
        agree_inc: int,
        disagree_inc: int,
        self_corr_inc: int,
        user_id: str,
    ):
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_stats
                (agent_name, user_id, total_calls, agrees, disagrees, self_corrections, current_weight)
                VALUES (?, ?, 0, 0, 0, 0, 1.0)
                """,
                (agent_name, user_id),
            )
            conn.execute(
                """
                UPDATE agent_stats
                SET total_calls = total_calls + 1,
                    agrees = agrees + ?,
                    disagrees = disagrees + ?,
                    self_corrections = self_corrections + ?,
                    current_weight = MIN(?, MAX(?, current_weight + ?)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND user_id = ?
                """,
                (agree_inc, disagree_inc, self_corr_inc,
                 WEIGHT_MAX, WEIGHT_MIN, delta, agent_name, user_id),
            )
            conn.commit()
            cursor = conn.execute(
                "SELECT current_weight FROM agent_stats WHERE agent_name = ? AND user_id = ?",
                (agent_name, user_id),
            )
            row = cursor.fetchone()
            return row[0] if row else 1.0

    # ------------------------------------------------------------------
    # User Rules
    # ------------------------------------------------------------------

    async def add_or_update_rule(
        self,
        rule_text: str,
        agent_name: str = "ALL",
        user_id: str = "anonymous",
        source: str = "explicit_feedback",
    ) -> UserRule:
        """Add a new rule or increment trigger_count if identical rule exists."""
        agent_name = canonical_agent_name(agent_name)
        if aiosqlite is None:
            return self._sync_add_or_update_rule(rule_text, agent_name, user_id, source)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            # Check for existing identical active rule
            cursor = await conn.execute(
                """
                SELECT id, trigger_count FROM user_rules
                WHERE user_id = ? AND agent_name = ? AND rule_text = ? AND active = 1
                """,
                (user_id, agent_name, rule_text),
            )
            row = await cursor.fetchone()
            if row:
                new_count = row["trigger_count"] + 1
                await conn.execute(
                    "UPDATE user_rules SET trigger_count = ? WHERE id = ?",
                    (new_count, row["id"]),
                )
                await conn.commit()
                return UserRule(
                    id=row["id"],
                    user_id=user_id,
                    agent_name=agent_name,
                    rule_text=rule_text,
                    trigger_count=new_count,
                    source=source,
                    active=True,
                    created_at="",
                )
            else:
                cursor = await conn.execute(
                    """
                    INSERT INTO user_rules (user_id, agent_name, rule_text, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, agent_name, rule_text, source),
                )
                await conn.commit()
                return UserRule(
                    id=cursor.lastrowid,
                    user_id=user_id,
                    agent_name=agent_name,
                    rule_text=rule_text,
                    trigger_count=1,
                    source=source,
                    active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),  # 与 SQLite CURRENT_TIMESTAMP(UTC) 口径一致
                )

    def _sync_add_or_update_rule(
        self, rule_text: str, agent_name: str, user_id: str, source: str
    ) -> UserRule:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, trigger_count FROM user_rules
                WHERE user_id = ? AND agent_name = ? AND rule_text = ? AND active = 1
                """,
                (user_id, agent_name, rule_text),
            )
            row = cursor.fetchone()
            if row:
                new_count = row["trigger_count"] + 1
                conn.execute(
                    "UPDATE user_rules SET trigger_count = ? WHERE id = ?",
                    (new_count, row["id"]),
                )
                conn.commit()
                return UserRule(
                    id=row["id"],
                    user_id=user_id,
                    agent_name=agent_name,
                    rule_text=rule_text,
                    trigger_count=new_count,
                    source=source,
                    active=True,
                    created_at="",
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO user_rules (user_id, agent_name, rule_text, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, agent_name, rule_text, source),
                )
                conn.commit()
                return UserRule(
                    id=cursor.lastrowid,
                    user_id=user_id,
                    agent_name=agent_name,
                    rule_text=rule_text,
                    trigger_count=1,
                    source=source,
                    active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),  # 与 SQLite CURRENT_TIMESTAMP(UTC) 口径一致
                )

    async def get_rules(
        self,
        user_id: str,
        agent_name: Optional[str] = None,
        min_trigger_count: int = 1,
    ) -> List[UserRule]:
        """Get active rules for a user, optionally filtered by agent."""
        # 注入时传入的是类名（如 Morefit），存储的是标准名（morefit），必须统一
        agent_name = canonical_agent_name(agent_name) if agent_name else agent_name
        if aiosqlite is None:
            return self._sync_get_rules(user_id, agent_name, min_trigger_count)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            if agent_name:
                cursor = await conn.execute(
                    """
                    SELECT * FROM user_rules
                    WHERE user_id = ? AND (agent_name = ? OR agent_name = 'ALL')
                      AND active = 1 AND trigger_count >= ?
                    ORDER BY trigger_count DESC, created_at DESC
                    """,
                    (user_id, agent_name, min_trigger_count),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM user_rules
                    WHERE user_id = ? AND active = 1 AND trigger_count >= ?
                    ORDER BY trigger_count DESC, created_at DESC
                    """,
                    (user_id, min_trigger_count),
                )
            rows = await cursor.fetchall()
            return [self._row_to_rule(r) for r in rows]

    def _sync_get_rules(
        self, user_id: str, agent_name: Optional[str] = None, min_trigger_count: int = 1
    ) -> List[UserRule]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if agent_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM user_rules
                    WHERE user_id = ? AND (agent_name = ? OR agent_name = 'ALL')
                      AND active = 1 AND trigger_count >= ?
                    ORDER BY trigger_count DESC, created_at DESC
                    """,
                    (user_id, agent_name, min_trigger_count),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM user_rules
                    WHERE user_id = ? AND active = 1 AND trigger_count >= ?
                    ORDER BY trigger_count DESC, created_at DESC
                    """,
                    (user_id, min_trigger_count),
                )
            rows = cursor.fetchall()
            return [self._row_to_rule(r) for r in rows]

    def _row_to_rule(self, row: sqlite3.Row) -> UserRule:
        return UserRule(
            id=row["id"],
            user_id=row["user_id"],
            agent_name=row["agent_name"],
            rule_text=row["rule_text"],
            trigger_count=row["trigger_count"],
            source=row["source"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Memory Injection Helpers
    # ------------------------------------------------------------------

    async def build_memory_context(
        self,
        user_id: str,
        stock_symbol: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Build a memory context string for injection into an agent's system prompt.
        Combines: stock history + agent stats + user rules.
        """
        parts = []

        # 1. Recent analysis history for this stock
        if stock_symbol:
            recent = await self.get_recent_analyses(user_id, stock_symbol, limit=3)
            if recent:
                parts.append(f"【{stock_symbol} 历史分析】")
                for r in recent:
                    fb = f" (用户反馈: {r.user_feedback})" if r.user_feedback else ""
                    parts.append(
                        f"- {r.created_at[:10]} 综合决策: {r.final_decision}{fb}"
                    )

        # 2. Agent-specific stats
        if agent_name:
            weights = await self.resolve_weights(user_id, [agent_name])
            weight = weights.get(agent_name, 1.0)
            stats = await self._get_agent_stat(user_id, agent_name)
            if stats:
                parts.append(f"【你的表现统计】")
                parts.append(
                    f"- 当前权重: {weight:.2f} (基准 1.0)"
                )
                if stats.disagrees > 0:
                    parts.append(
                        f"- 用户否定次数: {stats.disagrees}，同意次数: {stats.agrees}"
                    )
                if stats.self_corrections > 0:
                    parts.append(
                        f"- 自我修正次数: {stats.self_corrections}"
                    )

        # 3. User rules for this agent
        rules = await self.get_rules(user_id, agent_name, min_trigger_count=1)
        if rules:
            parts.append("【用户特定规则】")
            for rule in rules:
                tag = " (强规则)" if rule.trigger_count >= WEIGHT_RULE_PROMOTE_THRESHOLD else ""
                parts.append(f"- {rule.rule_text}{tag}")

        return "\n".join(parts) if parts else ""

    async def _get_agent_stat(self, user_id: str, agent_name: str) -> Optional[AgentStat]:
        agent_name = canonical_agent_name(agent_name)
        if aiosqlite is None:
            return self._sync_get_agent_stat(user_id, agent_name)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT * FROM agent_stats WHERE user_id = ? AND agent_name = ?",
                (user_id, agent_name),
            )
            row = await cursor.fetchone()
            if row:
                return AgentStat(
                    agent_name=row["agent_name"],
                    user_id=row["user_id"],
                    total_calls=row["total_calls"],
                    agrees=row["agrees"],
                    disagrees=row["disagrees"],
                    self_corrections=row["self_corrections"],
                    current_weight=row["current_weight"],
                    updated_at=row["updated_at"],
                )
            return None

    def _sync_get_agent_stat(self, user_id: str, agent_name: str) -> Optional[AgentStat]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM agent_stats WHERE user_id = ? AND agent_name = ?",
                (user_id, agent_name),
            )
            row = cursor.fetchone()
            if row:
                return AgentStat(
                    agent_name=row["agent_name"],
                    user_id=row["user_id"],
                    total_calls=row["total_calls"],
                    agrees=row["agrees"],
                    disagrees=row["disagrees"],
                    self_corrections=row["self_corrections"],
                    current_weight=row["current_weight"],
                    updated_at=row["updated_at"],
                )
            return None

    # ------------------------------------------------------------------
    # Self-Improve: Consistency Check
    # ------------------------------------------------------------------

    async def get_last_decision_for_stock(
        self, user_id: str, stock_symbol: str, before_days: int = 30
    ) -> Optional[AnalysisRecord]:
        """Get the most recent analysis for a stock within the last N days."""
        cutoff = _utc_cutoff(before_days)
        if aiosqlite is None:
            return self._sync_get_last_decision_for_stock(user_id, stock_symbol, cutoff)
        async with self._aconnection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                """
                SELECT * FROM analysis_history
                WHERE user_id = ? AND stock_symbol = ? AND created_at > ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id, stock_symbol.upper(), cutoff),
            )
            row = await cursor.fetchone()
            return self._row_to_analysis(row) if row else None

    def _sync_get_last_decision_for_stock(
        self, user_id: str, stock_symbol: str, cutoff: str
    ) -> Optional[AnalysisRecord]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM analysis_history
                WHERE user_id = ? AND stock_symbol = ? AND created_at > ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id, stock_symbol.upper(), cutoff),
            )
            row = cursor.fetchone()
            return self._row_to_analysis(row) if row else None

    async def find_inconsistencies(
        self, user_id: str, stock_symbol: str, current_decision: str,
        exclude_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        Check if current decision reverses the direction of the previous decision.
        exclude_id: 本次刚写入的记录 id，必须排除，否则会和自己比较（旧实现的缺陷）。
        """
        prev = await asyncio.to_thread(
            self._sync_previous_decision, user_id, stock_symbol.upper(), exclude_id
        )
        if not prev:
            return None
        old = prev.final_decision
        buy_like = {"BUY", "STRONG_BUY"}
        sell_like = {"SELL", "STRONG_SELL"}
        if (old in buy_like and current_decision in sell_like) or (old in sell_like and current_decision in buy_like):
            return (
                f"方向反转告警: 上次 ({prev.created_at[:10]}) 决策为 {old}，"
                f"本次为 {current_decision}。请确认是否有重大新数据支撑此反转。"
            )
        return None

    def _sync_previous_decision(self, user_id: str, stock: str, exclude_id: Optional[int]):
        with self._sync_conn() as conn:
            conn.row_factory = sqlite3.Row
            sql = ("SELECT * FROM analysis_history WHERE user_id = ? AND stock_symbol = ? "
                   "AND created_at > ?")
            params: list = [user_id, stock, _utc_cutoff(30)]
            if exclude_id is not None:
                sql += " AND id < ?"
                params.append(exclude_id)
            row = conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()
            return self._row_to_analysis(row) if row else None

    # ------------------------------------------------------------------
    # 权重解析：有效权重 = 客观权重(global, 来自前向验证) × 个人调整(用户行, 来自反馈)
    # ------------------------------------------------------------------

    async def resolve_weights(self, user_id: str, agent_names: List[str]) -> Dict[str, float]:
        return await asyncio.to_thread(
            self._sync_resolve_weights, user_id or "anonymous", list(agent_names)
        )

    def _sync_resolve_weights(self, user_id: str, agent_names: List[str]) -> Dict[str, float]:
        if not agent_names:
            return {}
        canon = {n: canonical_agent_name(n) for n in agent_names}
        keys = sorted(set(canon.values()))
        ph = ",".join("?" * len(keys))
        with self._sync_conn() as conn:
            rows = conn.execute(
                f"SELECT user_id, agent_name, current_weight FROM agent_stats "
                f"WHERE user_id IN (?, 'global') AND agent_name IN ({ph})",
                (user_id, *keys),
            ).fetchall()
        table = {(u, a): float(w) for u, a, w in rows}
        out: Dict[str, float] = {}
        for orig, a in canon.items():
            objective = table.get(("global", a), 1.0)
            personal = table.get((user_id, a), 1.0) if user_id != "global" else 1.0
            out[orig] = max(WEIGHT_MIN, min(WEIGHT_MAX, objective * personal))
        return out

    # ------------------------------------------------------------------
    # 结果验证：待验证记录 / 写入结果 / 客观权重
    # ------------------------------------------------------------------

    async def pending_verifications(self, horizon: int, min_age_days: int, limit: int = 200) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_pending_verifications, horizon, min_age_days, limit)

    def _sync_pending_verifications(self, horizon: int, min_age_days: int, limit: int) -> List[Dict[str, Any]]:
        with self._sync_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT a.id, a.stock_symbol, a.created_at FROM analysis_history a
                WHERE a.created_at <= ?
                  AND NOT EXISTS (SELECT 1 FROM agent_outcomes o
                                  WHERE o.analysis_id = a.id AND o.horizon_days = ?)
                  AND a.verify_attempts < ?
                ORDER BY a.id LIMIT ?
                """,
                (_utc_cutoff(min_age_days), horizon, MAX_VERIFY_ATTEMPTS, limit),
            ).fetchall()
            out = []
            for r in rows:
                votes = conn.execute(
                    "SELECT agent_name, vote FROM agent_votes WHERE analysis_id = ?", (r["id"],)
                ).fetchall()
                out.append({
                    "id": r["id"],
                    "stock_symbol": r["stock_symbol"],
                    "created_at": r["created_at"],
                    "votes": {a: v for a, v in votes},
                })
            return out

    async def bump_verify_attempts(self, analysis_id: int) -> None:
        """验证仍未完成时累加尝试计数；达到 MAX_VERIFY_ATTEMPTS 后不再被拉取。"""
        await asyncio.to_thread(self._sync_bump_verify_attempts, analysis_id)

    def _sync_bump_verify_attempts(self, analysis_id: int) -> None:
        with self._sync_conn() as conn:
            conn.execute(
                "UPDATE analysis_history SET verify_attempts = verify_attempts + 1 WHERE id = ?",
                (analysis_id,),
            )
            conn.commit()

    async def save_outcome_rows(self, analysis_id: int, horizon: int, rows: List[tuple],
                                mark_verified: bool) -> None:
        """rows: (agent_name, vote, entry_date, exit_date, entry_price, exit_price,
                  stock_return, bench_return, excess_return, correct)"""
        await asyncio.to_thread(self._sync_save_outcome_rows, analysis_id, horizon, rows, mark_verified)

    def _sync_save_outcome_rows(self, analysis_id, horizon, rows, mark_verified) -> None:
        with self._sync_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO agent_outcomes
                (analysis_id, agent_name, vote, horizon_days, entry_date, exit_date,
                 entry_price, exit_price, stock_return, bench_return, excess_return, correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(analysis_id, r[0], r[1], horizon, r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
                 for r in rows],
            )
            if mark_verified:
                conn.execute(
                    "UPDATE analysis_history SET market_outcome = 'verified' WHERE id = ?",
                    (analysis_id,),
                )
            conn.commit()

    async def refresh_objective_weights(self, horizon: int = PRIMARY_HORIZON,
                                        min_samples: int = MIN_VERIFIED_SAMPLES) -> Dict[str, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_refresh_objective_weights, horizon, min_samples)

    def _sync_refresh_objective_weights(self, horizon: int, min_samples: int) -> Dict[str, Dict[str, Any]]:
        """客观权重 = clamp(1 + 2*(命中率 - 1/3), MIN, MAX)。1/3 为三分类随机基线。
        样本不足的 Agent 不更新（保持原值），避免小样本噪声。"""
        report: Dict[str, Dict[str, Any]] = {}
        with self._sync_conn() as conn:
            agents = [r[0] for r in conn.execute(
                "SELECT DISTINCT agent_name FROM agent_outcomes WHERE horizon_days = ?", (horizon,)
            ).fetchall()]
            for agent in agents:
                n, c = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM agent_outcomes "
                    "WHERE agent_name = ? AND horizon_days = ? AND correct IS NOT NULL",
                    (agent, horizon),
                ).fetchone()
                entry: Dict[str, Any] = {"n": n, "correct": c,
                                         "accuracy": round(c / n, 3) if n else None, "weight": None}
                if n >= min_samples:
                    w = max(WEIGHT_MIN, min(WEIGHT_MAX, 1.0 + 2.0 * (c / n - 1 / 3)))
                    conn.execute(
                        """
                        INSERT INTO agent_stats (agent_name, user_id, current_weight, updated_at)
                        VALUES (?, 'global', ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(agent_name, user_id) DO UPDATE SET
                            current_weight = excluded.current_weight,
                            updated_at = excluded.updated_at
                        """,
                        (agent, round(w, 4)),
                    )
                    entry["weight"] = round(w, 4)
                report[agent] = entry
            conn.commit()
        return report

    async def accuracy_report(self, horizon: int = PRIMARY_HORIZON) -> Dict[str, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_accuracy_report, horizon)

    def _sync_accuracy_report(self, horizon: int) -> Dict[str, Dict[str, Any]]:
        with self._sync_conn() as conn:
            rows = conn.execute(
                "SELECT agent_name, COUNT(*), SUM(correct) FROM agent_outcomes "
                "WHERE horizon_days = ? AND correct IS NOT NULL GROUP BY agent_name",
                (horizon,),
            ).fetchall()
        return {a: {"n": n, "correct": int(c or 0), "accuracy": round((c or 0) / n, 3)} for a, n, c in rows}

    async def analysis_outcomes(self, analysis_id: int) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_analysis_outcomes, analysis_id)

    def _sync_analysis_outcomes(self, analysis_id: int) -> List[Dict[str, Any]]:
        with self._sync_conn() as conn:
            rows = conn.execute(
                "SELECT agent_name, vote, horizon_days, entry_date, exit_date, stock_return, "
                "bench_return, excess_return, correct FROM agent_outcomes WHERE analysis_id = ? "
                "ORDER BY horizon_days, agent_name",
                (analysis_id,),
            ).fetchall()
        keys = ["agent", "vote", "horizon_days", "entry_date", "exit_date",
                "stock_return", "bench_return", "excess_return", "correct"]
        return [dict(zip(keys, r)) for r in rows]

    # ------------------------------------------------------------------
    # 自挖掘规则（self_mined 候选，默认禁用，需人工在规则面板启用后才注入）
    # ------------------------------------------------------------------

    # 与 Data.outcome 的判定口径一致；在此本地定义以避免循环导入
    _SELF_MINED_BULLISH = {"BUY", "STRONG_BUY"}
    _SELF_MINED_BEARISH = {"SELL", "STRONG_SELL", "REDUCE"}

    async def mine_outcome_rules(self, horizon: int = PRIMARY_HORIZON,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """从已验证且判为错的投票中提炼候选纠正规则。

        低配版（不调 LLM，纯模板）：
        - 数据源：agent_outcomes 里 correct=0 的行（主验证期 horizon）。
        - 规则文本对 (user_id, agent_name, stock, horizon, 失败方向) 唯一：
          同一标的同类错误反复出现时 trigger_count 自增，作为证据强度展示。
        - 新规则一律 active=0（默认禁用），source='self_mined'；
          不启用就不会注入提示词，零风险；在规则管理界面手动启用后生效。
        """
        return await asyncio.to_thread(self._sync_mine_outcome_rules, horizon, limit)

    def _sync_mine_outcome_rules(self, horizon: int = PRIMARY_HORIZON,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        with self._sync_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ao.agent_name, ao.vote, ao.excess_return,
                       h.stock_symbol, h.user_id
                FROM agent_outcomes ao
                JOIN analysis_history h ON h.id = ao.analysis_id
                WHERE ao.correct = 0 AND ao.horizon_days = ?
                  AND h.stock_symbol IS NOT NULL AND h.stock_symbol != ''
                ORDER BY ao.verified_at DESC
                LIMIT ?
                """,
                (horizon, limit),
            ).fetchall()

            mined: List[Dict[str, Any]] = []
            for r in rows:
                vote = r["vote"] or ""
                excess = r["excess_return"]
                if vote in self._SELF_MINED_BULLISH and excess < 0:
                    body = f"给出看多（{vote}），但 {horizon} 个交易日后的超额收益为负，判断不成立。此后对同一标的给出看多结论时应提高证据门槛。"
                elif vote in self._SELF_MINED_BEARISH and excess > 0:
                    body = f"给出看空（{vote}），但 {horizon} 个交易日后的超额收益为正，判断不成立。此后对同一标的给出看空结论时应提高证据门槛。"
                elif vote == "HOLD":
                    body = f"给出观望（HOLD），但 {horizon} 个交易日后的超额收益偏离基准超过阈值，观望判断不成立。此后对同一标的给出观望结论前应确认多空证据均衡。"
                else:
                    continue  # 方向矛盾/无法归类的失败不生成规则

                rule_text = f"[结果验证反馈] {r['agent_name']} 对 {r['stock_symbol']} {body}"
                user_id = r["user_id"] or "anonymous"
                agent_name = canonical_agent_name(r["agent_name"])

                existing = conn.execute(
                    "SELECT id, trigger_count, active FROM user_rules "
                    "WHERE user_id = ? AND agent_name = ? AND rule_text = ?",
                    (user_id, agent_name, rule_text),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE user_rules SET trigger_count = trigger_count + 1 WHERE id = ?",
                        (existing["id"],),
                    )
                    mined_result = {"id": existing["id"], "rule_text": rule_text,
                                    "trigger_count": existing["trigger_count"] + 1,
                                    "active": bool(existing["active"]), "created": False}
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO user_rules (user_id, agent_name, rule_text, source, active)
                        VALUES (?, ?, ?, 'self_mined', 0)
                        """,
                        (user_id, agent_name, rule_text),
                    )
                    mined_result = {"id": cur.lastrowid, "rule_text": rule_text,
                                    "trigger_count": 1, "active": False, "created": True}
                mined.append(mined_result)
            conn.commit()
        return mined

    # ------------------------------------------------------------------
    # 规则管理（列表 / 启用禁用 / 删除）
    # ------------------------------------------------------------------

    async def list_rules(self, user_id: str, include_inactive: bool = True) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_list_rules, user_id, include_inactive)

    def _sync_list_rules(self, user_id: str,
                         include_inactive: bool = True) -> List[Dict[str, Any]]:
        sql = ("SELECT id, agent_name, rule_text, trigger_count, source, active, created_at "
               "FROM user_rules WHERE user_id = ?")
        if not include_inactive:
            sql += " AND active = 1"
        sql += " ORDER BY active DESC, trigger_count DESC, id DESC"
        with self._sync_conn() as conn:
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [
            {"id": r[0], "agent_name": r[1], "rule_text": r[2], "trigger_count": r[3],
             "source": r[4], "active": bool(r[5]), "created_at": r[6]}
            for r in rows
        ]

    async def set_rule_active(self, rule_id: int, user_id: str, active: bool) -> bool:
        return await asyncio.to_thread(self._sync_set_rule_active, rule_id, user_id, active)

    def _sync_set_rule_active(self, rule_id: int, user_id: str, active: bool) -> bool:
        with self._sync_conn() as conn:
            cur = conn.execute(
                "UPDATE user_rules SET active = ? WHERE id = ? AND user_id = ?",
                (1 if active else 0, rule_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0

    async def delete_rule(self, rule_id: int, user_id: str) -> bool:
        return await asyncio.to_thread(self._sync_delete_rule, rule_id, user_id)

    def _sync_delete_rule(self, rule_id: int, user_id: str) -> bool:
        with self._sync_conn() as conn:
            cur = conn.execute("DELETE FROM user_rules WHERE id = ? AND user_id = ?", (rule_id, user_id))
            conn.commit()
            return cur.rowcount > 0


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
