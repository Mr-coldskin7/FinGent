"""
L3 Long-term Memory Manager — SQLite-based persistent memory layer
for FinGent multi-agent stock analysis system.

Tables:
- analysis_history: Record of every analysis run (votes, decisions, feedback)
- agent_stats: Per-agent performance statistics and dynamic weights
- user_rules: Rules extracted from user feedback, injected into agent prompts
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
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
            conn.commit()

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
    ) -> int:
        """Record a new analysis entry. Returns the row id."""
        if aiosqlite is None:
            return self._sync_record_analysis(
                session_id, stock_symbol, query, agent_votes,
                final_decision, reasoning_summary, user_id,
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
                    json.dumps(agent_votes, ensure_ascii=False),
                    final_decision,
                    reasoning_summary,
                ),
            )
            await conn.commit()
            return cursor.lastrowid

    def _sync_record_analysis(
        self,
        session_id: str,
        stock_symbol: str,
        query: str,
        agent_votes: Dict[str, str],
        final_decision: str,
        reasoning_summary: str = "",
        user_id: str = "anonymous",
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
                    json.dumps(agent_votes, ensure_ascii=False),
                    final_decision,
                    reasoning_summary,
                ),
            )
            conn.commit()
            return cursor.lastrowid

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
    ):
        """Record the outcome of an agent run and update stats/weights accordingly."""
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
                    created_at=datetime.now().isoformat(),
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
                    created_at=datetime.now().isoformat(),
                )

    async def get_rules(
        self,
        user_id: str,
        agent_name: Optional[str] = None,
        min_trigger_count: int = 1,
    ) -> List[UserRule]:
        """Get active rules for a user, optionally filtered by agent."""
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
            weights = await self.get_agent_weights(user_id, [agent_name])
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
        cutoff = (datetime.now() - timedelta(days=before_days)).isoformat()
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
        self, user_id: str, stock_symbol: str, current_decision: str
    ) -> Optional[str]:
        """
        Check if current decision contradicts recent history without obvious reason.
        Returns a description of the inconsistency, or None if consistent.
        """
        last = await self.get_last_decision_for_stock(user_id, stock_symbol, before_days=30)
        if not last:
            return None
        # Simple heuristic: direction changed
        old = last.final_decision
        current = current_decision
        buy_like = {"BUY", "STRONG_BUY"}
        sell_like = {"SELL", "STRONG_SELL"}
        if (old in buy_like and current in sell_like) or (old in sell_like and current in buy_like):
            return (
                f"方向反转告警: 上次 ({last.created_at[:10]}) 决策为 {old}，"
                f"本次为 {current}。请确认是否有重大新数据支撑此反转。"
            )
        return None


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
