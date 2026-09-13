# karyon_agent_runtime/db_manager.py
"""
===============================================================================
KAGGLE AGENT PERSISTENT DATABASE (AIOSQLITE) - SELF-HEALING & MIGRATION ENGINE (v30.0)
High-Performance SQLite Storage with Auto-Recovery, Indexing, and WAL Immunity.
Features Seamless Column/Table Migrations, Lossless Turn Archival, Empirical Ledger,
Key-Value Memory, Unified Background Process Tracking, Runtime Settings Persistence,
and Hierarchical Multi-Agent Cortical Architecture (Sub-Agents, Lineage Protection,
Synaptic Inter-Agent Messaging, Delegated Tasks, and Autonomous Dialogue Turns).
Enhanced with Re-entrant Coroutine Lock Protection (AsyncRLock) and Concurrency Immunity.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import aiosqlite
import asyncio
import json
import base64
import logging
import os
import time
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from google.genai import types
import karyon_agent_runtime.config as config

logger = logging.getLogger("ProxyAgent.Database")

__all__ = ["AgentDBManager", "dict_to_content", "content_to_dict", "clean_for_json", "AsyncRLock"]


class AsyncRLock:
    """Re-entrant asyncio lock preventing deadlocks on recursive internal calls."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count = 0

    async def acquire(self):
        current_task = asyncio.current_task()
        if self._owner == current_task:
            self._count += 1
            return True
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return True

    def release(self):
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError("Cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


def clean_for_json(obj):
    """Recursively converts objects to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, "model_dump"):
        return clean_for_json(obj.model_dump())
    elif hasattr(obj, "__dict__"):
        return clean_for_json(obj.__dict__)
    return obj


def content_to_dict(content: types.Content) -> dict:
    """Converts a Google GenAI Content object into a JSON-compatible dict."""
    if hasattr(content, "model_dump"):
        data = content.model_dump()
    else:
        data = dict(content)
    return clean_for_json(data)


def dict_to_content(data: dict) -> types.Content:
    """Restores a Google GenAI Content object from a dictionary."""
    def restore_bytes(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k in ["thought_signature", "thoughtSignature"] and isinstance(v, str):
                    try:
                        new_dict[k] = base64.b64decode(v)
                    except Exception:
                        new_dict[k] = v
                else:
                    new_dict[k] = restore_bytes(v)
            return new_dict
        elif isinstance(obj, list):
            return [restore_bytes(v) for v in obj]
        return obj

    restored = restore_bytes(data)
    if "parts" not in restored or restored["parts"] is None:
        restored["parts"] = []
    return types.Content.model_validate(restored)


class AgentDBManager:
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is not None:
            self.db_path = config.resolve_database_path(db_path)
        else:
            self.db_path = config.get_active_db_path()
        self.db: Optional[aiosqlite.Connection] = None
        self._lock = AsyncRLock()

    def _cleanup_stale_lock_files(self):
        """Removes stale SQLite lock artifacts that cause disk I/O freezes on overlay filesystems."""
        for suffix in ["-wal", "-shm", "-journal", "-lock"]:
            stale = Path(str(self.db_path) + suffix)
            if stale.exists():
                try:
                    stale.unlink()
                    logger.debug(f"Cleaned up stale SQLite artifact: {stale.name}")
                except Exception:
                    pass

    async def connect(self):
        """Connects to SQLite with self-healing auto-recovery, speed optimizations and auto-migration."""
        async with self._lock:
            if self.db is not None:
                try:
                    await self.db.execute("SELECT 1;")
                    return
                except Exception:
                    try:
                        await self.db.close()
                    except Exception:
                        pass
                    self.db = None

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cleanup_stale_lock_files()

            try:
                self.db = await aiosqlite.connect(str(self.db_path), timeout=15.0)
                await self.db.execute("PRAGMA journal_mode=DELETE;")
                await self.db.execute("PRAGMA synchronous=NORMAL;")
                await self.db.execute("PRAGMA busy_timeout=15000;")
                await self.db.execute("PRAGMA cache_size=-32000;")
                await self._init_tables()
                await self._run_migrations()
                logger.info(f"Agent SQLite database connected and migrated: {self.db_path.name}")

            except Exception as e:
                logger.warning(f"Initial DB connection hit error ({str(e)}). Running Self-Healing Recovery...")
                if self.db:
                    try:
                        await self.db.close()
                    except Exception:
                        pass

                self._cleanup_stale_lock_files()

                if self.db_path.exists():
                    backup_name = str(self.db_path) + f".corrupted_{int(time.time())}"
                    try:
                        self.db_path.rename(backup_name)
                        logger.warning(f"Moved corrupted database to: {backup_name}")
                    except Exception:
                        pass

                self.db = await aiosqlite.connect(str(self.db_path), timeout=15.0)
                await self.db.execute("PRAGMA journal_mode=DELETE;")
                await self.db.execute("PRAGMA synchronous=NORMAL;")
                await self.db.execute("PRAGMA busy_timeout=15000;")
                await self._init_tables()
                await self._run_migrations()
                logger.info(f"Agent SQLite database self-healed and reconnected: {self.db_path.name}")

    async def switch_active_database(self, new_db_path: Union[str, Path]) -> Path:
        """Atomically switches active database under lock, updating pointers."""
        canonical_target = config.resolve_database_path(new_db_path)
        async with self._lock:
            if self.db:
                try:
                    await self.checkpoint()
                    await self.db.close()
                except Exception as e:
                    logger.debug(f"Close notice during database switch: {str(e)}")
                finally:
                    self.db = None

            self.db_path = canonical_target
            config.set_active_db_pointer(canonical_target)

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cleanup_stale_lock_files()
            self.db = await aiosqlite.connect(str(self.db_path), timeout=15.0)
            await self.db.execute("PRAGMA journal_mode=DELETE;")
            await self.db.execute("PRAGMA synchronous=NORMAL;")
            await self.db.execute("PRAGMA busy_timeout=15000;")
            await self.db.execute("PRAGMA cache_size=-32000;")
            await self._init_tables()
            await self._run_migrations()
            logger.info(f"Agent SQLite database switched and migrated: {self.db_path.name}")
            return canonical_target

    async def _init_tables(self):
        """Creates tables and indexes for dialogs, ledger, memory, jobs, settings and multi-agents."""
        async with self.db.cursor() as cursor:
            # 1. Turns & History
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    raw_parts_json TEXT DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_id ON turns(id DESC);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_role ON turns(role);")

            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns_archive (
                    id INTEGER PRIMARY KEY,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    raw_parts_json TEXT DEFAULT NULL,
                    timestamp DATETIME,
                    archived_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_arch_id ON turns_archive(id DESC);")

            # 2. Empirical Scientific Ledger
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS empirical_ledger (
                    exp_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    hypothesis TEXT NOT NULL,
                    architecture_delta TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    final_loss REAL,
                    metrics_json TEXT DEFAULT NULL,
                    config_json TEXT DEFAULT NULL,
                    notes TEXT DEFAULT NULL
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON empirical_ledger(timestamp DESC);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_verdict ON empirical_ledger(verdict);")

            # 3. Key-Value Memory
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category);")

            # 4. Background Unified Processes & Jobs
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS unified_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    command_or_code TEXT NOT NULL,
                    pid INTEGER,
                    pgid INTEGER,
                    status TEXT NOT NULL,
                    exit_code INTEGER DEFAULT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL DEFAULT NULL,
                    duration REAL DEFAULT NULL,
                    log_file TEXT NOT NULL,
                    total_lines INTEGER DEFAULT 0,
                    last_polled_line INTEGER DEFAULT 0,
                    meta_json TEXT DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_status ON unified_jobs(status);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_type ON unified_jobs(job_type);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_start ON unified_jobs(start_time DESC);")

            # Backward-compatibility legacy job tables
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS background_jobs (
                    job_id TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    pid INTEGER,
                    pgid INTEGER,
                    status TEXT NOT NULL,
                    exit_code INTEGER DEFAULT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL DEFAULT NULL,
                    duration REAL DEFAULT NULL,
                    log_file TEXT NOT NULL,
                    total_lines INTEGER DEFAULT 0,
                    last_polled_line INTEGER DEFAULT 0,
                    meta_json TEXT DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS background_tool_jobs (
                    job_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    tool_args_json TEXT DEFAULT NULL,
                    description TEXT DEFAULT NULL,
                    status TEXT NOT NULL,
                    result_preview TEXT DEFAULT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL DEFAULT NULL,
                    duration REAL DEFAULT NULL,
                    log_file TEXT NOT NULL,
                    total_lines INTEGER DEFAULT 0,
                    last_polled_line INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS custom_tools (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    parameters_schema_json TEXT DEFAULT NULL,
                    code TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 5. Persistent Runtime Configuration Settings (Survives Notebook Restarts)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 6. Multi-Agent Cortical Hierarchy & Synaptic Network
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS sub_agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    parent_id TEXT DEFAULT NULL,
                    system_prompt TEXT NOT NULL,
                    model TEXT DEFAULT 'gemini-3.7-flash',
                    thinking_level TEXT DEFAULT 'HIGH',
                    thinking_budget INTEGER DEFAULT 2048,
                    allowed_tools_json TEXT DEFAULT '["*"]',
                    blocked_tools_json TEXT DEFAULT '[]',
                    can_communicate_with_peers INTEGER DEFAULT 1,
                    allowed_peers_json TEXT DEFAULT '["*"]',
                    max_turns INTEGER DEFAULT 30,
                    is_active INTEGER DEFAULT 1,
                    metadata_json TEXT DEFAULT '{}',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_agents_parent ON sub_agents(parent_id);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_agents_role ON sub_agents(role);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_agents_active ON sub_agents(is_active);")

            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_agent_id TEXT NOT NULL,
                    to_agent_id TEXT NOT NULL,
                    msg_type TEXT NOT NULL,
                    subject TEXT DEFAULT '',
                    body TEXT NOT NULL,
                    task_id TEXT DEFAULT NULL,
                    reply_to_id INTEGER DEFAULT NULL,
                    status TEXT DEFAULT 'DELIVERED',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_msg_to ON agent_messages(to_agent_id, status);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_msg_from ON agent_messages(from_agent_id);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_msg_task ON agent_messages(task_id);")

            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    creator_agent_id TEXT NOT NULL,
                    assigned_agent_id TEXT NOT NULL,
                    task_prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT DEFAULT NULL,
                    error TEXT DEFAULT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL DEFAULT NULL,
                    duration REAL DEFAULT NULL,
                    turns_used INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_tasks_assigned ON agent_tasks(assigned_agent_id);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_tasks_creator ON agent_tasks(creator_agent_id);")

            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    task_id TEXT DEFAULT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    raw_parts_json TEXT DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_turns_agent ON agent_turns(agent_id, id DESC);")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_turns_task ON agent_turns(task_id);")

        await self.db.commit()

    async def _run_migrations(self):
        """Seamlessly adds missing columns or tables to older database schemas."""
        async def add_column_if_missing(table_name: str, column_name: str, column_type: str):
            try:
                async with self.db.execute(f"PRAGMA table_info({table_name});") as cur:
                    cols = [r[1] for r in await cur.fetchall()]
                    if cols and column_name not in cols:
                        await self.db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
                        logger.info(f"Database Migration: Added column '{column_name}' to table '{table_name}'.")
            except Exception as e:
                logger.debug(f"Migration notice for {table_name}.{column_name}: {str(e)}")

        await add_column_if_missing("background_jobs", "last_polled_line", "INTEGER DEFAULT 0")
        await add_column_if_missing("background_jobs", "total_lines", "INTEGER DEFAULT 0")
        await add_column_if_missing("background_jobs", "duration", "REAL DEFAULT NULL")
        await add_column_if_missing("background_tool_jobs", "last_polled_line", "INTEGER DEFAULT 0")
        await add_column_if_missing("background_tool_jobs", "total_lines", "INTEGER DEFAULT 0")
        await add_column_if_missing("turns", "raw_parts_json", "TEXT DEFAULT NULL")
        await add_column_if_missing("turns_archive", "raw_parts_json", "TEXT DEFAULT NULL")
        await self.db.commit()

    async def checkpoint(self):
        """Forces pending transactions to commit to disk."""
        async with self._lock:
            if self.db:
                try:
                    await self.db.commit()
                except Exception as e:
                    logger.debug(f"DB commit notice: {str(e)}")

    # =========================================================================
    # 1. DIALOGUE TURNS & ARCHIVE
    # =========================================================================
    async def save_turn(self, role: str, text: str, content_obj: types.Content = None):
        raw_json = json.dumps(content_to_dict(content_obj), ensure_ascii=False) if content_obj else None
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute(
                    "INSERT INTO turns (role, text, raw_parts_json) VALUES (?, ?, ?)",
                    (role, text, raw_json)
                )
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error in save_turn ({str(e)}). Reconnecting...")
                await self.connect()
                await self.db.execute(
                    "INSERT INTO turns (role, text, raw_parts_json) VALUES (?, ?, ?)",
                    (role, text, raw_json)
                )
                await self.db.commit()

    async def get_recent_turns(self, limit: int = 60, include_tools: bool = True) -> List[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                query = "SELECT id, role, text, raw_parts_json FROM turns ORDER BY id DESC LIMIT ?"
                async with self.db.execute(query, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    rows.reverse()
                    cleaned = []
                    for r in rows:
                        role_val = r[1]
                        if not include_tools and role_val in ["tool_call", "tool_result"]:
                            continue
                        cleaned.append({"id": r[0], "role": role_val, "text": r[2], "raw_parts_json": r[3]})
                    return cleaned
            except Exception as e:
                logger.warning(f"Error reading turns ({str(e)}). Attempting reconnect...")
                await self.connect()
                async with self.db.execute("SELECT id, role, text, raw_parts_json FROM turns ORDER BY id DESC LIMIT ?", (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    rows.reverse()
                    return [{"id": r[0], "role": r[1], "text": r[2], "raw_parts_json": r[3]} for r in rows if include_tools or r[1] not in ["tool_call", "tool_result"]]

    async def archive_old_turns(self, keep_count: int = 16) -> int:
        """Archives and prunes older turns from active working memory into turns_archive."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()

                async with self.db.execute("SELECT id FROM turns ORDER BY id DESC LIMIT ?", (keep_count,)) as cursor:
                    rows = await cursor.fetchall()
                    if not rows or len(rows) < keep_count:
                        return 0
                    oldest_preserved_id = rows[-1][0]

                await self.db.execute("""
                    INSERT OR REPLACE INTO turns_archive (id, role, text, raw_parts_json, timestamp)
                    SELECT id, role, text, raw_parts_json, timestamp FROM turns WHERE id < ?
                """, (oldest_preserved_id,))

                async with self.db.execute("DELETE FROM turns WHERE id < ?", (oldest_preserved_id,)) as del_cur:
                    pruned_count = del_cur.rowcount

                await self.db.commit()
                logger.info(f"Context Pruning: Archived {pruned_count} turns into 'turns_archive'. Preserved {keep_count} active turns.")
                return pruned_count

            except Exception as e:
                logger.error(f"Error archiving old turns: {str(e)}")
                return 0

    async def clear_turns(self):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("DELETE FROM turns")
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error in clear_turns ({str(e)})")
                await self.connect()
                await self.db.execute("DELETE FROM turns")
                await self.db.commit()

    # =========================================================================
    # 2. EMPIRICAL SCIENTIFIC LEDGER
    # =========================================================================
    async def record_experiment(
        self,
        exp_id: str,
        hypothesis: str,
        delta: str,
        verdict: str,
        final_loss: float = None,
        metrics: dict = None,
        config_dict: dict = None,
        notes: str = None
    ):
        metrics_str = json.dumps(clean_for_json(metrics), ensure_ascii=False) if metrics else None
        config_str = json.dumps(clean_for_json(config_dict), ensure_ascii=False) if config_dict else None

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO empirical_ledger (
                        exp_id, hypothesis, architecture_delta, verdict, final_loss, metrics_json, config_json, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (exp_id, hypothesis, delta, verdict, final_loss, metrics_str, config_str, notes))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error recording experiment ({str(e)}). Reconnecting...")
                await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO empirical_ledger (
                        exp_id, hypothesis, architecture_delta, verdict, final_loss, metrics_json, config_json, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (exp_id, hypothesis, delta, verdict, final_loss, metrics_str, config_str, notes))
                await self.db.commit()

    async def get_all_experiments(
        self,
        search_query: str = None,
        verdict_filter: str = None,
        order: str = "desc",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM empirical_ledger"
        params = []
        conditions = []

        if verdict_filter:
            conditions.append("verdict LIKE ?")
            params.append(f"%{verdict_filter}%")

        if search_query:
            conditions.append("(exp_id LIKE ? OR hypothesis LIKE ? OR architecture_delta LIKE ? OR notes LIKE ? OR metrics_json LIKE ?)")
            q = f"%{search_query}%"
            params.extend([q, q, q, q, q])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        order_dir = "DESC" if str(order).lower() == "desc" else "ASC"
        query += f" ORDER BY timestamp {order_dir} LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
                    cols = [d[0] for d in cursor.description]
                    results = []
                    for r in rows:
                        item = dict(zip(cols, r))
                        if item.get("metrics_json"):
                            try:
                                item["metrics"] = json.loads(item["metrics_json"])
                            except Exception:
                                item["metrics"] = {}
                        if item.get("config_json"):
                            try:
                                item["config"] = json.loads(item["config_json"])
                            except Exception:
                                item["config"] = {}
                        results.append(item)
                    return results
            except Exception as e:
                logger.warning(f"Error fetching experiments ({str(e)}). Self-healing...")
                await self.connect()
                return []

    async def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM empirical_ledger WHERE exp_id LIKE ?", (f"%{exp_id.strip()}%",)) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cursor.description]
                    item = dict(zip(cols, row))
                    if item.get("metrics_json"):
                        try:
                            item["metrics"] = json.loads(item["metrics_json"])
                        except Exception:
                            item["metrics"] = {}
                    if item.get("config_json"):
                        try:
                            item["config"] = json.loads(item["config_json"])
                        except Exception:
                            item["config"] = {}
                    return item
            except Exception as e:
                logger.warning(f"Error fetching experiment {exp_id}: {str(e)}")
                return None

    # =========================================================================
    # 3. KEY-VALUE MEMORY
    # =========================================================================
    async def set_memory(self, key: str, value: Union[str, dict, list], category: str = "general"):
        val_str = json.dumps(clean_for_json(value), ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO memory (key, value, category, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (key, val_str, category))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error saving memory ({str(e)}). Reconnecting...")
                await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO memory (key, value, category, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (key, val_str, category))
                await self.db.commit()

    async def get_memory(self, key: str) -> Optional[str]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT value FROM memory WHERE key = ?", (key,)) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else None
            except Exception as e:
                logger.warning(f"Error reading memory ({str(e)})")
                return None

    async def delete_memory(self, key: str) -> bool:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM memory WHERE key = ?", (key,)) as cursor:
                    await self.db.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting memory ({str(e)})")
                return False

    async def list_memory(self, category: str = None) -> List[Dict[str, str]]:
        query = "SELECT key, value, category, updated_at FROM memory"
        params = []
        if category:
            query += " WHERE category = ?"
            params.append(category)
        query += " ORDER BY updated_at DESC"

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
                    return [{"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]} for r in rows]
            except Exception as e:
                logger.warning(f"Error listing memory ({str(e)})")
                return []

    # =========================================================================
    # 4. PERSISTENT RUNTIME SETTINGS (RESTART SURVIVABILITY)
    # =========================================================================
    async def set_setting(self, key: str, value: Any):
        """Persistently saves a setting into SQLite as JSON."""
        val_json = json.dumps(clean_for_json(value), ensure_ascii=False)
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO runtime_settings (key, value_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (str(key), val_json))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error saving runtime setting '{key}': {str(e)}")

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieves and deserializes a persistent runtime setting from SQLite."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT value_json FROM runtime_settings WHERE key = ?", (str(key),)) as cur:
                    row = await cur.fetchone()
                    if row and row[0]:
                        try:
                            return json.loads(row[0])
                        except Exception:
                            return row[0]
                    return default
            except Exception as e:
                logger.debug(f"Notice reading runtime setting '{key}': {str(e)}")
                return default

    async def get_all_settings(self) -> Dict[str, Any]:
        """Retrieves all persistent runtime settings in a dictionary."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT key, value_json FROM runtime_settings") as cur:
                    rows = await cur.fetchall()
                    result = {}
                    for k, val_json in rows:
                        try:
                            result[k] = json.loads(val_json)
                        except Exception:
                            result[k] = val_json
                    return result
            except Exception as e:
                logger.warning(f"Error loading runtime settings: {str(e)}")
                return {}

    async def delete_setting(self, key: str) -> bool:
        """Deletes a persistent runtime setting from SQLite."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM runtime_settings WHERE key = ?", (str(key),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting runtime setting '{key}': {str(e)}")
                return False

    # =========================================================================
    # 5. MULTI-AGENT CORTICAL HIERARCHY (REGISTRY, LINEAGE & PERMISSIONS)
    # =========================================================================
    async def create_sub_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        parent_id: Optional[str] = None,
        system_prompt: str = "",
        model: str = "gemini-3.7-flash",
        thinking_level: str = "HIGH",
        thinking_budget: int = 2048,
        allowed_tools: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        can_communicate_with_peers: bool = True,
        allowed_peers: Optional[List[str]] = None,
        max_turns: int = 30,
        metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Registers a new subagent into SQLite with strict parent binding."""
        allowed_json = json.dumps(allowed_tools or ["*"], ensure_ascii=False)
        blocked_json = json.dumps(blocked_tools or [], ensure_ascii=False)
        peers_json = json.dumps(allowed_peers or ["*"], ensure_ascii=False)
        meta_json = json.dumps(clean_for_json(metadata or {}), ensure_ascii=False)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO sub_agents (
                        agent_id, name, role, parent_id, system_prompt, model, thinking_level, thinking_budget,
                        allowed_tools_json, blocked_tools_json, can_communicate_with_peers, allowed_peers_json,
                        max_turns, is_active, metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                """, (
                    agent_id.strip(), name.strip(), role.strip(), parent_id, system_prompt,
                    model.strip(), thinking_level.strip().upper(), int(thinking_budget),
                    allowed_json, blocked_json, 1 if can_communicate_with_peers else 0,
                    peers_json, int(max_turns), meta_json
                ))
                await self.db.commit()
                return await self.get_sub_agent(agent_id)
            except Exception as e:
                logger.error(f"Error creating subagent '{agent_id}': {str(e)}")
                raise

    async def update_sub_agent(self, agent_id: str, **kwargs) -> bool:
        """Updates specific fields of an existing subagent."""
        if not kwargs:
            return False

        allowed_fields = {
            "name", "role", "system_prompt", "model", "thinking_level",
            "thinking_budget", "allowed_tools_json", "blocked_tools_json",
            "can_communicate_with_peers", "allowed_peers_json", "max_turns",
            "is_active", "metadata_json"
        }

        set_clauses = []
        params = []
        for k, v in kwargs.items():
            if k in allowed_fields:
                set_clauses.append(f"{k} = ?")
                if isinstance(v, (dict, list)):
                    params.append(json.dumps(clean_for_json(v), ensure_ascii=False))
                elif isinstance(v, bool):
                    params.append(1 if v else 0)
                else:
                    params.append(v)

        if not set_clauses:
            return False

        query = f"UPDATE sub_agents SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE agent_id = ?"
        params.append(agent_id.strip())

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.error(f"Error updating subagent '{agent_id}': {str(e)}")
                return False

    async def delete_sub_agent(self, agent_id: str) -> bool:
        """Deletes a subagent from SQLite."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM sub_agents WHERE agent_id = ?", (agent_id.strip(),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.error(f"Error deleting subagent '{agent_id}': {str(e)}")
                return False

    async def get_sub_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a subagent record by ID with parsed JSON fields."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM sub_agents WHERE agent_id = ?", (agent_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    item = dict(zip(cols, row))
                    for f in ["allowed_tools_json", "blocked_tools_json", "allowed_peers_json", "metadata_json"]:
                        if item.get(f):
                            try:
                                item[f.replace("_json", "")] = json.loads(item[f])
                            except Exception:
                                item[f.replace("_json", "")] = []
                    return item
            except Exception as e:
                logger.warning(f"Error fetching subagent '{agent_id}': {str(e)}")
                return None

    async def list_sub_agents(self, parent_id: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Lists subagents matching optional parent filter."""
        query = "SELECT * FROM sub_agents WHERE 1=1"
        params = []
        if active_only:
            query += " AND is_active = 1"
        if parent_id is not None:
            query += " AND parent_id = ?"
            params.append(parent_id.strip())
        query += " ORDER BY created_at ASC"

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    results = []
                    for r in rows:
                        item = dict(zip(cols, r))
                        for f in ["allowed_tools_json", "blocked_tools_json", "allowed_peers_json", "metadata_json"]:
                            if item.get(f):
                                try:
                                    item[f.replace("_json", "")] = json.loads(item[f])
                                except Exception:
                                    item[f.replace("_json", "")] = []
                        results.append(item)
                    return results
            except Exception as e:
                logger.warning(f"Error listing subagents: {str(e)}")
                return []

    async def get_agent_ancestors(self, agent_id: str) -> List[str]:
        """Returns the list of all ancestor IDs for an agent up to root."""
        ancestors = []
        curr_id = agent_id.strip()
        visited = set()

        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            agent = await self.get_sub_agent(curr_id)
            if agent and agent.get("parent_id"):
                p_id = agent["parent_id"].strip()
                ancestors.append(p_id)
                curr_id = p_id
            else:
                break

        return ancestors

    async def is_descendant_of(self, child_id: str, parent_id: str) -> bool:
        """Verifies if child_id is in the lineage descendant tree of parent_id."""
        ancestors = await self.get_agent_ancestors(child_id)
        return parent_id.strip() in ancestors

    async def can_manage_agent(self, acting_agent_id: str, target_agent_id: str) -> Tuple[bool, str]:
        """
        Enforces Strict Hierarchical Permission Invariant:
        1. Target cannot be 'root' (Root is immutable).
        2. Agent cannot edit or delete itself.
        3. A child CANNOT edit or delete its parent or any ancestor.
        4. An agent can only manage agents that are its own descendants (or root can manage all).
        """
        acting = acting_agent_id.strip()
        target = target_agent_id.strip()

        if target == "root":
            return False, "Permission Denied: The root executive agent cannot be modified or deleted."

        if acting == target:
            return False, "Permission Denied: An agent cannot modify or delete itself directly."

        ancestors = await self.get_agent_ancestors(acting)
        if target in ancestors:
            return False, f"Permission Denied: Agent '{acting}' cannot manage parent/ancestor '{target}'."

        if acting == "root":
            return True, "Authorized (Root sovereign authority)"

        is_desc = await self.is_descendant_of(target, acting)
        if is_desc:
            return True, "Authorized (Descendant management)"

        return False, f"Permission Denied: Agent '{acting}' does not possess ownership over '{target}'."

    # =========================================================================
    # 6. SYNAPTIC INTER-AGENT MESSAGING
    # =========================================================================
    async def record_agent_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        msg_type: str,
        body: str,
        subject: str = "",
        task_id: Optional[str] = None,
        reply_to_id: Optional[int] = None
    ) -> int:
        """Records an inter-agent message into SQLite."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("""
                    INSERT INTO agent_messages (
                        from_agent_id, to_agent_id, msg_type, subject, body, task_id, reply_to_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'DELIVERED')
                """, (
                    from_agent_id.strip(), to_agent_id.strip(), msg_type.strip(),
                    subject.strip(), body.strip(), task_id, reply_to_id
                )) as cur:
                    await self.db.commit()
                    return cur.lastrowid
            except Exception as e:
                logger.error(f"Error recording message from '{from_agent_id}' to '{to_agent_id}': {str(e)}")
                raise

    async def get_agent_inbox_messages(
        self,
        agent_id: str,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves messages received by this agent."""
        query = "SELECT * FROM agent_messages WHERE to_agent_id = ?"
        params = [agent_id.strip()]
        if unread_only:
            query += " AND status = 'DELIVERED'"
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error fetching inbox for '{agent_id}': {str(e)}")
                return []

    async def mark_message_read(self, message_id: int) -> bool:
        """Marks a delivered message as READ."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("UPDATE agent_messages SET status = 'READ' WHERE id = ?", (message_id,)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.debug(f"Notice marking message #{message_id} read: {str(e)}")
                return False

    async def get_conversation_between_agents(
        self,
        agent_a: str,
        agent_b: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves bidirectional dialogue messages exchanged between two agents."""
        query = """
            SELECT * FROM agent_messages
            WHERE (from_agent_id = ? AND to_agent_id = ?)
               OR (from_agent_id = ? AND to_agent_id = ?)
            ORDER BY id ASC LIMIT ?
        """
        params = (agent_a.strip(), agent_b.strip(), agent_b.strip(), agent_a.strip(), limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, params) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error fetching conversation: {str(e)}")
                return []

    # =========================================================================
    # 7. DELEGATED AGENT TASKS
    # =========================================================================
    async def create_agent_task(
        self,
        task_id: str,
        creator_agent_id: str,
        assigned_agent_id: str,
        task_prompt: str
    ):
        """Records a new delegated agent task."""
        now = time.time()
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT INTO agent_tasks (
                        task_id, creator_agent_id, assigned_agent_id, task_prompt, status, start_time, updated_at
                    ) VALUES (?, ?, ?, ?, 'RUNNING', ?, CURRENT_TIMESTAMP)
                """, (task_id.strip(), creator_agent_id.strip(), assigned_agent_id.strip(), task_prompt.strip(), now))
                await self.db.commit()
            except Exception as e:
                logger.error(f"Error recording agent task '{task_id}': {str(e)}")

    async def update_agent_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
        end_time: Optional[float] = None,
        turns_used: Optional[int] = None
    ):
        """Updates the status and output of a delegated agent task."""
        now = end_time or time.time()
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()

                duration = None
                async with self.db.execute("SELECT start_time FROM agent_tasks WHERE task_id = ?", (task_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if row and row[0]:
                        duration = max(0.0, now - row[0])

                query = "UPDATE agent_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP"
                params = [status.strip().upper()]

                if result is not None:
                    query += ", result = ?"
                    params.append(result)
                if error is not None:
                    query += ", error = ?"
                    params.append(error)
                if duration is not None:
                    query += ", duration = ?, end_time = ?"
                    params.extend([duration, now])
                if turns_used is not None:
                    query += ", turns_used = ?"
                    params.append(int(turns_used))

                query += " WHERE task_id = ?"
                params.append(task_id.strip())

                await self.db.execute(query, tuple(params))
                await self.db.commit()
            except Exception as e:
                logger.error(f"Error updating agent task '{task_id}': {str(e)}")

    async def get_agent_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an agent task record by ID."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM agent_tasks WHERE task_id = ?", (task_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
            except Exception as e:
                logger.warning(f"Error fetching agent task '{task_id}': {str(e)}")
                return None

    async def list_agent_tasks(
        self,
        assigned_agent_id: Optional[str] = None,
        creator_agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Lists agent tasks with optional filters."""
        query = "SELECT * FROM agent_tasks WHERE 1=1"
        params = []
        if assigned_agent_id:
            query += " AND assigned_agent_id = ?"
            params.append(assigned_agent_id.strip())
        if creator_agent_id:
            query += " AND creator_agent_id = ?"
            params.append(creator_agent_id.strip())
        if status:
            query += " AND status = ?"
            params.append(status.strip().upper())
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error listing agent tasks: {str(e)}")
                return []

    # =========================================================================
    # 8. SUB-AGENT DIALOGUE TURNS
    # =========================================================================
    async def save_subagent_turn(
        self,
        agent_id: str,
        role: str,
        text: str,
        content_obj: Optional[types.Content] = None,
        task_id: Optional[str] = None
    ):
        """Records a dialogue turn performed by a subagent."""
        raw_json = json.dumps(content_to_dict(content_obj), ensure_ascii=False) if content_obj else None
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT INTO agent_turns (agent_id, task_id, role, text, raw_parts_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (agent_id.strip(), task_id, role.strip(), text, raw_json))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error in save_subagent_turn ({str(e)})")

    async def get_subagent_turns(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieves recent dialogue turns executed by a specific subagent."""
        query = "SELECT id, role, text, raw_parts_json, task_id, timestamp FROM agent_turns WHERE agent_id = ?"
        params = [agent_id.strip()]
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id.strip())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    rows.reverse()
                    return [{"id": r[0], "role": r[1], "text": r[2], "raw_parts_json": r[3], "task_id": r[4], "timestamp": r[5]} for r in rows]
            except Exception as e:
                logger.warning(f"Error reading subagent turns: {str(e)}")
                return []

    # =========================================================================
    # 9. UNIFIED JOBS ENGINE
    # =========================================================================
    async def record_unified_job(
        self,
        job_id: str,
        job_type: str,
        command_or_code: str,
        pid: int,
        pgid: int,
        status: str,
        start_time: float,
        log_file: str,
        meta_dict: Optional[dict] = None
    ):
        meta_str = json.dumps(clean_for_json(meta_dict), ensure_ascii=False) if meta_dict else None
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO unified_jobs (
                        job_id, job_type, command_or_code, pid, pgid, status, start_time, log_file, total_lines, last_polled_line, meta_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, CURRENT_TIMESTAMP)
                """, (job_id, job_type, command_or_code, pid, pgid, status, start_time, log_file, meta_str))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error recording unified job ({str(e)})")

    async def update_unified_job_status(
        self,
        job_id: str,
        status: str,
        exit_code: Optional[int] = None,
        end_time: Optional[float] = None,
        total_lines: Optional[int] = None
    ):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()

                duration = None
                if end_time is not None:
                    async with self.db.execute("SELECT start_time FROM unified_jobs WHERE job_id = ?", (job_id,)) as cur:
                        row = await cur.fetchone()
                        if row and row[0]:
                            duration = max(0.0, end_time - row[0])

                query = "UPDATE unified_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP"
                params = [status]

                if exit_code is not None:
                    query += ", exit_code = ?"
                    params.append(exit_code)
                if end_time is not None:
                    query += ", end_time = ?"
                    params.append(end_time)
                if duration is not None:
                    query += ", duration = ?"
                    params.append(duration)
                if total_lines is not None:
                    query += ", total_lines = ?"
                    params.append(total_lines)

                query += " WHERE job_id = ?"
                params.append(job_id)

                await self.db.execute(query, tuple(params))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error updating unified job status ({str(e)})")

    async def update_unified_job_polling(self, job_id: str, total_lines: int, last_polled_line: int):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    UPDATE unified_jobs
                    SET total_lines = ?, last_polled_line = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (total_lines, last_polled_line, job_id))
                await self.db.commit()
            except Exception as e:
                logger.debug(f"Polling watermark notice: {str(e)}")

    async def get_unified_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM unified_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    item = dict(zip(cols, row))
                    if item.get("meta_json"):
                        try:
                            item["meta"] = json.loads(item["meta_json"])
                        except Exception:
                            item["meta"] = {}
                    return item
            except Exception as e:
                logger.warning(f"Error fetching unified job '{job_id}': {str(e)}")
                return None

    async def list_unified_jobs(
        self,
        status_filter: Optional[str] = None,
        job_type_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM unified_jobs WHERE 1=1"
        params = []

        if status_filter and status_filter.upper() != "ALL":
            query += " AND status = ?"
            params.append(status_filter.upper().strip())

        if job_type_filter and job_type_filter.upper() != "ALL":
            query += " AND job_type = ?"
            params.append(job_type_filter.lower().strip())

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error listing unified jobs: {str(e)}")
                return []

    async def delete_unified_job(self, job_id: str) -> bool:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM unified_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting unified job '{job_id}': {str(e)}")
                return False

    async def clear_unified_jobs(self, status_filter: Optional[str] = None) -> int:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                if status_filter and status_filter.upper() != "ALL":
                    query = "DELETE FROM unified_jobs WHERE status = ?"
                    params = (status_filter.upper().strip(),)
                else:
                    query = "DELETE FROM unified_jobs WHERE status != 'RUNNING'"
                    params = ()
                async with self.db.execute(query, params) as cur:
                    await self.db.commit()
                    return cur.rowcount
            except Exception as e:
                logger.warning(f"Error clearing unified jobs: {str(e)}")
                return 0

    async def reconcile_stale_unified_jobs(self) -> int:
        """Audits running unified jobs against active OS PID liveness."""
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT job_id, pid FROM unified_jobs WHERE status = 'RUNNING'") as cur:
                    running = await cur.fetchall()

                stale_count = 0
                now = time.time()

                for j_id, pid in running:
                    alive = False
                    if pid:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except OSError:
                            alive = False
                    if not alive:
                        await self.update_unified_job_status(job_id=j_id, status="STALE", exit_code=-1, end_time=now)
                        stale_count += 1

                return stale_count
            except Exception as e:
                logger.warning(f"Error reconciling stale unified jobs: {str(e)}")
                return 0

    # =========================================================================
    # 10. LEGACY BACKGROUND BASH JOBS & TOOLS
    # =========================================================================
    async def record_background_job(
        self,
        job_id: str,
        command: str,
        pid: int,
        pgid: int,
        status: str,
        start_time: float,
        log_file: str,
        meta_dict: Optional[dict] = None
    ):
        meta_str = json.dumps(clean_for_json(meta_dict), ensure_ascii=False) if meta_dict else None
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO background_jobs (
                        job_id, command, pid, pgid, status, start_time, log_file, total_lines, last_polled_line, meta_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, CURRENT_TIMESTAMP)
                """, (job_id, command, pid, pgid, status, start_time, log_file, meta_str))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error recording background job ({str(e)})")

    async def update_background_job_status(
        self,
        job_id: str,
        status: str,
        exit_code: Optional[int] = None,
        end_time: Optional[float] = None,
        total_lines: Optional[int] = None
    ):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()

                duration = None
                if end_time is not None:
                    async with self.db.execute("SELECT start_time FROM background_jobs WHERE job_id = ?", (job_id,)) as cur:
                        row = await cur.fetchone()
                        if row and row[0]:
                            duration = max(0.0, end_time - row[0])

                query = "UPDATE background_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP"
                params = [status]

                if exit_code is not None:
                    query += ", exit_code = ?"
                    params.append(exit_code)
                if end_time is not None:
                    query += ", end_time = ?"
                    params.append(end_time)
                if duration is not None:
                    query += ", duration = ?"
                    params.append(duration)
                if total_lines is not None:
                    query += ", total_lines = ?"
                    params.append(total_lines)

                query += " WHERE job_id = ?"
                params.append(job_id)

                await self.db.execute(query, tuple(params))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error updating job status for '{job_id}' ({str(e)})")

    async def update_background_job_polling(self, job_id: str, total_lines: int, last_polled_line: int):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    UPDATE background_jobs
                    SET total_lines = ?, last_polled_line = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (total_lines, last_polled_line, job_id))
                await self.db.commit()
            except Exception as e:
                logger.debug(f"Polling watermark update notice: {str(e)}")

    async def get_background_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM background_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    item = dict(zip(cols, row))
                    if item.get("meta_json"):
                        try:
                            item["meta"] = json.loads(item["meta_json"])
                        except Exception:
                            item["meta"] = {}
                    return item
            except Exception as e:
                logger.warning(f"Error fetching background job '{job_id}': {str(e)}")
                return None

    async def list_background_jobs(self, status_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT * FROM background_jobs"
        params = []
        if status_filter and status_filter.upper() != "ALL":
            query += " WHERE status = ?"
            params.append(status_filter.upper().strip())
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error listing background jobs: {str(e)}")
                return []

    async def delete_background_job(self, job_id: str) -> bool:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM background_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting background job '{job_id}': {str(e)}")
                return False

    async def clear_background_jobs(self, status_filter: Optional[str] = None) -> int:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                if status_filter and status_filter.upper() != "ALL":
                    query = "DELETE FROM background_jobs WHERE status = ?"
                    params = (status_filter.upper().strip(),)
                else:
                    query = "DELETE FROM background_jobs WHERE status != 'RUNNING'"
                    params = ()

                async with self.db.execute(query, params) as cur:
                    await self.db.commit()
                    return cur.rowcount
            except Exception as e:
                logger.warning(f"Error clearing background jobs: {str(e)}")
                return 0

    async def reconcile_stale_jobs(self) -> int:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT job_id, pid FROM background_jobs WHERE status = 'RUNNING'") as cur:
                    running_rows = await cur.fetchall()

                stale_count = 0
                now = time.time()

                for j_id, pid in running_rows:
                    is_alive = False
                    if pid:
                        try:
                            os.kill(pid, 0)
                            is_alive = True
                        except OSError:
                            is_alive = False

                    if not is_alive:
                        await self.update_background_job_status(job_id=j_id, status="STALE", exit_code=-1, end_time=now)
                        stale_count += 1

                return stale_count
            except Exception as e:
                logger.warning(f"Error reconciling stale jobs: {str(e)}")
                return 0

    # Legacy Tool Jobs
    async def record_tool_job(self, job_id: str, tool_name: str, tool_args_json: str, description: str, status: str, start_time: float, log_file: str):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO background_tool_jobs (
                        job_id, tool_name, tool_args_json, description, status, start_time, log_file, total_lines, last_polled_line, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, CURRENT_TIMESTAMP)
                """, (job_id, tool_name, tool_args_json, description, status, start_time, log_file))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error recording background tool job ({str(e)})")

    async def update_tool_job_status(self, job_id: str, status: str, result_preview: Optional[str] = None, end_time: Optional[float] = None, total_lines: Optional[int] = None):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()

                duration = None
                if end_time is not None:
                    async with self.db.execute("SELECT start_time FROM background_tool_jobs WHERE job_id = ?", (job_id,)) as cur:
                        row = await cur.fetchone()
                        if row and row[0]:
                            duration = max(0.0, end_time - row[0])

                query = "UPDATE background_tool_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP"
                params = [status]

                if result_preview is not None:
                    query += ", result_preview = ?"
                    params.append(result_preview[:2000])
                if end_time is not None:
                    query += ", end_time = ?"
                    params.append(end_time)
                if duration is not None:
                    query += ", duration = ?"
                    params.append(duration)
                if total_lines is not None:
                    query += ", total_lines = ?"
                    params.append(total_lines)

                query += " WHERE job_id = ?"
                params.append(job_id)

                await self.db.execute(query, tuple(params))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error updating tool job status for '{job_id}' ({str(e)})")

    async def update_tool_job_polling(self, job_id: str, total_lines: int, last_polled_line: int):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    UPDATE background_tool_jobs
                    SET total_lines = ?, last_polled_line = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (total_lines, last_polled_line, job_id))
                await self.db.commit()
            except Exception as e:
                logger.debug(f"Tool job polling update notice: {str(e)}")

    async def get_tool_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM background_tool_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
            except Exception as e:
                logger.warning(f"Error fetching tool job '{job_id}': {str(e)}")
                return None

    async def list_tool_jobs(self, status_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT * FROM background_tool_jobs"
        params = []
        if status_filter and status_filter.upper() != "ALL":
            query += " WHERE status = ?"
            params.append(status_filter.upper().strip())
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query, tuple(params)) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error listing tool jobs: {str(e)}")
                return []

    async def delete_tool_job(self, job_id: str) -> bool:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM background_tool_jobs WHERE job_id = ?", (job_id.strip(),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting tool job '{job_id}': {str(e)}")
                return False

    async def clear_tool_jobs(self, status_filter: Optional[str] = None) -> int:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                if status_filter and status_filter.upper() != "ALL":
                    query = "DELETE FROM background_tool_jobs WHERE status = ?"
                    params = (status_filter.upper().strip(),)
                else:
                    query = "DELETE FROM background_tool_jobs WHERE status != 'RUNNING'"
                    params = ()

                async with self.db.execute(query, params) as cur:
                    await self.db.commit()
                    return cur.rowcount
            except Exception as e:
                logger.warning(f"Error clearing tool jobs: {str(e)}")
                return 0

    async def reconcile_stale_tool_jobs(self) -> int:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                now = time.time()
                async with self.db.execute("UPDATE background_tool_jobs SET status = 'STALE', end_time = ?, updated_at = CURRENT_TIMESTAMP WHERE status = 'RUNNING'", (now,)) as cur:
                    await self.db.commit()
                    return cur.rowcount
            except Exception as e:
                logger.warning(f"Error reconciling stale tool jobs: {str(e)}")
                return 0

    # Custom Tools Storage
    async def record_custom_tool(self, name: str, description: str, code: str, file_path: str, parameters_schema_json=None, is_active=True):
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                await self.db.execute("""
                    INSERT OR REPLACE INTO custom_tools (name, description, parameters_schema_json, code, file_path, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, description, parameters_schema_json, code, file_path, 1 if is_active else 0))
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Error recording custom tool: {str(e)}")

    async def get_custom_tool(self, name: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("SELECT * FROM custom_tools WHERE name = ?", (name.strip(),)) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
            except Exception as e:
                logger.warning(f"Error fetching custom tool '{name}': {str(e)}")
                return None

    async def list_custom_tools_records(self, active_only: bool = True) -> List[Dict[str, Any]]:
        query = "SELECT * FROM custom_tools"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY name ASC"

        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute(query) as cur:
                    rows = await cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                logger.warning(f"Error listing custom tools: {str(e)}")
                return []

    async def delete_custom_tool_record(self, name: str) -> bool:
        async with self._lock:
            try:
                if not self.db:
                    await self.connect()
                async with self.db.execute("DELETE FROM custom_tools WHERE name = ?", (name.strip(),)) as cur:
                    await self.db.commit()
                    return cur.rowcount > 0
            except Exception as e:
                logger.warning(f"Error deleting custom tool: {str(e)}")
                return False

    async def close(self):
        async with self._lock:
            if self.db:
                try:
                    await self.checkpoint()
                    await self.db.close()
                except Exception:
                    pass
                finally:
                    self.db = None
