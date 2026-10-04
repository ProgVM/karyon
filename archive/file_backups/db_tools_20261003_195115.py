# karyon_agent_runtime/tools/db_tools.py
"""
===============================================================================
DATABASE PERSISTENCE, MEMORY, SNAPSHOT & REPLICATION ENGINE (v22.0 MASTER)
Full-Text Search, Scientific Ledger, Snapshot Creation/Renaming/Deletion,
Detailed Diagnostics, Mutual Continuity Switching, and Background GitHub Sync.
Includes SQLite VACUUM Defragmentation, Memory Search, and JSON Database Export.
Seamlessly Synchronized with Multi-Tier Persistent Pointer Engine (active_db.txt).
===============================================================================
"""

import re
import json
import asyncio
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import karyon_agent_runtime.config as config

logger = logging.getLogger("ProxyAgent.DBTools")


def _get_db():
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    if not agent or not agent.db:
        from karyon_agent_runtime.db_manager import AgentDBManager
        return AgentDBManager()
    return agent.db


def _resolve_db_file(db_name: Optional[str] = None) -> Path:
    """Resolves database file path in agent_data/ directory via config.resolve_database_path."""
    return config.resolve_database_path(db_name)


# === 1. GIT REPOSITORY SYNC ===
async def sync_agent_database(commit_message: str = "chore(db): auto-sync persistent empirical ledger, memory and pointer") -> str:
    """
    Forces SQLite commit to flush all data into disk, stages agent_data/ (including active_db.txt pointer),
    and pushes the database state to the private 'karyon_agent_runtime' repository
    to prevent loss of memory if the Kaggle notebook session disconnects or resets.

    Args:
        commit_message: Git commit message for the database snapshot.
    """
    try:
        db = _get_db()
        await db.checkpoint()

        agent_dir = config.BASE_DIR
        clean_msg = commit_message.replace('"', '\\"')
        cmd = f'git add agent_data/ && git commit -m "{clean_msg}" && git push origin main'
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(agent_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode('utf-8', errors='replace').strip()
        err = stderr.decode('utf-8', errors='replace').strip()
        res = f"{out}\n{err}".strip()
        logger.info(f"Database Remote Sync Result: {res}")
        return f"=== Database Remote Git Sync Result ===\n{res if res else '[Synced cleanly with remote main branch]'}"
    except Exception as e:
        return f"Error syncing database to remote repository: {str(e)}"


# === 2. BACKUP & SNAPSHOT OPERATIONS ===
async def list_database_backups() -> str:
    """Lists all local database files, snapshots, and emergency backups in agent_data/."""
    agent_data_dir = config.AGENT_DATA_DIR
    if not agent_data_dir.exists():
        return "No agent_data directory found."

    raw_files = sorted(list(agent_data_dir.glob("*.db*")), key=lambda p: p.stat().st_mtime, reverse=True)
    files = [
        f for f in raw_files
        if not any(f.name.endswith(sfx) for sfx in ["-wal", "-shm", "-journal", "-lock"])
    ]
    if not files:
        return "No database files found in agent_data/."

    active_path = config.get_active_db_path().resolve()
    pointer_info = (
        f"Active Pointer: `{config.ACTIVE_DB_POINTER_FILE.name}` -> `{active_path.name}`"
        if config.ACTIVE_DB_POINTER_FILE.exists()
        else f"Default Fallback: `{active_path.name}` (No pointer file created yet)"
    )

    lines = [
        f"=== Agent Database Files in {agent_data_dir.name}/ ({len(files)} files) ===",
        f"- {pointer_info}\n"
    ]
    for f in files:
        size_kb = f.stat().st_size / 1024
        is_active = (f.resolve() == active_path)
        tag = "[ACTIVE DB] 🌟" if is_active else "[CORRUPTED BACKUP] ⚠️" if "corrupted" in f.name else "[SNAPSHOT / DB] 💾"
        lines.append(f"• {tag} `{f.name}` ({size_kb:.1f} KB)")
    return "\n".join(lines)


async def create_database_snapshot(snapshot_name: Optional[str] = None, source_db: Optional[str] = None) -> str:
    """Creates a named or timestamped snapshot copy of the current active database or a specified database."""
    try:
        src_path = config.resolve_database_path(source_db) if source_db else config.get_active_db_path().resolve()
        if not src_path.exists():
            return f"Error: Source database '{src_path.name}' does not exist at `{src_path}`."

        db = _get_db()
        if src_path == config.get_active_db_path().resolve():
            await db.checkpoint()

        if not snapshot_name:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"{src_path.stem}_snapshot_{timestamp_str}.db"
        elif not snapshot_name.endswith(".db"):
            snapshot_name = f"{snapshot_name}.db"

        dst_path = config.resolve_database_path(snapshot_name)
        shutil.copy2(str(src_path), str(dst_path))

        size_kb = dst_path.stat().st_size / 1024
        msg = f"Success: Database snapshot created: '{dst_path.name}' ({size_kb:.1f} KB) from source '{src_path.name}'."
        logger.info(msg)
        return msg
    except Exception as e:
        return f"Error creating database snapshot: {str(e)}"


async def rename_database_snapshot(old_name: str, new_name: str) -> str:
    """Renames an existing database snapshot file or the active database with atomic pointer update."""
    try:
        src_path = config.resolve_database_path(old_name)
        if not src_path.exists():
            return f"Error: Database file '{old_name}' not found at `{src_path}`."

        dst_path = config.resolve_database_path(new_name)
        if dst_path.exists() and dst_path != src_path:
            return f"Error: Target database name '{dst_path.name}' already exists in {config.AGENT_DATA_DIR}."

        is_active = (src_path == config.get_active_db_path().resolve())
        db = _get_db()

        if is_active:
            await db.close()

        src_path.rename(dst_path)

        if is_active:
            await db.switch_active_database(dst_path)
            await db.save_turn(
                "system",
                f"[System Event: Active database was renamed from '{src_path.name}' to '{dst_path.name}'. Pointer updated in active_db.txt]"
            )
            asyncio.create_task(sync_agent_database(f"chore(db): rename active database to {dst_path.name}"))

        msg = f"Success: Database '{src_path.name}' renamed to '{dst_path.name}' (Active: {is_active})."
        logger.info(msg)
        return msg
    except Exception as e:
        return f"Error renaming database: {str(e)}"


async def delete_database_snapshot(snapshot_name: str, force: bool = False) -> str:
    """Deletes a specified database snapshot or backup file from agent_data/."""
    try:
        target_path = config.resolve_database_path(snapshot_name)
        if not target_path.exists():
            return f"Error: Database file '{snapshot_name}' not found at `{target_path}`."

        is_active = (target_path == config.get_active_db_path().resolve())
        if is_active and not force:
            return f"Error: Cannot delete the active database '{target_path.name}' without force=True. Switch databases first."

        db = _get_db()
        if is_active:
            default_path = config.resolve_database_path("agent_state.db")
            await db.switch_active_database(default_path)
            await db.save_turn(
                "system",
                f"[System Event: Active database '{target_path.name}' was deleted. Active DB reverted to '{default_path.name}']"
            )

        target_path.unlink()
        for suffix in ["-wal", "-shm", "-journal", "-lock"]:
            side_file = Path(str(target_path) + suffix)
            if side_file.exists():
                side_file.unlink()

        if is_active:
            asyncio.create_task(sync_agent_database(f"chore(db): deleted active database {target_path.name}"))

        msg = f"Success: Database file '{target_path.name}' deleted (Active reverted: {is_active})."
        logger.info(msg)
        return msg
    except Exception as e:
        return f"Error deleting database: {str(e)}"


# === 3. DEEP DATABASE DIAGNOSTICS & RECIPROCAL SWITCHING ===
async def get_database_info(db_name: Optional[str] = None) -> str:
    """Returns deep diagnostics, telemetry, turn statistics, experiment counts, and integrity status for current or specified DB."""
    try:
        target_path = config.resolve_database_path(db_name) if db_name else config.get_active_db_path().resolve()
        if not target_path.exists():
            return f"Error: Database file '{target_path.name}' not found at `{target_path}`."

        active_path = config.get_active_db_path().resolve()
        is_active = (target_path == active_path)
        stat = target_path.stat()
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")

        # Read-only connection for non-blocking inspection
        conn = sqlite3.connect(f"file:{str(target_path)}?mode=ro", uri=True, timeout=5.0)
        cursor = conn.cursor()

        # Integrity check
        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]

        # Journal mode
        cursor.execute("PRAGMA journal_mode;")
        journal = cursor.fetchone()[0]

        # Turns stats
        turns_summary = {}
        total_turns = 0
        try:
            cursor.execute("SELECT role, count(*) FROM turns GROUP BY role;")
            for r, cnt in cursor.fetchall():
                turns_summary[r] = cnt
                total_turns += cnt
        except Exception:
            pass

        # Turns archive stats
        total_archived = 0
        try:
            cursor.execute("SELECT count(*) FROM turns_archive;")
            total_archived = cursor.fetchone()[0]
        except Exception:
            pass

        # Ledger stats
        verdicts_summary = {}
        total_exps = 0
        latest_exp = "None"
        try:
            cursor.execute("SELECT verdict, count(*) FROM empirical_ledger GROUP BY verdict;")
            for v, cnt in cursor.fetchall():
                verdicts_summary[v] = cnt
                total_exps += cnt
            cursor.execute("SELECT exp_id, timestamp, final_loss FROM empirical_ledger ORDER BY timestamp DESC LIMIT 1;")
            row = cursor.fetchone()
            if row:
                latest_exp = f"{row[0]} (Loss: {row[2]}, Date: {row[1]})"
        except Exception:
            pass

        # Memory stats
        memory_summary = {}
        total_mem = 0
        try:
            cursor.execute("SELECT category, count(*) FROM memory GROUP BY category;")
            for cat, cnt in cursor.fetchall():
                memory_summary[cat] = cnt
                total_mem += cnt
        except Exception:
            pass

        # Background Jobs & Custom tools
        total_bg_jobs = 0
        total_custom_tools = 0
        try:
            cursor.execute("SELECT count(*) FROM background_jobs;")
            total_bg_jobs = cursor.fetchone()[0]
        except Exception:
            pass
        try:
            cursor.execute("SELECT count(*) FROM custom_tools;")
            total_custom_tools = cursor.fetchone()[0]
        except Exception:
            pass

        conn.close()

        turns_str = ", ".join(f"{k}: {v}" for k, v in turns_summary.items()) or "0"
        verdicts_str = ", ".join(f"{k}: {v}" for k, v in verdicts_summary.items()) or "0"
        memory_str = ", ".join(f"{k}: {v}" for k, v in memory_summary.items()) or "0"

        pointer_status = (
            f"Pointer: `{config.ACTIVE_DB_POINTER_FILE.name}` -> `{active_path.name}`"
            if config.ACTIVE_DB_POINTER_FILE.exists()
            else "Pointer: Fallback default"
        )

        lines = [
            f"=== Database Diagnostics & Telemetry: {target_path.name} ===",
            f"- Status: {'[ACTIVE SYSTEM DATABASE 🌟]' if is_active else '[BACKUP / SNAPSHOT 💾]'}",
            f"- Persistence: {pointer_status}",
            f"- File Size: {size_kb:.1f} KB ({stat.st_size} bytes)",
            f"- Path: {target_path}",
            f"- Created: {ctime} | Modified: {mtime}",
            f"- SQLite Integrity: {integrity.upper()} | Journal Mode: {journal.upper()}",
            f"- Active Dialogue Turns: {total_turns} ({turns_str}) | Archived: {total_archived}",
            f"- Recorded Experiments: {total_exps} ({verdicts_str})",
            f"- Latest Recorded Experiment: {latest_exp}",
            f"- Persistent Memory Items: {total_mem} ({memory_str})",
            f"- Background Jobs Recorded: {total_bg_jobs} | Custom Tools: {total_custom_tools}"
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving database diagnostics: {str(e)}"


async def switch_database(target_db_name: str, create_if_missing: bool = False) -> str:
    """
    Switches active database to another existing snapshot or newly initialized database,
    logging reciprocal transition events in BOTH databases to maintain unbroken epistemic continuity,
    and updates the persistent pointer (active_db.txt) so the selection survives restarts.
    """
    try:
        old_db_path = config.get_active_db_path().resolve()
        target_path = config.resolve_database_path(target_db_name)

        if not target_path.exists() and not create_if_missing:
            return f"Error: Target database '{target_db_name}' not found at `{target_path}`. Set create_if_missing=True to initialize a new one."

        if target_path == old_db_path:
            return f"Database '{target_path.name}' is already active (Pointer: `{config.ACTIVE_DB_POINTER_FILE.name}`)."

        db = _get_db()
        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Log transition event in previous active DB before closing
        try:
            if db.db:
                await db.save_turn(
                    "system",
                    f"[System Event: Database switch initiated. Deactivating '{old_db_path.name}' and activating '{target_path.name}' at {timestamp_now}]"
                )
                await db.checkpoint()
        except Exception as e:
            logger.debug(f"Exit log notice for {old_db_path.name}: {str(e)}")

        # 2. Atomically switch database in db manager (updates pointer and config.DB_PATH)
        await db.switch_active_database(target_path)

        # 3. Log activation event in new active DB
        await db.save_turn(
            "system",
            f"[System Event: Database activated. Switched from '{old_db_path.name}' to '{target_path.name}' at {timestamp_now}. Epistemic continuity preserved across sessions and restarts.]"
        )

        # 4. Trigger auto-sync to git so the active_db.txt pointer and new DB state are pushed
        asyncio.create_task(sync_agent_database(f"chore(db): switch active database to {target_path.name}"))

        msg = (
            f"=== Database Successfully Switched ===\n"
            f"- Previous Database : '{old_db_path.name}' (logged deactivation event)\n"
            f"- Active Database   : '{target_path.name}' (logged activation event)\n"
            f"- Persistent Pointer: `{config.ACTIVE_DB_POINTER_FILE.name}` updated to `{target_path.name}`\n"
            f"- Epistemic continuity and persistent pointer preserved across future restarts."
        )
        logger.info(msg)
        return msg
    except Exception as e:
        return f"Error switching database: {str(e)}"


async def restore_database_backup(backup_filename: str) -> str:
    """Restores a previous backup snapshot as the active database."""
    return await switch_database(backup_filename, create_if_missing=False)


# === 4. SCIENTIFIC LEDGER & LATEST EXPERIMENT RESOLVER ===
async def get_latest_experiment() -> str:
    """
    Finds the latest conducted experiment across both SQLite DB and files on disk.
    Returns summary metrics, verdict, and the next recommended experiment ID (e.g. EXP-71).
    """
    try:
        db = _get_db()
        db_exps = await db.get_all_experiments(order="desc", limit=5)

        # Scan experiments folder on disk for latest file index
        exp_dir = config.PROJECT_ROOT / "experiments"
        disk_exp_nums = []
        if exp_dir.exists():
            for f in exp_dir.glob("exp_*.py"):
                m = re.search(r"exp_(\d+)", f.name)
                if m:
                    disk_exp_nums.append(int(m.group(1)))

        latest_disk_num = max(disk_exp_nums) if disk_exp_nums else 0

        latest_db = db_exps[0] if db_exps else None
        db_num = 0
        if latest_db:
            m = re.search(r"(\d+)", latest_db["exp_id"])
            if m:
                db_num = int(m.group(1))

        highest_num = max(latest_disk_num, db_num)
        next_num = highest_num + 1

        active_db_name = config.get_active_db_path().name
        lines = [f"=== LATEST EXPERIMENT STATUS & NEXT INDEX (Active DB: {active_db_name}) ==="]
        lines.append(f"- Highest Completed Experiment Number: EXP-{highest_num}")
        lines.append(f"- Recommended Next Experiment ID: EXP-{next_num}")

        if latest_db:
            lines.append(f"\n--- Latest Ledger Record in '{active_db_name}': {latest_db['exp_id']} ---")
            lines.append(f"- Verdict: {latest_db['verdict']}")
            lines.append(f"- Final Loss: {latest_db.get('final_loss')}")
            lines.append(f"- Hypothesis: {latest_db['hypothesis']}")
            lines.append(f"- Delta: {latest_db['architecture_delta']}")
            if latest_db.get("metrics"):
                lines.append(f"- Key Metrics: {json.dumps(latest_db['metrics'], ensure_ascii=False)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error detecting latest experiment: {str(e)}"


async def get_experiment_history(search_query: str = None, exp_id: str = None, verdict_filter: str = None, order: str = "desc", limit: int = 20) -> str:
    """Searches and queries the persistent empirical scientific ledger."""
    try:
        db = _get_db()
        active_db_name = config.get_active_db_path().name
        if exp_id:
            exp = await db.get_experiment(exp_id.strip())
            if not exp:
                return f"Error: Experiment matching '{exp_id}' not found in active ledger ({active_db_name})."

            lines = [
                f"=== Detailed Record for {exp['exp_id']} ({active_db_name}) ===",
                f"- Timestamp: {exp['timestamp']}",
                f"- Hypothesis: {exp['hypothesis']}",
                f"- Architecture Delta: {exp['architecture_delta']}",
                f"- Verdict: {exp['verdict']}",
                f"- Final Speech Loss: {exp.get('final_loss')}",
                f"- Metrics: {json.dumps(exp.get('metrics', {}), indent=2, ensure_ascii=False)}",
                f"- Config: {json.dumps(exp.get('config', {}), indent=2, ensure_ascii=False)}",
                f"- Notes: {exp.get('notes') or 'None'}"
            ]
            return "\n".join(lines)

        experiments = await db.get_all_experiments(search_query=search_query, verdict_filter=verdict_filter, order=order, limit=limit)
        if not experiments:
            return f"No experiments found matching search_query='{search_query}', verdict='{verdict_filter}' in '{active_db_name}'."

        title = f"=== Empirical Ledger (DB: '{active_db_name}', Search: '{search_query or 'ALL'}', Order: {order.upper()}, Found: {len(experiments)}) ==="
        lines = [title]
        for exp in experiments:
            metrics_brief = ", ".join(f"{k}={v}" for k, v in list(exp.get("metrics", {}).items())[:4])
            lines.append(f"- [{exp['exp_id']}] {exp['verdict']} | Loss: {exp.get('final_loss')} | {exp['hypothesis'][:45]}... | ({metrics_brief})")

        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving experiment history: {str(e)}"


async def record_experiment_result(
    exp_id: str,
    hypothesis: str,
    architecture_delta: str,
    verdict: str,
    final_loss: float = None,
    metrics: Dict[str, Any] = None,
    config_params: Dict[str, Any] = None,
    notes: str = None
) -> str:
    """Records a completed KEP experiment into SQLite and triggers cloud sync."""
    try:
        db = _get_db()
        await db.record_experiment(
            exp_id=exp_id.strip(),
            hypothesis=hypothesis.strip(),
            delta=architecture_delta.strip(),
            verdict=verdict.strip(),
            final_loss=final_loss,
            metrics=metrics or {},
            config_dict=config_params or {},
            notes=notes
        )
        asyncio.create_task(sync_agent_database(f"chore(ledger): record {exp_id} ({verdict})"))
        logger.info(f"Experiment '{exp_id}' ({verdict}) recorded in empirical ledger.")
        return f"Success: Experiment '{exp_id}' recorded in empirical ledger with verdict '{verdict}'. (Auto-synced to remote git)."
    except Exception as e:
        return f"Error recording experiment: {str(e)}"


# === 5. KEY-VALUE MEMORY MANAGEMENT ===
async def set_agent_memory(key: str, value: str, category: str = "general") -> str:
    """Stores a persistent key-value memory item in SQLite database."""
    try:
        db = _get_db()
        await db.set_memory(key.strip(), value.strip(), category=category.strip())
        return f"Success: Key '{key}' saved in persistent memory under category '{category}'."
    except Exception as e:
        return f"Error saving to persistent memory: {str(e)}"


async def get_agent_memory(key: str) -> str:
    """Retrieves a persistent memory string from SQLite database by key."""
    try:
        db = _get_db()
        val = await db.get_memory(key.strip())
        if val is not None:
            return f"Memory Key '{key}':\n{val}"
        return f"Memory Key '{key}' not found."
    except Exception as e:
        return f"Error reading persistent memory: {str(e)}"


async def list_agent_memory(category: str = None) -> str:
    """Lists all stored persistent memory keys and values."""
    try:
        db = _get_db()
        items = await db.list_memory(category=category)
        if not items:
            return "No memory records found."

        lines = [f"=== Persistent Agent Memory ({len(items)} items) ==="]
        for it in items:
            val_preview = it['value'].replace('\n', ' ')[:60]
            lines.append(f"- [{it['category']}] '{it['key']}': {val_preview}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing persistent memory: {str(e)}"


async def delete_agent_memory(key: str) -> bool:
    """Deletes a persistent memory item from SQLite database by key."""
    try:
        db = _get_db()
        deleted = await db.delete_memory(key.strip())
        if deleted:
            return f"Success: Memory key '{key}' deleted."
        return f"Memory key '{key}' not found."
    except Exception as e:
        return f"Error deleting memory key '{key}': {str(e)}"


async def search_persistent_memory(query: str, category: Optional[str] = None) -> str:
    """Searches memory keys and values by regular expression or substring."""
    db = _get_db()
    items = await db.list_memory(category=category)
    pattern = re.compile(query, re.IGNORECASE)
    matches = [it for it in items if pattern.search(it["key"]) or pattern.search(it["value"])]
    if not matches:
        return f"No memory entries matched query '{query}'."

    lines = [f"=== Memory Search Results for '{query}' ({len(matches)} matches) ==="]
    for it in matches:
        lines.append(f"• **`{it['key']}`** [{it['category']}]:\n  {it['value'][:150]}...")
    return "\n".join(lines)


# === 6. VACUUM & JSON EXPORT ===
async def vacuum_and_optimize_database() -> str:
    """Defragments SQLite database, reclaims deleted storage, and runs PRAGMA optimize."""
    db = _get_db()
    await db.checkpoint()
    t0_size = db.db_path.stat().st_size if db.db_path.exists() else 0

    try:
        if db.db:
            await db.db.execute("VACUUM;")
            await db.db.execute("PRAGMA optimize;")
            await db.checkpoint()
        t1_size = db.db_path.stat().st_size if db.db_path.exists() else 0
        reclaimed_kb = max(0.0, (t0_size - t1_size) / 1024)
        return f"=== SQLite Database VACUUM Complete ({db.db_path.name}) ===\n- Pre-size: {t0_size / 1024:.1f} KB | Post-size: {t1_size / 1024:.1f} KB\n- Reclaimed: {reclaimed_kb:.1f} KB dead storage"
    except Exception as e:
        return f"Error executing database vacuum: {str(e)}"


async def export_empirical_database_json(output_path: str = "agent_data/database_export.json") -> str:
    """Exports all dialogue turns, empirical ledger, memory, background jobs, and custom tools into JSON."""
    db = _get_db()
    dest = (config.PROJECT_ROOT / output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    turns = await db.get_recent_turns(limit=500, include_tools=True)
    ledger = await db.get_all_experiments(limit=500)
    memory = await db.list_memory()
    jobs = await db.list_background_jobs(limit=200)
    tool_jobs = await db.list_tool_jobs(limit=200)
    custom_tools = await db.list_custom_tools_records(active_only=False)

    export_obj = {
        "database_file": db.db_path.name,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "turns": turns,
        "empirical_ledger": ledger,
        "memory": memory,
        "background_jobs": jobs,
        "background_tool_jobs": tool_jobs,
        "custom_tools": custom_tools
    }

    dest.write_text(json.dumps(export_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = dest.stat().st_size / 1024
    return f"Success: Database '{db.db_path.name}' exported to `{output_path}` ({size_kb:.1f} KB, {len(turns)} turns, {len(ledger)} experiments, {len(custom_tools)} custom tools)."
