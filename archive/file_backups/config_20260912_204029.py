# karyon_agent_runtime/config.py
"""
===============================================================================
KAGGLE / COLAB CORE AGENT RUNTIME CONFIGURATION MATRIX (v30.2 MASTER)
Optimized for 250,000 TPM Free-Tier Quota Resilience across 36 Independent Projects,
Lossless 40k Context Compaction, SQLite Persistent Settings Synchronization,
and Hierarchical Multi-Agent Cortical Orchestration (Laminar Specialization).
Enhanced with Resilient Model Cascades (Auto-Fallback from 3.8 to 3.7/2.5/2.0).
===============================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger("ProxyAgent.Config")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if not (PROJECT_ROOT / "karyon_agent.py").exists() and (BASE_DIR / "karyon_agent.py").exists():
    PROJECT_ROOT = BASE_DIR

load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)

AGENT_DATA_DIR = BASE_DIR / "agent_data"
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_DB_POINTER_FILE = AGENT_DATA_DIR / "active_db.txt"
DEFAULT_DB_FILENAME = "agent_state.db"


def resolve_database_path(db_identifier: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolves any database name or path into a canonical Path inside AGENT_DATA_DIR.
    Handles bare filenames, relative paths, and prevents redundant nested directory prefixes.
    """
    if not db_identifier or str(db_identifier).strip() in ["", "current", "active", "default"]:
        return get_active_db_path()

    clean_str = str(db_identifier).strip()
    path_obj = Path(clean_str)

    if path_obj.is_absolute():
        resolved = path_obj
    else:
        parts = list(path_obj.parts)
        if parts and parts[0] == "agent_data":
            parts = parts[1:]
        clean_name = Path(*parts) if parts else Path(clean_str).name
        resolved = AGENT_DATA_DIR / clean_name

    if not resolved.suffix:
        resolved = resolved.with_suffix(".db")

    return resolved.resolve()


def get_active_db_path() -> Path:
    """
    Resolves the active SQLite database path with multi-tier persistence:
    1. Environment variables (AGENT_DB_NAME, AGENT_DB_PATH, DB_PATH)
    2. Persistent disk pointer file (agent_data/active_db.txt)
    3. Fallback default (agent_data/agent_state.db)
    """
    env_target = (
        os.getenv("AGENT_DB_NAME") or
        os.getenv("AGENT_DB_PATH") or
        os.getenv("DB_PATH")
    )
    if env_target and env_target.strip():
        return resolve_database_path(env_target.strip())

    if ACTIVE_DB_POINTER_FILE.exists():
        try:
            content = ACTIVE_DB_POINTER_FILE.read_text(encoding="utf-8").strip()
            if content:
                candidate = Path(content)
                target = candidate if candidate.is_absolute() else AGENT_DATA_DIR / candidate
                if not target.suffix:
                    target = target.with_suffix(".db")
                return target.resolve()
        except Exception:
            pass

    return (AGENT_DATA_DIR / DEFAULT_DB_FILENAME).resolve()


def set_active_db_pointer(db_path: Union[str, Path]) -> Path:
    """
    Persistently writes the active database filename into agent_data/active_db.txt
    and synchronizes config.DB_PATH in working memory.
    """
    global DB_PATH
    canonical_path = resolve_database_path(db_path)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ACTIVE_DB_POINTER_FILE.write_text(canonical_path.name, encoding="utf-8")
    except Exception:
        pass
    DB_PATH = canonical_path
    return canonical_path


DB_PATH = get_active_db_path()

MASTER_DOCS_DIR = AGENT_DATA_DIR / "master_docs"
MASTER_DOCS_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DIR = PROJECT_ROOT / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

BUILD_DIR = PROJECT_ROOT / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

# 36-Key Deduplicated Dynamic Pool (6 Accounts x 6 Projects)
_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_KEYS", "")
GEMINI_API_KEYS = list(dict.fromkeys([k.strip() for k in _raw_keys.split(",") if k.strip()]))
GEMINI_NUM_ACCOUNTS = int(os.getenv("GEMINI_NUM_ACCOUNTS", "6"))

# Flagship Gemini 3.x/2.5 Model Matrix with Guaranteed High-Quota Fallbacks
_raw_models = (
    os.getenv("GEMINI_MODELS", "") or
    os.getenv("GEMINI_MODEL", "") or
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash"
)
_parsed_models = [m.strip() for m in _raw_models.split(",") if m.strip()]
# Guarantee that high-quota survival models exist in the cascade
for _fallback_m in ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
    if _fallback_m not in _parsed_models:
        _parsed_models.append(_fallback_m)
GEMINI_MODELS = _parsed_models

# Dedicated Resilient Summarizer Models Pool
_raw_sum_models = (
    os.getenv("SUMMARIZER_MODELS", "") or
    os.getenv("SUMMARIZER_MODEL", "") or
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-2.5-flash,gemini-2.5-flash-lite"
)
_parsed_sum_models = [m.strip() for m in _raw_sum_models.split(",") if m.strip()]
for _fallback_sm in ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
    if _fallback_sm not in _parsed_sum_models:
        _parsed_sum_models.append(_fallback_sm)
SUMMARIZER_MODELS = _parsed_sum_models
SUMMARIZER_MODEL = SUMMARIZER_MODELS[0] if SUMMARIZER_MODELS else "gemini-3.7-flash"
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "16384"))

# SLIM_PROMPT_MODE: When True, prevents inlining 120k+ codebase tokens into every turn
SLIM_PROMPT_MODE = os.getenv("SLIM_PROMPT_MODE", "true").lower() in ["true", "1", "yes"]

# Free-Tier Quota Hardening & 250k TPM Safeguards (per project)
FREE_TIER_TPM_LIMIT = int(os.getenv("FREE_TIER_TPM_LIMIT", "250000"))
FREE_TIER_RPM_LIMIT = int(os.getenv("FREE_TIER_RPM_LIMIT", "15"))
PREEMPTIVE_TPM_SAFETY_MARGIN = float(os.getenv("PREEMPTIVE_TPM_SAFETY_MARGIN", "0.85"))
MAX_HISTORICAL_TOOL_CHARS = int(os.getenv("MAX_HISTORICAL_TOOL_CHARS", "1200"))

# Adaptive Context Compression Threshold (Hardened to 40k to guarantee <= 250k TPM)
CONTEXT_COMPRESSION_THRESHOLD = int(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "100000"))
RECENT_TURNS_PRESERVE_COUNT = int(os.getenv("RECENT_TURNS_PRESERVE_COUNT", "10"))
AUTO_COMPACT_ON_THRESHOLD = os.getenv("AUTO_COMPACT_ON_THRESHOLD", "true").lower() in ["true", "1", "yes"]

# Pacing & Retry Queue Configuration
INTER_TURN_DELAY = float(os.getenv("INTER_TURN_DELAY", "0.5"))
API_BACKOFF_BASE_DELAY = float(os.getenv("API_BACKOFF_BASE_DELAY", "5.0"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", max(60, len(GEMINI_API_KEYS) * 2)))

# Core Hyperparameters & Scaled Token Thresholds
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_OUTPUT_TOKENS = int(os.getenv("OUTPUT_LENGTH", "65536"))
INPUT_TOKEN_LIMIT = int(os.getenv("INPUT_TOKEN_LIMIT", "1048576"))
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "10000"))

# Thinking & Reasoning Settings
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "HIGH").upper()
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "24576"))

# Safety Settings (BLOCK_NONE across all 4 categories)
SAFETY_BLOCK_NONE = os.getenv("SAFETY_BLOCK_NONE", "true").lower() in ["true", "1", "yes"]

# Multi-Agent Cortical Hierarchy System Configuration
MULTI_AGENT_ENABLED = os.getenv("MULTI_AGENT_ENABLED", "true").lower() in ["true", "1", "yes"]
SWARM_MODE = os.getenv("SWARM_MODE", "true").lower() in ["true", "1", "yes"]
SWARM_MAX_CYCLES_PER_STAGE = int(os.getenv("SWARM_MAX_CYCLES_PER_STAGE", "1"))
DEFAULT_SUBAGENT_MAX_TURNS = int(os.getenv("DEFAULT_SUBAGENT_MAX_TURNS", "30"))
SUBAGENT_COMMUNICATION_ENABLED = os.getenv("SUBAGENT_COMMUNICATION_ENABLED", "true").lower() in ["true", "1", "yes"]

# Proxy Configuration
GEMINI_PROXIES = [p.strip() for p in os.getenv("GEMINI_PROXIES", "").split(",") if p.strip()]
ALL_PROXY = os.getenv("ALL_PROXY", "") or os.getenv("all_proxy", "")

# Networking Limits
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "15.0"))
SCRAPE_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT", "20.0"))
WEB_SEARCH_RESULTS_LIMIT = int(os.getenv("WEB_SEARCH_RESULTS_LIMIT", "8"))
WEB_DEEP_SEARCH_CANDIDATES_LIMIT = int(os.getenv("WEB_DEEP_SEARCH_CANDIDATES_LIMIT", "4"))
WEB_DEEP_SEARCH_CHAR_LIMIT = int(os.getenv("WEB_DEEP_SEARCH_CHAR_LIMIT", "30000"))
SCRAPE_CHAR_LIMIT = int(os.getenv("SCRAPE_CHAR_LIMIT", "25000"))

# Bash & Process Timeouts
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "180.0"))
DEFAULT_BASH_IDLE_TIMEOUT = float(os.getenv("DEFAULT_BASH_IDLE_TIMEOUT", "15.0"))
MAX_BASH_TIMEOUT = float(os.getenv("MAX_BASH_TIMEOUT", "3600.0"))
MAX_BASH_OUTPUT_CHARS = int(os.getenv("MAX_BASH_OUTPUT_CHARS", "60000"))

# Pacific Time Offsets for Daily Reset
PACIFIC_STANDARD_TIME_OFFSET = -8
PACIFIC_DAYLIGHT_TIME_OFFSET = -7
GEMINI_MIN_COOLDOWN_SECONDS = 3

# External Integrations (HF Hub & Notifications)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
GENERIC_WEBHOOK_URL = os.getenv("GENERIC_WEBHOOK_URL", "")


def apply_persistent_settings(settings: Dict[str, Any]):
    """
    Applies persistent runtime settings loaded from the SQLite database
    directly into config working memory, overriding initial environment defaults.
    """
    global TEMPERATURE, TOP_P, MAX_AGENT_TURNS, THINKING_LEVEL, THINKING_BUDGET
    global CONTEXT_COMPRESSION_THRESHOLD, SUMMARIZER_MODEL, SUMMARIZER_MODELS
    global GEMINI_MODELS, SLIM_PROMPT_MODE, API_MAX_RETRIES, INTER_TURN_DELAY
    global MAX_HISTORICAL_TOOL_CHARS, RECENT_TURNS_PRESERVE_COUNT, AUTO_COMPACT_ON_THRESHOLD
    global FREE_TIER_TPM_LIMIT, FREE_TIER_RPM_LIMIT, MULTI_AGENT_ENABLED, SWARM_MODE

    if not settings:
        return

    if "temperature" in settings and settings["temperature"] is not None:
        TEMPERATURE = float(settings["temperature"])
    if "top_p" in settings and settings["top_p"] is not None:
        TOP_P = float(settings["top_p"])
    if "max_agent_turns" in settings and settings["max_agent_turns"] is not None:
        MAX_AGENT_TURNS = int(settings["max_agent_turns"])
    if "thinking_level" in settings and settings["thinking_level"] is not None:
        THINKING_LEVEL = str(settings["thinking_level"]).upper()
    if "thinking_budget" in settings and settings["thinking_budget"] is not None:
        THINKING_BUDGET = int(settings["thinking_budget"])
    if "context_compression_threshold" in settings and settings["context_compression_threshold"] is not None:
        CONTEXT_COMPRESSION_THRESHOLD = int(settings["context_compression_threshold"])
    if "summarizer_model" in settings and settings["summarizer_model"]:
        SUMMARIZER_MODEL = str(settings["summarizer_model"])
    if "summarizer_models" in settings and settings["summarizer_models"]:
        if isinstance(settings["summarizer_models"], list):
            SUMMARIZER_MODELS = [str(m).strip() for m in settings["summarizer_models"] if str(m).strip()]
        elif isinstance(settings["summarizer_models"], str):
            SUMMARIZER_MODELS = [m.strip() for m in settings["summarizer_models"].split(",") if m.strip()]
    if "gemini_models" in settings and settings["gemini_models"]:
        if isinstance(settings["gemini_models"], list):
            GEMINI_MODELS = [str(m).strip() for m in settings["gemini_models"] if str(m).strip()]
        elif isinstance(settings["gemini_models"], str):
            GEMINI_MODELS = [m.strip() for m in settings["gemini_models"].split(",") if m.strip()]
    if "slim_prompt_mode" in settings and settings["slim_prompt_mode"] is not None:
        SLIM_PROMPT_MODE = bool(settings["slim_prompt_mode"])
    if "api_max_retries" in settings and settings["api_max_retries"] is not None:
        API_MAX_RETRIES = int(settings["api_max_retries"])
    if "inter_turn_delay" in settings and settings["inter_turn_delay"] is not None:
        INTER_TURN_DELAY = float(settings["inter_turn_delay"])
    if "max_historical_tool_chars" in settings and settings["max_historical_tool_chars"] is not None:
        MAX_HISTORICAL_TOOL_CHARS = int(settings["max_historical_tool_chars"])
    if "recent_turns_preserve_count" in settings and settings["recent_turns_preserve_count"] is not None:
        RECENT_TURNS_PRESERVE_COUNT = int(settings["recent_turns_preserve_count"])
    if "auto_compact_on_threshold" in settings and settings["auto_compact_on_threshold"] is not None:
        AUTO_COMPACT_ON_THRESHOLD = bool(settings["auto_compact_on_threshold"])
    if "free_tier_tpm_limit" in settings and settings["free_tier_tpm_limit"] is not None:
        FREE_TIER_TPM_LIMIT = int(settings["free_tier_tpm_limit"])
    if "free_tier_rpm_limit" in settings and settings["free_tier_rpm_limit"] is not None:
        FREE_TIER_RPM_LIMIT = int(settings["free_tier_rpm_limit"])
    if "multi_agent_enabled" in settings and settings["multi_agent_enabled"] is not None:
        MULTI_AGENT_ENABLED = bool(settings["multi_agent_enabled"])
    if "swarm_mode" in settings and settings["swarm_mode"] is not None:
        SWARM_MODE = bool(settings["swarm_mode"])

    logger.info("Persistent runtime settings applied successfully to config memory.")
