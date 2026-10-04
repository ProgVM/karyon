# karyon_agent_runtime/tools/config_tools.py
"""
===============================================================================
RUNTIME CONFIGURATION, 250K TPM QUOTA & QUEUE MANAGEMENT TOOL (v31.0 MASTER)
Allows Agent and Operator to Dynamically Adjust Models, Fallback Pools, 250k TPM
Ceilings, 40k Context Thresholds, Tool Output Truncation, Key Cooldowns & Pacing.
Fully Synchronized with SQLite Persistent Settings (runtime_settings), KeyManager,
Compactor Engine, and Live UI Callbacks Across Kaggle / Colab Restarts.
Features Full In-Flight Inspection (get_runtime_config) and Cascading Model
Rotation (force_rotate_model) along with Swarm & Autonomous Loop Settings.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import karyon_agent_runtime.config as config

logger = logging.getLogger("ProxyAgent.ConfigTool")


def _get_agent_and_db():
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    db = agent.db if agent else None
    return agent, db


async def reset_key_cooldowns() -> str:
    """
    Instantly clears all cooldown locks, rate-limit blocks, and rolling TPM counters
    across the entire Gemini API key pool, clears model blacklists, resets execution
    locks, and registers the recovery event in SQLite persistent memory.
    """
    agent, db = _get_agent_and_db()
    if not agent or not agent.key_manager:
        return "Error: Agent key manager is not initialized."

    # 1. Reset Key Manager cooldowns and sliding window telemetry
    cleared_count = len(agent.key_manager.exhausted_keys)
    agent.key_manager.reset_all_cooldowns()

    # 2. Reset agent execution locks and clean orphaned subprocesses
    agent.reset_locks()

    # 3. Log event into SQLite persistent database turns
    event_msg = f"[System Event: API key cooldown locks cleared ({cleared_count} locks removed across 36 projects), execution locks reset]"
    if db:
        try:
            await db.save_turn("system", event_msg)
            await db.checkpoint()
        except Exception as db_err:
            logger.debug(f"DB log notice in reset_key_cooldowns: {str(db_err)}")

    # 4. Broadcast to active UI if running
    if agent.active_event_callback:
        asyncio.create_task(
            agent.active_event_callback("info", "⚡ **Key Pool Unlocked:** Cooldowns cleared across all project keys. Pool is 100% active!")
        )

    return f"Success: Reset all {cleared_count} rate-limit cooldown locks. 60-second rolling TPM windows flushed. Execution locks cleared. Key pool is 100% ready."


async def force_rotate_key(target_key_index: Optional[int] = None, target_model: Optional[str] = None) -> str:
    """
    Forces immediate rotation to another API key in the pool on-demand,
    optionally switching the active model as well.

    Args:
        target_key_index: Optional 1-based index of key to activate (e.g. 1, 2, ..., 36). If None, advances to next key.
        target_model: Optional model identifier to activate (e.g. 'gemini-3.7-flash', 'gemini-2.5-flash').
    """
    agent, db = _get_agent_and_db()
    if not agent or not agent.key_manager:
        return "Error: Agent key manager is not initialized."

    km = agent.key_manager
    old_key_idx = km.current_key_index
    old_model = km.get_model()

    if target_model:
        clean_model = target_model.strip()
        km.invalid_models.discard(clean_model)
        if clean_model in km.models:
            km.current_model_index = km.models.index(clean_model)
        else:
            km.models.insert(0, clean_model)
            km.current_model_index = 0

    if target_key_index is not None and 1 <= target_key_index <= len(km.keys):
        km.current_key_index = target_key_index - 1
    else:
        km.current_key_index = (km.current_key_index + 1) % len(km.keys)

    km._init_client()
    new_key_idx = km.current_key_index
    new_model = km.get_model()

    if db:
        asyncio.create_task(db.set_setting("model", new_model))
        event_msg = f"[System Event: Key rotated: Key #{old_key_idx + 1} ({old_model}) ➔ Key #{new_key_idx + 1} ({new_model})]"
        asyncio.create_task(db.save_turn("system", event_msg))

    if agent.active_event_callback:
        asyncio.create_task(
            agent.active_event_callback("info", f"🔄 **Rotated:** Switched to Key #{new_key_idx + 1} (`{new_model}`)")
        )

    return f"Success: Switched active credentials from Key #{old_key_idx + 1} ({old_model}) to Key #{new_key_idx + 1} ({new_model})."


async def force_rotate_model(target_model: Optional[str] = None) -> str:
    """
    Forces immediate cascading failover to another Gemini model in the pool.
    If target_model is omitted, automatically advances to the next model in the cascade
    (e.g. gemini-3.8-flash -> gemini-3.7-flash -> gemini-3.5-flash -> gemini-2.5-flash).

    Args:
        target_model: Optional specific model to activate. If None, steps to the next model in GEMINI_MODELS.
    """
    agent, db = _get_agent_and_db()
    if not agent or not agent.key_manager:
        return "Error: Agent key manager is not initialized."

    km = agent.key_manager
    old_model = km.get_model()

    if target_model and target_model.strip():
        clean_model = target_model.strip()
        km.invalid_models.discard(clean_model)
        if clean_model in km.models:
            km.current_model_index = km.models.index(clean_model)
        else:
            km.models.insert(0, clean_model)
            km.current_model_index = 0
        new_model = km.get_model()
        km._init_client()
    else:
        new_model = km.rotate_to_next_model(reason="Operator or agent manual force_rotate_model request")

    if db:
        asyncio.create_task(db.set_setting("model", new_model))
        event_msg = f"[System Event: Model rotated: '{old_model}' ➔ '{new_model}' (Active Key #{km.current_key_index + 1})]"
        asyncio.create_task(db.save_turn("system", event_msg))

    if agent.active_event_callback:
        asyncio.create_task(
            agent.active_event_callback("info", f"🔄 **Model Switched:** `{old_model}` ➔ `{new_model}` across project keys.")
        )

    return f"Success: Switched active inference model from `{old_model}` to `{new_model}`. Key pool client refreshed."


async def get_key_pool_telemetry() -> str:
    """
    Returns deep real-time diagnostics of the 36-key Gemini API pool:
    Rolling 60-second TPM load per key, active cooldown countdowns, 250k TPM limit status,
    40k context compression thresholds, and queue pacing parameters across all 36 independent projects.
    """
    agent, _ = _get_agent_and_db()
    if not agent or not agent.key_manager:
        return "Error: Agent key manager is not initialized."

    km = agent.key_manager
    stat = km.get_pool_status()
    now = time.time()

    models_str = ", ".join(getattr(config, "GEMINI_MODELS", []))
    sum_models_str = ", ".join(getattr(config, "SUMMARIZER_MODELS", []))
    tpm_limit = getattr(config, "FREE_TIER_TPM_LIMIT", 250000)
    current_key_tpm = km.get_key_60s_tokens(km.current_key_index, now)
    tpm_pct = (current_key_tpm / tpm_limit * 100.0) if tpm_limit > 0 else 0.0

    lines = [
        "=== GEMINI API KEY POOL & 250K TPM QUOTA TELEMETRY (36 INDEPENDENT PROJECTS) ===",
        f"- Total Unique Project Keys : {stat['total_keys']} (6 Accounts x 6 Projects)",
        f"- Active Ready Keys         : {stat['active_ready_keys']} / {stat['total_keys']} (for model `{stat['current_model']}`)",
        f"- Keys on Cooldown          : {stat['cooldown_keys']}",
        f"- Active Key Pointer        : #{stat['current_key_index']} (Account #{stat['current_account_index']}, Project #{stat['current_project_index']})",
        f"- Active Key 60s Token Load : {current_key_tpm:,} / {tpm_limit:,} TPM ({tpm_pct:.1f}% used)",
        f"- Active Inference Model    : `{stat['current_model']}`",
        f"- Inference Models Pool     : `{models_str}`",
        f"- Summarizer Pool           : `{sum_models_str}`",
        f"- Context Threshold         : {getattr(config, 'CONTEXT_COMPRESSION_THRESHOLD', 40000):,} tokens",
        f"- Historical Tool Output Cap: {getattr(config, 'MAX_HISTORICAL_TOOL_CHARS', 1200)} chars max",
        f"- Recent Turns Kept Raw     : {getattr(config, 'RECENT_TURNS_PRESERVE_COUNT', 10)} turns",
        f"- Auto-Compact on Threshold : {getattr(config, 'AUTO_COMPACT_ON_THRESHOLD', True)}",
        f"- Slim Mode Active          : {getattr(config, 'SLIM_PROMPT_MODE', True)} (95% TPM saving)",
        f"- API Max Retries           : {getattr(config, 'API_MAX_RETRIES', 60)}",
        f"- Inter-Turn Delay (Pacing) : {getattr(config, 'INTER_TURN_DELAY', 0.5)}s",
        f"- Multi-Agent System Active : {getattr(config, 'MULTI_AGENT_ENABLED', True)}",
        f"- Cortical Swarm Mode       : {getattr(config, 'SWARM_MODE', True)}"
    ]

    # Safe unpacking of (key_idx, model) cooldown entries
    if km.exhausted_keys:
        lines.append("\n--- Keys Currently on Rate-Limit Cooldown ---")
        lines.append("| Key # | Model | Account | Project | Seconds Remaining | Type |")
        lines.append("|---|---|---|---|---|---|")

        sorted_cooldowns = sorted(
            km.exhausted_keys.items(),
            key=lambda it: (it[0][0] if isinstance(it[0], tuple) else it[0], str(it[0]))
        )

        for key_ident, expire_ts in sorted_cooldowns:
            if isinstance(key_ident, tuple):
                k_idx, m_name = key_ident[0], key_ident[1]
            else:
                k_idx, m_name = key_ident, "all"

            rem = max(0.0, expire_ts - now)
            acc = km.get_account_for_key(k_idx) + 1
            proj = km.get_project_for_key(k_idx) + 1
            quota_type = "Daily Quota (Midnight Reset)" if rem > 600.0 else "Rolling RPM/TPM Window"
            lines.append(f"| Key #{k_idx + 1} | `{m_name}` | Acc #{acc} | Proj #{proj} | {rem:.1f}s | {quota_type} |")
    else:
        lines.append(f"\n✅ Clean Pool: Zero keys are currently rate-limited on `{stat['current_model']}`. 100% capacity ready.")

    return "\n".join(lines)


async def get_runtime_config() -> str:
    """
    Returns a complete, transparent snapshot of all active runtime configuration settings,
    hyperparameters, model cascades, quota limits, and autonomous daemon parameters.
    """
    agent, _ = _get_agent_and_db()
    active_model = agent.key_manager.get_model() if agent else config.GEMINI_MODELS[0]
    active_key = agent.key_manager.current_key_index + 1 if agent else 1
    active_db = Path(config.DB_PATH).name
    loop_status = agent.get_autonomous_loop_status() if agent else {}

    lines = [
        "=== KARYON AGENT RUNTIME CONFIGURATION MATRIX (LIVE STATE) ===",
        f"- Active Inference Model     : `{active_model}` (Active Key #{active_key})",
        f"- Active Database            : `{active_db}`",
        f"- Inference Models Cascade   : `{', '.join(getattr(config, 'GEMINI_MODELS', []))}`",
        f"- Primary Summarizer Model   : `{getattr(config, 'SUMMARIZER_MODEL', 'gemini-3.7-flash')}`",
        f"- Summarizer Pool Cascade    : `{', '.join(getattr(config, 'SUMMARIZER_MODELS', []))}`",
        f"- Temperature / Top-P        : {config.TEMPERATURE:.2f} / {config.TOP_P:.2f}",
        f"- Max Output Tokens          : {config.MAX_OUTPUT_TOKENS:,} tokens",
        f"- Thinking Reasoning Profile : Level=`{config.THINKING_LEVEL}`, Budget={config.THINKING_BUDGET:,} tokens",
        f"- Safety Policy              : {'BLOCK_NONE (All 4 categories unblocked)' if config.SAFETY_BLOCK_NONE else 'BLOCK_MEDIUM_AND_ABOVE'}",
        f"- Max Agent Turns per Query  : {config.MAX_AGENT_TURNS}",
        f"- Context Compression Limit  : {config.CONTEXT_COMPRESSION_THRESHOLD:,} tokens (Preserve {config.RECENT_TURNS_PRESERVE_COUNT} turns raw)",
        f"- Auto-Compact on Threshold  : {config.AUTO_COMPACT_ON_THRESHOLD}",
        f"- Historical Tool Char Cap   : {config.MAX_HISTORICAL_TOOL_CHARS:,} chars",
        f"- Slim Prompt Mode           : {config.SLIM_PROMPT_MODE} (Prevents 120k token prompt bloat)",
        f"- Free-Tier Project Quotas   : TPM={config.FREE_TIER_TPM_LIMIT:,}, RPM={config.FREE_TIER_RPM_LIMIT}, Margin={config.PREEMPTIVE_TPM_SAFETY_MARGIN:.2f}",
        f"- Pacing & Retries           : Inter-Turn Delay={config.INTER_TURN_DELAY:.1f}s, Max API Retries={config.API_MAX_RETRIES}, Backoff Base={config.API_BACKOFF_BASE_DELAY:.1f}s",
        f"- Multi-Agent System         : Enabled={config.MULTI_AGENT_ENABLED}, Swarm Mode={config.SWARM_MODE}",
        f"- Autonomous 24/7 Loop       : Active={loop_status.get('active', False)}, Paused={loop_status.get('paused', False)}, Cycle=#{loop_status.get('cycle', 0)}/{loop_status.get('max_cycles', 1000)}, Interval={loop_status.get('interval_seconds', 15.0):.1f}s"
    ]
    return "\n".join(lines)


async def update_runtime_config(
    model: Optional[str] = None,
    models_pool: Optional[str] = None,
    summarizer_model: Optional[str] = None,
    summarizer_models_pool: Optional[str] = None,
    context_compression_threshold: Optional[int] = None,
    compression_threshold: Optional[int] = None,
    threshold: Optional[int] = None,
    free_tier_tpm_limit: Optional[int] = None,
    tpm_limit: Optional[int] = None,
    free_tier_rpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    max_historical_tool_chars: Optional[int] = None,
    preemptive_tpm_safety_margin: Optional[float] = None,
    recent_turns_preserve_count: Optional[int] = None,
    auto_compact_on_threshold: Optional[bool] = None,
    slim_prompt_mode: Optional[bool] = None,
    api_max_retries: Optional[int] = None,
    inter_turn_delay: Optional[float] = None,
    api_backoff_base_delay: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_turns: Optional[int] = None,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    safety_block_none: Optional[bool] = None,
    max_output_tokens: Optional[int] = None,
    multi_agent_enabled: Optional[bool] = None,
    swarm_mode: Optional[bool] = None,
    autonomous_max_cycles: Optional[int] = None,
    autonomous_interval_seconds: Optional[float] = None,
    autonomous_agenda: Optional[str] = None,
    **kwargs
) -> str:
    """
    Dynamically adjusts agent hyperparameters, active LLM model, fallback model pools,
    250k TPM ceilings, context compaction thresholds, tool output caps, queue delays,
    swarm mode, and turn limits in-flight.
    CRITICAL: All modified settings are persistently saved to the SQLite 'runtime_settings'
    table so they survive kernel restarts and session reconnections seamlessly.
    """
    agent, db = _get_agent_and_db()
    changes = []
    persisted_map: Dict[str, Any] = {}

    # 1. Context Compression Threshold
    eff_threshold = (
        context_compression_threshold
        if context_compression_threshold is not None
        else compression_threshold
        if compression_threshold is not None
        else threshold
        if threshold is not None
        else kwargs.get("compact_at")
    )
    if eff_threshold is not None:
        old_val = getattr(config, "CONTEXT_COMPRESSION_THRESHOLD", 40000)
        new_val = max(10000, int(eff_threshold))
        config.CONTEXT_COMPRESSION_THRESHOLD = new_val
        if agent and agent.compactor:
            agent.compactor.compression_threshold = new_val
        persisted_map["context_compression_threshold"] = new_val
        changes.append(f"Context Threshold: {old_val:,} ➔ {new_val:,} tokens")

    # 2. Free-Tier TPM & RPM Quotas
    eff_tpm = free_tier_tpm_limit if free_tier_tpm_limit is not None else tpm_limit if tpm_limit is not None else kwargs.get("max_tpm")
    if eff_tpm is not None:
        old_tpm = getattr(config, "FREE_TIER_TPM_LIMIT", 250000)
        config.FREE_TIER_TPM_LIMIT = max(10000, int(eff_tpm))
        persisted_map["free_tier_tpm_limit"] = config.FREE_TIER_TPM_LIMIT
        changes.append(f"Free-Tier TPM Limit: {old_tpm:,} ➔ {config.FREE_TIER_TPM_LIMIT:,} tokens/min")

    eff_rpm = free_tier_rpm_limit if free_tier_rpm_limit is not None else rpm_limit if rpm_limit is not None else kwargs.get("max_rpm")
    if eff_rpm is not None:
        old_rpm = getattr(config, "FREE_TIER_RPM_LIMIT", 15)
        config.FREE_TIER_RPM_LIMIT = max(1, int(eff_rpm))
        persisted_map["free_tier_rpm_limit"] = config.FREE_TIER_RPM_LIMIT
        changes.append(f"Free-Tier RPM Limit: {old_rpm} ➔ {config.FREE_TIER_RPM_LIMIT} req/min")

    # 3. Tool Output Capping & Pacing
    eff_tool_cap = max_historical_tool_chars if max_historical_tool_chars is not None else kwargs.get("historical_tool_cap")
    if eff_tool_cap is not None:
        old_cap = getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)
        config.MAX_HISTORICAL_TOOL_CHARS = max(200, int(eff_tool_cap))
        persisted_map["max_historical_tool_chars"] = config.MAX_HISTORICAL_TOOL_CHARS
        changes.append(f"Historical Tool Char Cap: {old_cap} ➔ {config.MAX_HISTORICAL_TOOL_CHARS} chars")

    if preemptive_tpm_safety_margin is not None:
        old_margin = getattr(config, "PREEMPTIVE_TPM_SAFETY_MARGIN", 0.85)
        config.PREEMPTIVE_TPM_SAFETY_MARGIN = max(0.5, min(0.99, float(preemptive_tpm_safety_margin)))
        persisted_map["preemptive_tpm_safety_margin"] = config.PREEMPTIVE_TPM_SAFETY_MARGIN
        changes.append(f"Preemptive TPM Safety Margin: {old_margin:.2f} ➔ {config.PREEMPTIVE_TPM_SAFETY_MARGIN:.2f}")

    eff_preserve = recent_turns_preserve_count if recent_turns_preserve_count is not None else kwargs.get("preserve_count")
    if eff_preserve is not None:
        old_pres = getattr(config, "RECENT_TURNS_PRESERVE_COUNT", 10)
        config.RECENT_TURNS_PRESERVE_COUNT = max(4, int(eff_preserve))
        if agent and agent.compactor:
            agent.compactor.preserve_count = config.RECENT_TURNS_PRESERVE_COUNT
        persisted_map["recent_turns_preserve_count"] = config.RECENT_TURNS_PRESERVE_COUNT
        changes.append(f"Recent Turns Preserved Raw: {old_pres} ➔ {config.RECENT_TURNS_PRESERVE_COUNT}")

    if auto_compact_on_threshold is not None:
        config.AUTO_COMPACT_ON_THRESHOLD = bool(auto_compact_on_threshold)
        persisted_map["auto_compact_on_threshold"] = config.AUTO_COMPACT_ON_THRESHOLD
        changes.append(f"Auto-Compact on Threshold: {config.AUTO_COMPACT_ON_THRESHOLD}")

    # 4. Models and Summarizer Pools
    if models_pool is not None and agent and agent.key_manager:
        parsed_models = [m.strip() for m in models_pool.split(",") if m.strip()]
        if parsed_models:
            config.GEMINI_MODELS = parsed_models
            agent.key_manager.models = parsed_models
            agent.key_manager.invalid_models.clear()
            persisted_map["gemini_models"] = parsed_models
            changes.append(f"Inference Models Pool: `{', '.join(parsed_models)}`")

    if summarizer_models_pool is not None and agent and agent.compactor:
        parsed_sum = [m.strip() for m in summarizer_models_pool.split(",") if m.strip()]
        if parsed_sum:
            config.SUMMARIZER_MODELS = parsed_sum
            agent.compactor.summarizer_models = parsed_sum
            persisted_map["summarizer_models"] = parsed_sum
            changes.append(f"Summarizer Models Pool: `{', '.join(parsed_sum)}`")

    if summarizer_model is not None:
        clean_sum_model = summarizer_model.strip()
        config.SUMMARIZER_MODEL = clean_sum_model
        if agent and agent.compactor:
            if clean_sum_model in agent.compactor.summarizer_models:
                agent.compactor.current_summarizer_idx = agent.compactor.summarizer_models.index(clean_sum_model)
            else:
                agent.compactor.summarizer_models.insert(0, clean_sum_model)
                agent.compactor.current_summarizer_idx = 0
        persisted_map["summarizer_model"] = clean_sum_model
        changes.append(f"Primary Summarizer: `{clean_sum_model}`")

    if slim_prompt_mode is not None:
        config.SLIM_PROMPT_MODE = bool(slim_prompt_mode)
        persisted_map["slim_prompt_mode"] = config.SLIM_PROMPT_MODE
        changes.append(f"Slim Prompt Mode: {config.SLIM_PROMPT_MODE}")

    # 5. Delays, Retries and Queue Tuning
    eff_retries = api_max_retries if api_max_retries is not None else kwargs.get("retries")
    if eff_retries is not None:
        old_retries = getattr(config, "API_MAX_RETRIES", 60)
        config.API_MAX_RETRIES = max(1, int(eff_retries))
        persisted_map["api_max_retries"] = config.API_MAX_RETRIES
        changes.append(f"API Max Retries: {old_retries} ➔ {config.API_MAX_RETRIES}")

    eff_turn_delay = inter_turn_delay if inter_turn_delay is not None else kwargs.get("turn_delay") or kwargs.get("delay")
    if eff_turn_delay is not None:
        old_delay = getattr(config, "INTER_TURN_DELAY", 0.5)
        config.INTER_TURN_DELAY = max(0.0, float(eff_turn_delay))
        persisted_map["inter_turn_delay"] = config.INTER_TURN_DELAY
        changes.append(f"Inter-Turn Pacing Delay: {old_delay:.1f}s ➔ {config.INTER_TURN_DELAY:.1f}s")

    if api_backoff_base_delay is not None:
        old_backoff = getattr(config, "API_BACKOFF_BASE_DELAY", 5.0)
        config.API_BACKOFF_BASE_DELAY = max(1.0, float(api_backoff_base_delay))
        persisted_map["api_backoff_base_delay"] = config.API_BACKOFF_BASE_DELAY
        changes.append(f"API Backoff Base Delay: {old_backoff:.1f}s ➔ {config.API_BACKOFF_BASE_DELAY:.1f}s")

    # 6. Sampling Parameters
    if temperature is not None:
        old_temp = config.TEMPERATURE
        config.TEMPERATURE = max(0.0, min(1.0, float(temperature)))
        persisted_map["temperature"] = config.TEMPERATURE
        changes.append(f"Temperature: {old_temp:.2f} ➔ {config.TEMPERATURE:.2f}")

    if top_p is not None:
        old_top_p = config.TOP_P
        config.TOP_P = max(0.1, min(1.0, float(top_p)))
        persisted_map["top_p"] = config.TOP_P
        changes.append(f"Top-P: {old_top_p:.2f} ➔ {config.TOP_P:.2f}")

    if max_turns is not None:
        old_turns = getattr(config, "MAX_AGENT_TURNS", 500)
        config.MAX_AGENT_TURNS = max(10, int(max_turns))
        persisted_map["max_agent_turns"] = config.MAX_AGENT_TURNS
        changes.append(f"Max Agent Turns: {old_turns} ➔ {config.MAX_AGENT_TURNS}")

    if thinking_level is not None:
        old_level = getattr(config, "THINKING_LEVEL", "HIGH")
        config.THINKING_LEVEL = str(thinking_level).upper().strip()
        persisted_map["thinking_level"] = config.THINKING_LEVEL
        changes.append(f"Thinking Level: '{old_level}' ➔ '{config.THINKING_LEVEL}'")

    if thinking_budget is not None:
        old_budget = getattr(config, "THINKING_BUDGET", 24576)
        config.THINKING_BUDGET = max(0, int(thinking_budget))
        persisted_map["thinking_budget"] = config.THINKING_BUDGET
        changes.append(f"Thinking Budget: {old_budget} ➔ {config.THINKING_BUDGET} tokens")

    if safety_block_none is not None:
        config.SAFETY_BLOCK_NONE = bool(safety_block_none)
        persisted_map["safety_block_none"] = config.SAFETY_BLOCK_NONE
        changes.append(f"Safety BLOCK_NONE: {config.SAFETY_BLOCK_NONE}")

    if max_output_tokens is not None:
        old_tokens = config.MAX_OUTPUT_TOKENS
        config.MAX_OUTPUT_TOKENS = max(1024, int(max_output_tokens))
        persisted_map["max_output_tokens"] = config.MAX_OUTPUT_TOKENS
        changes.append(f"Max Output Tokens: {old_tokens} ➔ {config.MAX_OUTPUT_TOKENS}")

    # 7. Model Override
    if model is not None and agent and agent.key_manager:
        old_model = agent.key_manager.get_model()
        model_clean = model.strip()
        agent.key_manager.invalid_models.discard(model_clean)
        if model_clean in agent.key_manager.models:
            agent.key_manager.current_model_index = agent.key_manager.models.index(model_clean)
        else:
            agent.key_manager.models.insert(0, model_clean)
            agent.key_manager.current_model_index = 0
        if model_clean not in config.GEMINI_MODELS:
            config.GEMINI_MODELS.insert(0, model_clean)
        agent.key_manager._init_client()
        persisted_map["model"] = agent.key_manager.get_model()
        changes.append(f"Model: '{old_model}' ➔ '{agent.key_manager.get_model()}'")

    # 8. Multi-Agent System & Cortical Swarm Toggles
    if multi_agent_enabled is not None:
        config.MULTI_AGENT_ENABLED = bool(multi_agent_enabled)
        persisted_map["multi_agent_enabled"] = config.MULTI_AGENT_ENABLED
        changes.append(f"Multi-Agent System: {'Enabled' if config.MULTI_AGENT_ENABLED else 'Disabled'}")

    if swarm_mode is not None:
        config.SWARM_MODE = bool(swarm_mode)
        if agent:
            agent.swarm_mode = bool(swarm_mode)
        persisted_map["swarm_mode"] = config.SWARM_MODE
        changes.append(f"Cortical Swarm Mode: {'Enabled' if config.SWARM_MODE else 'Disabled'}")

    # 9. Autonomous Loop Parameters
    if autonomous_max_cycles is not None:
        new_max_cycles = max(1, int(autonomous_max_cycles))
        if agent:
            agent.autonomous_max_cycles = new_max_cycles
        persisted_map["autonomous_max_cycles"] = new_max_cycles
        changes.append(f"Autonomous Max Cycles: {new_max_cycles}")

    if autonomous_interval_seconds is not None:
        new_interval = max(1.0, float(autonomous_interval_seconds))
        if agent:
            agent.autonomous_interval_seconds = new_interval
        persisted_map["autonomous_interval_seconds"] = new_interval
        changes.append(f"Autonomous Interval Delay: {new_interval:.1f}s")

    if autonomous_agenda is not None and str(autonomous_agenda).strip():
        persisted_map["autonomous_agenda"] = str(autonomous_agenda).strip()
        changes.append(f"Autonomous Agenda updated ({len(str(autonomous_agenda))} chars)")

    if not changes:
        return "No configuration changes were specified. Current settings preserved."

    # 10. Persist all updated settings to SQLite runtime_settings table
    if db:
        try:
            for k, v in persisted_map.items():
                asyncio.create_task(db.set_setting(k, v))

            event_text = (
                "[System Event: Runtime configuration updated and saved persistently to SQLite:\n" +
                "\n".join(f"- {c}" for c in changes) + "]"
            )
            asyncio.create_task(db.save_turn("system", event_text))
        except Exception as p_err:
            logger.debug(f"Notice saving runtime settings to SQLite: {str(p_err)}")

    result_msg = (
        "=== Runtime Configuration Updated & Persisted Successfully ===\n" +
        "\n".join(f"- {c}" for c in changes) +
        "\n*(All settings saved to SQLite 'runtime_settings' table and will survive future restarts)*"
    )
    logger.info(result_msg)

    # 11. Broadcast event to UI
    if agent and agent.active_event_callback:
        asyncio.create_task(
            agent.active_event_callback(
                "info",
                "⚙️ **Config Dynamically Updated & Persisted:**\n" + "\n".join(f"- {c}" for c in changes)
            )
        )

    return result_msg
