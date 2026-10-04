# karyon_agent_runtime/tools/context_tools.py
"""
===============================================================================
AUTONOMOUS CONTEXT TOKEN AUDIT & MULTI-MODEL COMPRESSION TOOLS (v23.0 MASTER)
Enables Live Inspection of Token Usage, Dynamic Threshold Monitoring, and
Resilient Multi-Model Context Distillation with Permanent Database Turn Archival.
===============================================================================
"""

import logging
from typing import Optional
import karyon_agent_runtime.config as config
from karyon_agent_runtime.prompt_builder import build_system_prompt

logger = logging.getLogger("ProxyAgent.ContextTools")


def _get_active_compactor():
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    if not agent or not agent.compactor:
        raise RuntimeError("Agent Context Compactor is not initialized.")
    return agent, agent.compactor


async def get_context_token_status() -> str:
    """
    Measures and returns the exact token utilization across the current session,
    including system instructions, master documents, empirical ledger, dialogue turns,
    and active compressed state.
    """
    agent, compactor = _get_active_compactor()
    raw_turns = await agent.db.get_recent_turns(limit=100, include_tools=True)
    system_instruction = build_system_prompt()

    telemetry = await compactor.get_token_breakdown(system_instruction, raw_turns)

    lines = [
        "=== CONTEXT TOKEN VALUATION & CAPACITY TELEMETRY ===",
        f"- Total Active Tokens in Window : {telemetry['total_tokens']:,} tokens",
        f"- Compression Threshold         : {telemetry['compression_threshold']:,} tokens",
        f"- Context Window Capacity       : {getattr(config, 'INPUT_TOKEN_LIMIT', 1048576):,} tokens",
        f"- Threshold Utilization         : {telemetry['usage_percentage']:.2f}%",
        f"- System Prompt & Codebase      : {telemetry['system_instruction_tokens']:,} tokens",
        f"- Active Working Contents       : {telemetry['active_contents_tokens']:,} tokens",
        f"- Active Turns in Memory        : {telemetry['raw_turns_count']} turns ({telemetry['raw_history_tokens']:,} tokens)",
        f"- Stored Distilled State Summary: {telemetry['active_summary_tokens']:,} tokens",
        f"- Active Inference Model        : `{telemetry['active_model']}`",
        f"- Active Summarizer Model       : `{telemetry['summarizer_model']}`",
        f"- Summarizer Models Pool        : `{', '.join(telemetry.get('summarizer_models_pool', []))}`",
        f"- Auto-Compaction on Threshold  : {getattr(config, 'AUTO_COMPACT_ON_THRESHOLD', True)}"
    ]

    if telemetry['total_tokens'] >= telemetry['compression_threshold']:
        lines.append("\n⚠️ Notice: Active token count exceeds threshold. Auto-compactor will activate on upcoming turns.")
    else:
        remaining = telemetry['compression_threshold'] - telemetry['total_tokens']
        lines.append(f"\n✅ Healthy Token Headroom: {remaining:,} tokens remaining before next compaction.")

    return "\n".join(lines)


async def compress_context_now(focus_directive: Optional[str] = None) -> str:
    """
    Immediately triggers lossless context compaction using the SUMMARIZER_MODELS pool,
    distilling older history while preserving exact telemetry and archiving older turns from active memory.

    Args:
        focus_directive: Optional custom directive (e.g. 'Preserve exact PAC theta-gamma equations and EXP-71 metrics').
    """
    agent, compactor = _get_active_compactor()
    raw_turns = await agent.db.get_recent_turns(limit=100, include_tools=True)

    if len(raw_turns) <= compactor.preserve_count:
        return f"Context compaction not required: only {len(raw_turns)} dialogue turns in history (minimum preserve threshold is {compactor.preserve_count})."

    logger.info(f"Executing on-demand context distillation across {len(raw_turns)} turns...")

    system_instruction = build_system_prompt()

    # 1. Measure Pre-compaction Tokens
    pre_telemetry = await compactor.get_token_breakdown(system_instruction, raw_turns)
    pre_tokens = pre_telemetry["total_tokens"]

    # 2. Partition and Execute Compaction
    k = max(8, compactor.preserve_count)
    older_turns = raw_turns[:-k]

    summary_text = await compactor.summarize_dialogue_slice(older_turns, focus_directive=focus_directive)

    if not summary_text:
        return f"❌ Error: Context distillation failed across all models in SUMMARIZER_MODELS pool ({', '.join(compactor.summarizer_models)}). Context was NOT pruned to prevent loss of working memory."

    # 3. Save to Persistent DB & Prune
    last_older_id = older_turns[-1]["id"] if older_turns else 0
    cache_key = f"lossless_state_snapshot_{last_older_id}"

    await agent.db.set_memory(cache_key, summary_text, category="summary")
    await agent.db.set_memory("active_context_summary", summary_text, category="summary")
    pruned_count = await agent.db.archive_old_turns(keep_count=k)

    # 4. Measure Post-compaction Tokens
    post_raw_turns = await agent.db.get_recent_turns(limit=100, include_tools=True)
    post_telemetry = await compactor.get_token_breakdown(system_instruction, post_raw_turns)
    post_tokens = post_telemetry["total_tokens"]

    saved_tokens = max(0, pre_tokens - post_tokens)

    await agent.db.save_turn(
        "system",
        f"[System Event: Lossless Context Compaction Completed. Distilled and archived {pruned_count} turns via '{compactor.get_active_summarizer_model()}'. Reclaimed ~{saved_tokens:,} tokens. New footprint: {post_tokens:,} tokens]"
    )

    result_msg = (
        f"=== LOSSLESS CONTEXT COMPACTION COMPLETE ===\n"
        f"- Extractor Model Utilized  : `{compactor.get_active_summarizer_model()}` (from SUMMARIZER_MODELS pool)\n"
        f"- Focus Directive Applied   : {focus_directive or 'Standard KEP Lossless Protocol'}\n"
        f"- Turns Distilled & Archived: {pruned_count} older turns\n"
        f"- Verbatim Turns Preserved  : {len(post_raw_turns)} latest turns\n"
        f"- Pre-Compaction Window     : {pre_tokens:,} tokens\n"
        f"- Post-Compaction Window    : {post_tokens:,} tokens\n"
        f"- Actual Tokens Reclaimed   : ~{saved_tokens:,} tokens\n\n"
        f"--- Active Distilled State Summary ---\n{summary_text}"
    )

    return result_msg
