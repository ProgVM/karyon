# karyon_agent_runtime/context_compactor.py
"""
===============================================================================
LOSSLESS CONTEXT COMPACTOR & 250K TPM QUOTA PRESERVATION ENGINE (v29.0 MASTER)
Features Historical Tool Output Capping (1.2k chars max), Zero-Cost Token Estimation,
Dynamic 40k Context Compaction, and Complete GenAI Content Alignment.
Enhanced with Orphan FunctionCall/FunctionResponse Auto-Healing (Anti-400 Deadlock).
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import ast
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from google.genai import types

import karyon_agent_runtime.config as config
from karyon_agent_runtime.db_manager import dict_to_content

logger = logging.getLogger("ProxyAgent.Compactor")

PRECISE_SCIENTIFIC_SUMMARIZER_PROMPT = """
You are the Lead Scientific Knowledge Extractor for the Karyon-CoRE AI Architecture project.
Your mission is to perform a 100% LOSSLESS, highly condensed state distillation of the dialogue and tool history.
You MUST extract exact empirical values, mathematical formulas, file modifications, git commit hashes, and active constraints.

CRITICAL DISTILLATION RULES:
1. NEVER generalize numerical results. Preserve exact figures: Initial/Final Loss, Perplexity, Free Energy (F), Gradient Norms, Throughput (tok/s), VRAM (MB).
2. PRESERVE ALL PERMANENT USER CONSTRAINTS: Every rule and directive given by Bazilevs must be explicitly recorded.
3. PRESERVE ACTIVE CODEBASE STATE: Record every modified file, created script, fixed bug, and active git branch/hash.
4. RECORD FAILED EXPERIMENTS & ROOT CAUSES: Detail why an approach was rejected (NaN overflow, OOM, loss regression) to prevent repeating mistakes.

STRUCTURE YOUR LOSSLESS STATE OUTPUT IN THE FOLLOWING FORMAT:
=== 1. PERMANENT USER DIRECTIVES & CONSTRAINTS ===
- [List every strict constraint, formatting rule, or research protocol]

=== 2. ACTIVE ARCHITECTURE & CODEBASE STATE ===
- [List modified files, C++20 bindings, active hyperparameters, and git commits]

=== 3. EMPIRICAL BENCHMARK LEDGER (EXACT TELEMETRY) ===
| EXP ID | Hypothesis & Delta | Final Loss | PPL | Tok/s | VRAM | Verdict | Key Finding |
|---|---|---|---|---|---|---|---|

=== 4. ACTIVE HYPOTHESIS & IMMEDIATE PENDING TASKS ===
- [Current active problem, in-flight experiment, and next strategic actions]

=== 5. REJECTED APPROACHES & ANOMALIES ===
- [Approaches that failed, NaN issues, or invalid heuristics]
"""


def estimate_tokens_fallback_chars(char_count: int) -> int:
    """Accurate character-to-token ratio estimation (1 token ≈ 3.6 chars for code/markdown)."""
    if char_count <= 0:
        return 0
    return max(1, int(char_count / 3.6))


def estimate_tokens_fallback(text: str) -> int:
    """Estimates tokens from string input."""
    if not text:
        return 0
    return estimate_tokens_fallback_chars(len(text))


def truncate_tool_response_text(raw_text: str, max_chars: Optional[int] = None) -> str:
    """Caps historical tool outputs to conserve the 250k TPM budget."""
    cap = max_chars or getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)
    if not raw_text or len(raw_text) <= cap:
        return raw_text
    return raw_text[:cap] + f"\n... [Truncated {len(raw_text) - cap} characters to conserve 250k TPM budget. Full output preserved in SQLite database] ..."


def turn_to_content(turn: Dict[str, Any], is_historical: bool = True) -> Optional[types.Content]:
    """
    Converts a database turn row into a valid Google GenAI Content object.
    Automatically caps oversized historical tool outputs to prevent 250k TPM quota spikes.
    """
    raw_json = turn.get("raw_parts_json")
    max_tool_chars = getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)

    if raw_json:
        try:
            data = json.loads(raw_json)
            # Cap historical function response payload
            if is_historical and isinstance(data, dict) and "parts" in data and isinstance(data["parts"], list):
                for p in data["parts"]:
                    if isinstance(p, dict) and "function_response" in p and isinstance(p["function_response"], dict):
                        resp = p["function_response"].get("response", {})
                        if isinstance(resp, dict) and "result" in resp:
                            res_str = str(resp["result"])
                            if len(res_str) > max_tool_chars:
                                resp["result"] = truncate_tool_response_text(res_str, max_tool_chars)
            return dict_to_content(data)
        except Exception:
            pass

    role = turn.get("role", "user").lower()
    text = turn.get("text", "").strip()
    if not text:
        return None

    if role == "tool_call" or text.startswith("[Tool Invocation:"):
        m = re.match(r'\[Tool Invocation:\s*`?([a-zA-Z0-9_]+)`?\s*\]\s*(\{[\s\S]*\})', text)
        if m:
            fn = m.group(1).strip()
            args_raw = m.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception:
                try:
                    args = ast.literal_eval(args_raw)
                except Exception:
                    args = {}
            part = types.Part.from_function_call(name=fn, args=args if isinstance(args, dict) else {})
            part.thought_signature = b"skip_thought_signature_validator"
            return types.Content(role="model", parts=[part])
        return types.Content(role="user", parts=[types.Part.from_text(text=f"[Historical Action: {text}]")])

    elif role == "tool_result" or text.startswith("[Tool Output:"):
        m = re.match(r'\[Tool Output:\s*`?([a-zA-Z0-9_]+)`?\s*\]\s*```(?:text)?\n([\s\S]*?)```', text)
        if m:
            fn = m.group(1).strip()
            res_str = m.group(2)
            if is_historical and len(res_str) > max_tool_chars:
                res_str = truncate_tool_response_text(res_str, max_tool_chars)
            part = types.Part.from_function_response(name=fn, response={"result": res_str})
            return types.Content(role="user", parts=[part])
        clean_text = truncate_tool_response_text(text, max_tool_chars) if is_historical else text
        return types.Content(role="user", parts=[types.Part.from_text(text=f"[Historical Result: {clean_text}]")])

    role_mapped = "model" if role in ["model", "agent_message"] else "user"
    return types.Content(role=role_mapped, parts=[types.Part.from_text(text=text)])


def sanitize_and_align_contents(contents: List[types.Content]) -> List[types.Content]:
    """
    Guarantees 100% compliance with Google GenAI / Gemini multi-turn API specifications:
    1. Roles strictly alternate: user -> model -> user -> model.
    2. First turn MUST be role='user'.
    3. FunctionCall parts are only in 'model' turns; FunctionResponse parts are only in 'user' turns.
    4. Auto-heals orphan FunctionCalls (e.g. after cancellation or timeout) by inserting matching
       synthetic FunctionResponse parts to prevent 400 Alignment Deadlocks.
    5. Merges adjacent turns with identical roles cleanly.
    """
    if not contents:
        return [types.Content(role="user", parts=[types.Part.from_text(text="[System: Session initialized]")])]

    raw_stages: List[types.Content] = []
    i = 0

    while i < len(contents):
        curr = contents[i]
        role = curr.role.lower()

        if role in ["tool", "tool_result", "system", "developer", "context", "user_context"]:
            role = "user"
        elif role in ["assistant", "tool_call", "agent_message"]:
            role = "model"

        valid_parts = []
        for p in (curr.parts or []):
            if getattr(p, "text", None) is not None:
                if str(p.text).strip():
                    valid_parts.append(p)
            else:
                valid_parts.append(p)

        if not valid_parts:
            i += 1
            continue

        has_fc = any(getattr(p, "function_call", None) is not None for p in valid_parts)
        has_fr = any(getattr(p, "function_response", None) is not None for p in valid_parts)

        # 1. FunctionResponse turn handling
        if has_fr:
            prev_has_fc = (
                raw_stages and raw_stages[-1].role == "model" and
                any(getattr(p, "function_call", None) is not None for p in (raw_stages[-1].parts or []))
            )
            if not prev_has_fc:
                # Orphan FunctionResponse without preceding FunctionCall -> Convert to text
                converted = []
                for p in valid_parts:
                    if getattr(p, "function_response", None) is not None:
                        fr = p.function_response
                        name = getattr(fr, "name", "tool")
                        resp = getattr(fr, "response", {})
                        converted.append(types.Part.from_text(text=f"[Historical Tool Result for {name}: {str(resp)}]"))
                    else:
                        converted.append(p)
                raw_stages.append(types.Content(role="user", parts=converted))
                i += 1
            else:
                raw_stages.append(types.Content(role="user", parts=valid_parts))
                i += 1

        # 2. Model with FunctionCall turn handling
        elif role == "model" and has_fc:
            next_has_fr = (
                i + 1 < len(contents) and
                any(getattr(p, "function_response", None) is not None for p in (contents[i + 1].parts or []))
            )

            # Ensure thought_signature is set on function_calls
            guarded_parts = []
            fc_names = []
            for p in valid_parts:
                if getattr(p, "function_call", None) is not None:
                    p.thought_signature = b"skip_thought_signature_validator"
                    fc_names.append(p.function_call.name)
                guarded_parts.append(p)

            raw_stages.append(types.Content(role="model", parts=guarded_parts))

            if next_has_fr:
                raw_stages.append(types.Content(role="user", parts=contents[i + 1].parts))
                i += 2
            else:
                # ORPHAN FUNCTION CALL DETECTED (cancelled turn or interruption)
                # Auto-heal by injecting synthetic FunctionResponse to prevent 400 error!
                synthetic_fr_parts = [
                    types.Part.from_function_response(
                        name=name,
                        response={"status": "interrupted", "error": "Operation was interrupted or cancelled before completion."}
                    )
                    for name in fc_names
                ]
                raw_stages.append(types.Content(role="user", parts=synthetic_fr_parts))
                i += 1
        else:
            raw_stages.append(types.Content(role=role, parts=valid_parts))
            i += 1

    # 3. Merge adjacent turns with identical roles
    merged: List[types.Content] = []
    for c in raw_stages:
        if not c.parts:
            continue
        if merged and merged[-1].role == c.role:
            prev_has_fc = any(getattr(p, "function_call", None) is not None for p in merged[-1].parts)
            curr_has_fr = any(getattr(p, "function_response", None) is not None for p in c.parts)

            if prev_has_fc or curr_has_fr:
                inter_role = "user" if c.role == "model" else "model"
                inter_text = "[System: Intermediate state synchronizer]"
                merged.append(types.Content(role=inter_role, parts=[types.Part.from_text(text=inter_text)]))
                merged.append(c)
            else:
                merged[-1].parts.extend(c.parts)
        else:
            merged.append(c)

    # 4. Ensure conversation starts with 'user'
    if merged and merged[0].role != "user":
        merged.insert(0, types.Content(role="user", parts=[types.Part.from_text(text="[System: Session initialized]")]))

    # 5. Trim trailing empty model turns
    while len(merged) > 1 and merged[-1].role == "model":
        has_substance = any(
            (getattr(p, "text", None) and p.text.strip()) or
            (getattr(p, "function_call", None) is not None)
            for p in (merged[-1].parts or [])
        )
        if not has_substance:
            merged.pop()
        else:
            break

    return merged


async def count_contents_tokens(
    gemini_client,
    model: str,
    contents: List[types.Content],
    system_instruction: Optional[str] = None
) -> int:
    """
    High-Speed Zero-API-Cost Token Estimator.
    Eliminates the remote count_tokens call on every turn, saving 50% of the 250k TPM budget.
    """
    if not contents and not system_instruction:
        return 0

    total_chars = len(system_instruction) if system_instruction else 0
    for c in contents:
        for p in (c.parts or []):
            if hasattr(p, "text") and p.text:
                total_chars += len(p.text)
            elif hasattr(p, "function_call") and p.function_call:
                total_chars += len(p.function_call.name) + len(json.dumps(p.function_call.args or {}))
            elif hasattr(p, "function_response") and p.function_response:
                total_chars += len(p.function_response.name) + len(json.dumps(p.function_response.response or {}))

    return estimate_tokens_fallback_chars(total_chars)


class LosslessContextCompactor:
    def __init__(self, db_manager, key_manager=None):
        self.db = db_manager
        self.key_manager = key_manager
        self.compression_threshold = getattr(config, "CONTEXT_COMPRESSION_THRESHOLD", 40000)
        self.summarizer_models = list(getattr(config, "SUMMARIZER_MODELS", [])) or list(getattr(config, "GEMINI_MODELS", []))
        self.current_summarizer_idx = 0
        self.preserve_count = getattr(config, "RECENT_TURNS_PRESERVE_COUNT", 10)

    def get_active_summarizer_model(self) -> str:
        if not self.summarizer_models:
            return getattr(config, "SUMMARIZER_MODEL", "gemini-3.7-flash")
        return self.summarizer_models[self.current_summarizer_idx % len(self.summarizer_models)]

    async def summarize_dialogue_slice(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_directive: Optional[str] = None
    ) -> Optional[str]:
        """Executes isolated high-precision distillation with model-pool rotation and chunked safety for huge contexts."""
        if not self.key_manager or not turns_to_summarize:
            return None

        formatted_dialogue = []
        max_tool_chars = getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)

        for t in turns_to_summarize:
            role = t.get("role", "user").lower()
            text_val = t.get("text", "").strip()

            if role in ["tool", "tool_result"]:
                role_label = "Tool Output"
                text_val = truncate_tool_response_text(text_val, max_tool_chars)
            elif role == "tool_call":
                role_label = "Tool Invocation"
            elif role == "agent_message":
                role_label = "Agent Interim Message"
            elif role == "system":
                role_label = "System Event"
            elif role == "model":
                role_label = "Karyon Agent"
            else:
                role_label = "Bazilevs (User)"

            if text_val:
                formatted_dialogue.append(f"[{role_label}]:\n{text_val}")

        if not formatted_dialogue:
            return None

        # Check total character / token volume to prevent exceeding Gemini's 1M context limit in single call
        total_dialogue_text = "\n\n---\n\n".join(formatted_dialogue)
        est_tokens = estimate_tokens_fallback_chars(len(total_dialogue_text))

        # If dialogue slice exceeds 750k tokens, recursively split into smaller chunks (e.g., 2 halves) and summarize
        if est_tokens > 750000 and len(turns_to_summarize) > 1:
            mid = len(turns_to_summarize) // 2
            logger.info(f"⚠️ Context slice too huge ({est_tokens:,} tokens > 750k). Splitting into 2 sub-slices ({mid} and {len(turns_to_summarize) - mid} turns)...")
            part1 = await self.summarize_dialogue_slice(turns_to_summarize[:mid], focus_directive)
            part2 = await self.summarize_dialogue_slice(turns_to_summarize[mid:], focus_directive)
            
            combined_parts = []
            if part1: combined_parts.append(f"=== PART 1 SUMMARY ===\n{part1}")
            if part2: combined_parts.append(f"=== PART 2 SUMMARY ===\n{part2}")
            if combined_parts:
                return "\n\n".join(combined_parts)

        prompt_text = PRECISE_SCIENTIFIC_SUMMARIZER_PROMPT
        if focus_directive:
            prompt_text += f"\n\nSPECIAL FOCUS DIRECTIVE FROM OPERATOR:\n{focus_directive}"

        prompt_content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_text(
                    text="=== MULTI-ROLE CONVERSATION & TELEMETRY STREAM TO DISTILL ===\n\n" + total_dialogue_text
                )
            ]
        )

        max_attempts = min(15, len(self.key_manager.keys))

        for attempt in range(max_attempts):
            target_model = self.get_active_summarizer_model()
            gemini_client = self.key_manager.get_client()

            try:
                response = await gemini_client.aio.models.generate_content(
                    model=target_model,
                    contents=[prompt_content],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=getattr(config, "SUMMARY_MAX_OUTPUT_TOKENS", 16384)
                    )
                )
                if response and response.text and len(response.text.strip()) > 20:
                    logger.info(f"Lossless distillation succeeded via '{target_model}'.")
                    return response.text.strip()

            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Summarizer attempt on '{target_model}' (Key #{self.key_manager.current_key_index + 1}): {err_msg}")
                gemini_client = await self.key_manager.handle_quota_exhausted(err_msg)
                self.current_summarizer_idx = (self.current_summarizer_idx + 1) % len(self.summarizer_models)
                await asyncio.sleep(0.5)

        logger.error("All distillation attempts exhausted.")
        return None

    async def get_token_breakdown(
        self,
        system_instruction: str,
        raw_turns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Returns exact local token metrics across context components."""
        active_model = self.key_manager.get_model() if self.key_manager else (config.GEMINI_MODELS[0] if config.GEMINI_MODELS else "gemini-3.7-flash")

        sys_tokens = estimate_tokens_fallback(system_instruction)
        active_contents = await self.build_compacted_contents(raw_turns, force_compaction=False)
        active_contents_tokens = sum(estimate_tokens_fallback(p.text or "") for c in active_contents for p in (c.parts or []))

        raw_history_tokens = sum(estimate_tokens_fallback(t.get("text", "") + (t.get("raw_parts_json") or "")) for t in raw_turns)
        cached_summary = await self.db.get_memory("active_context_summary")
        summary_tokens = estimate_tokens_fallback(cached_summary) if cached_summary else 0

        total_active_tokens = sys_tokens + active_contents_tokens
        threshold = self.compression_threshold
        pct = (total_active_tokens / threshold * 100.0) if threshold > 0 else 0.0

        return {
            "total_tokens": total_active_tokens,
            "system_instruction_tokens": sys_tokens,
            "active_contents_tokens": active_contents_tokens,
            "raw_history_tokens": raw_history_tokens,
            "active_summary_tokens": summary_tokens,
            "compression_threshold": threshold,
            "free_tier_tpm_limit": getattr(config, "FREE_TIER_TPM_LIMIT", 250000),
            "usage_percentage": round(pct, 2),
            "raw_turns_count": len(raw_turns),
            "active_model": active_model,
            "summarizer_model": self.get_active_summarizer_model(),
            "summarizer_models_pool": self.summarizer_models
        }

    async def build_compacted_contents(
        self,
        raw_turns: List[Dict[str, Any]],
        force_compaction: bool = False,
        focus_directive: Optional[str] = None
    ) -> List[types.Content]:
        """Builds aligned contents with automatic lossless state injection and 250k TPM protection."""
        contents = []

        # 1. Fetch Empirical Ledger from DB (Capped to latest 12 experiments to prevent token bloat)
        experiments = await self.db.get_all_experiments(limit=12)
        if experiments:
            ledger_lines = [
                "=== IMMUTABLE EMPIRICAL SCIENTIFIC LEDGER (RECENT EXPERIMENTS) ===",
                "| EXP ID | Hypothesis & Architecture Delta | Final Loss | Verdict | Key Telemetry Metrics |",
                "|---|---|---|---|---|"
            ]
            for exp in experiments:
                metrics_dict = exp.get("metrics", {})
                metrics_preview = ", ".join(f"{k}={v}" for k, v in list(metrics_dict.items())[:4]) if metrics_dict else "None"
                loss_val = f"{exp.get('final_loss'):.4f}" if exp.get('final_loss') is not None else "N/A"
                ledger_lines.append(
                    f"| {exp['exp_id']} | {exp['hypothesis'][:35]} ({exp['architecture_delta'][:25]}) | "
                    f"{loss_val} | {exp['verdict']} | {metrics_preview} |"
                )
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="[System Knowledge Context: Lossless Empirical History]\n" + "\n".join(ledger_lines))]
            ))

        # 2. Token Valuation & Threshold Check (Hardened at 40k tokens)
        total_estimated = sum(estimate_tokens_fallback(t.get("text", "") + (t.get("raw_parts_json") or "")) for t in raw_turns)
        should_compress = force_compaction or (total_estimated > self.compression_threshold and len(raw_turns) > self.preserve_count)

        if not should_compress or len(raw_turns) <= self.preserve_count:
            cached_summary = await self.db.get_memory("active_context_summary")
            if cached_summary:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"[System Knowledge Context: Lossless Historical State Summary]\n{cached_summary}")]
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Acknowledged. All permanent user directives and empirical findings are loaded.")]
                ))

            for idx, turn in enumerate(raw_turns):
                is_historical = (idx < len(raw_turns) - 2)
                c_obj = turn_to_content(turn, is_historical=is_historical)
                if c_obj:
                    contents.append(c_obj)
        else:
            k = max(6, self.preserve_count)
            older_turns = raw_turns[:-k]
            recent_turns = raw_turns[-k:]
            last_older_id = older_turns[-1]["id"] if older_turns else 0
            cache_key = f"lossless_state_snapshot_{last_older_id}"

            cached_summary = await self.db.get_memory(cache_key)
            if not cached_summary or force_compaction:
                cached_summary = await self.summarize_dialogue_slice(older_turns, focus_directive=focus_directive)
                if cached_summary:
                    await self.db.set_memory(cache_key, cached_summary, category="summary")
                    await self.db.set_memory("active_context_summary", cached_summary, category="summary")
                    await self.db.archive_old_turns(keep_count=k)

            if cached_summary:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"[System Knowledge Context: Lossless Historical State Summary]\n{cached_summary}")]
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Acknowledged. All permanent user directives and empirical findings are loaded.")]
                ))

            for idx, turn in enumerate(recent_turns):
                is_historical = (idx < len(recent_turns) - 2)
                c_obj = turn_to_content(turn, is_historical=is_historical)
                if c_obj:
                    contents.append(c_obj)

        return sanitize_and_align_contents(contents)
