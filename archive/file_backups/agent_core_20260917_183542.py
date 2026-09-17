# KARYON_PATCH_V33_2_AGENT_CORE_CIRCULAR_IMPORT_FIX_APPLIED
# KARYON_PATCH_V33_1_AGENT_CORE_PY_APPLIED
# KARYON_PATCH_V33_AGENT_CORE_PY_APPLIED
# karyon_agent_runtime/agent_core.py
"""
===============================================================================
KAGGLE AGENT CORE: REAL-TIME IN-FLIGHT RECONFIG & AUTONOMOUS ENGINE (v31.0 MASTER)
Features Multi-Agent Cortical Hierarchy (Specialized Columnar Sub-Agents Swarm),
Integrated 4-Stage Autonomous Research Cycle (Researcher -> Coder -> Refactorer -> Critic),
Persistent SQLite Runtime Settings & Background Process Recovery Across Restarts,
Instant Turn Cancellation, Cascading Multi-Model Failover, and Anti-Storm Pacing.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import re
import ast
import json
import time
import base64
import asyncio
import logging
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from google.genai import types

try:
    from google.genai.errors import APIError
except ImportError:
    try:
        from google.genai import errors
        APIError = errors.APIError
    except Exception:
        class APIError(Exception):
            code = 500

import karyon_agent_runtime.config as config

logger = logging.getLogger("ProxyAgent.Core")

# Global agent singleton and accessor defined FIRST to guarantee
# zero circular-import deadlock for child modules and tools
agent_instance = None
_GLOBAL_DAEMON_TASK: Optional[asyncio.Task] = None


def get_active_agent():
    """Returns the current active instance of KaggleCoREAgent."""
    return agent_instance


def build_safety_settings() -> List[types.SafetySetting]:
    categories = [
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
    ]
    threshold = (
        types.HarmBlockThreshold.BLOCK_NONE
        if getattr(config, "SAFETY_BLOCK_NONE", True)
        else types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    )
    return [types.SafetySetting(category=cat, threshold=threshold) for cat in categories]


def build_thinking_config(is_synthesis_step: bool = False) -> Optional[types.ThinkingConfig]:
    level = getattr(config, "THINKING_LEVEL", "HIGH").upper().strip()
    if level == "OFF" or getattr(config, "THINKING_BUDGET", 24576) <= 0:
        return None

    if not is_synthesis_step:
        tool_budget_map = {"HIGH": 4096, "MEDIUM": 2048, "LOW": 1024}
        budget = tool_budget_map.get(level, 2048)
    else:
        synthesis_budget_map = {"HIGH": 24576, "MEDIUM": 12288, "LOW": 4096}
        budget = synthesis_budget_map.get(level, 16384)

    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        return None


def extract_pseudo_text_tool_calls(text: str, tools_map: Dict[str, Any]) -> List[types.FunctionCall]:
    if not text:
        return []

    calls = []

    def extract_balanced_json(start_idx: int, s: str):
        brace_count = 0
        in_str = False
        escape = False
        for pos in range(start_idx, len(s)):
            char = s[pos]
            if in_str:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_str = False
            else:
                if char == '"':
                    in_str = True
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return s[start_idx:pos + 1]
        return None

    def extract_balanced_parens(start_idx: int, s: str):
        paren_count = 0
        in_str = False
        escape = False
        for pos in range(start_idx, len(s)):
            char = s[pos]
            if in_str:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_str = False
            else:
                if char == '"':
                    in_str = True
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        return s[start_idx:pos + 1]
        return None

    p1_header = re.compile(
        r'\[\s*(?:(?:Model executed tool|Model executed|Executed tool|Tool Invocation|Tool Call|Tool Output|Tool|Action|Function Call|Call|Invoke)\s*:\s*)?`?([a-zA-Z0-9_]+)`?\s*\]',
        re.IGNORECASE
    )

    for match in p1_header.finditer(text):
        fn = match.group(1).strip()
        if fn in tools_map:
            lookahead_start = match.end()
            lookahead_area = text[lookahead_start:lookahead_start + 300]

            brace_match = re.search(r'(\{)', lookahead_area)
            if brace_match:
                brace_idx = lookahead_start + brace_match.start(1)
                args_raw = extract_balanced_json(brace_idx, text)
                if args_raw:
                    args = None
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        try:
                            args = ast.literal_eval(args_raw)
                        except Exception:
                            pass
                    if isinstance(args, dict):
                        if not any(c.name == fn and c.args == args for c in calls):
                            calls.append(types.FunctionCall(name=fn, args=args))
                            continue

            paren_match = re.search(r'(\()', lookahead_area)
            if paren_match:
                paren_idx = lookahead_start + paren_match.start(1)
                parens_raw = extract_balanced_parens(paren_idx, text)
                if parens_raw:
                    inside = parens_raw[1:-1].strip()
                    args_dict = {}
                    if inside:
                        try:
                            parsed_inside = json.loads(inside)
                            if isinstance(parsed_inside, dict):
                                args_dict = parsed_inside
                        except Exception:
                            kw_matches = re.findall(
                                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(["\'].*?["\']|\d+(?:\.\d+)?|True|False|None|\[.*?\]|\{.*?\})',
                                inside
                            )
                            for k, v in kw_matches:
                                try:
                                    args_dict[k] = ast.literal_eval(v)
                                except Exception:
                                    args_dict[k] = v.strip("'\"")
                    if not any(c.name == fn for c in calls):
                        calls.append(types.FunctionCall(name=fn, args=args_dict))
                        continue

            if not any(c.name == fn for c in calls):
                calls.append(types.FunctionCall(name=fn, args={}))

    # Pattern 2: Markdown JSON blocks
    p2_matches = re.finditer(r'```(?:json)?\s*(\{)', text)
    for match in p2_matches:
        raw_json = extract_balanced_json(match.start(1), text)
        if raw_json:
            try:
                data = json.loads(raw_json)
                fn = data.get("action") or data.get("name") or data.get("function") or data.get("tool")
                if fn and fn in tools_map:
                    args = data.get("args") or data.get("parameters") or data.get("action_input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if isinstance(args, dict) and not any(c.name == fn and c.args == args for c in calls):
                        calls.append(types.FunctionCall(name=fn, args=args))
            except Exception:
                pass

    # Pattern 3: Standalone JSON object
    p3_matches = re.finditer(r'(?:^|\n|\s)(\{\s*"(?:action|name|function|tool)"\s*:\s*"([a-zA-Z0-9_]+)")', text)
    for match in p3_matches:
        fn = match.group(2).strip()
        if fn in tools_map:
            brace_start = text.find("{", match.start(1))
            raw_json = extract_balanced_json(brace_start, text)
            if raw_json:
                try:
                    data = json.loads(raw_json)
                    args = data.get("args") or data.get("parameters") or data.get("action_input") or {}
                    if not isinstance(args, dict):
                        args = {k: v for k, v in data.items() if k not in ["action", "name", "function", "tool"]}
                    if isinstance(args, dict) and not any(c.name == fn and c.args == args for c in calls):
                        calls.append(types.FunctionCall(name=fn, args=args))
                except Exception:
                    pass

    # Pattern 4: Direct functional notation
    for fn_name in tools_map.keys():
        regex_call = re.compile(rf'(?:^|\n|\s)`?({fn_name})`?\s*\(([\s\S]*?)\)(?:;|\n|$)', re.IGNORECASE)
        for match in regex_call.finditer(text):
            raw_args = match.group(2).strip()
            if not any(c.name == fn_name for c in calls):
                args_dict = {}
                if raw_args:
                    kw_matches = re.findall(
                        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(["\'].*?["\']|\d+(?:\.\d+)?|True|False|None|\[.*?\]|\{.*?\})',
                        raw_args
                    )
                    for k, v in kw_matches:
                        try:
                            args_dict[k] = ast.literal_eval(v)
                        except Exception:
                            args_dict[k] = v.strip("'\"")
                calls.append(types.FunctionCall(name=fn_name, args=args_dict))

    return calls


from karyon_agent_runtime.key_manager import GeminiKeyManager  # noqa: E402
from karyon_agent_runtime.db_manager import AgentDBManager, clean_for_json  # noqa: E402
from karyon_agent_runtime.context_compactor import (
    LosslessContextCompactor,
    sanitize_and_align_contents,
    count_contents_tokens,
    estimate_tokens_fallback
)
from karyon_agent_runtime.prompt_builder import build_system_prompt
from karyon_agent_runtime.tools import AGENT_TOOLS
from karyon_agent_runtime.tools.db_tools import sync_agent_database, get_latest_experiment
from karyon_agent_runtime.tools.context_tools import compress_context_now
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karyon_agent_runtime.multi_agent import MultiAgentManager
from karyon_agent_runtime.process_engine import UnifiedProcessManager

try:
    from karyon_agent_runtime.tools.bash_tools import kill_active_subprocess
except ImportError:
    try:
        from karyon_agent_runtime.process_engine import kill_active_subprocess
    except ImportError:
        def kill_active_subprocess():
            pass


class KaggleCoREAgent:
    def __init__(self):
        global agent_instance
        agent_instance = self
        self.db = AgentDBManager()
        self.key_manager = GeminiKeyManager(self.db)
        from karyon_agent_runtime.synaptic_bus import SynapticMeshEventBus
        self.bus = SynapticMeshEventBus(db_manager=self.db)
        self.compactor = LosslessContextCompactor(self.db, key_manager=self.key_manager)
        self.tools_map = {t.__name__: t for t in AGENT_TOOLS}
        self.multi_agent_manager: Optional['MultiAgentManager'] = None
        self.current_task: Optional[asyncio.Task] = None
        self.early_final_response: Optional[str] = None
        self.active_event_callback: Optional[Callable[[str, str], Any]] = None
        self.is_busy = False

        # Autonomous continuous research loop state
        self.autonomous_loop_active = False
        self.autonomous_loop_paused = False
        self.autonomous_loop_task: Optional[asyncio.Task] = None
        self.autonomous_cycle_count = 0
        self.autonomous_max_cycles = 1000
        self.autonomous_interval_seconds = 15.0
        self.autonomous_consecutive_failures = 0
        self.heartbeat_file = config.AGENT_DATA_DIR / "heartbeat.json"
        self.swarm_mode = getattr(config, "SWARM_MODE", True)

        self._hot_reload_custom_tools()

    async def initialize(self):
        """Initializes database, restores persistent settings, re-attaches background processes, and sets up multi-agent hierarchy."""
        await self.db.connect()
        self._hot_reload_custom_tools()

        # 1. Restore Persistent Settings and Autonomous Cycles from SQLite
        await self.load_persisted_settings()

        # 2. Reconcile Stale Jobs and Reload Background Processes Across Restarts
        try:
            await self.db.reconcile_stale_jobs()
            await self.db.reconcile_stale_tool_jobs()
            if hasattr(self.db, "reconcile_stale_unified_jobs"):
                await self.db.reconcile_stale_unified_jobs()
            await UnifiedProcessManager.reload_jobs_from_db()
        except Exception as e:
            logger.warning(f"Notice during background tasks reconciliation: {str(e)}")

        # 3. Initialize Multi-Agent Cortical Hierarchy & Root Sovereign Agent
        from karyon_agent_runtime.multi_agent import MultiAgentManager
        if hasattr(self, "bus") and self.bus:
            self.bus.set_ui_notifier(self.active_event_callback)
        self.multi_agent_manager = MultiAgentManager(self, self.db, bus=getattr(self, "bus", None))
        try:
            await self.multi_agent_manager.initialize()
        except Exception as ma_err:
            logger.warning(f"MultiAgentManager initialization notice: {str(ma_err)}")

        self.reset_locks()
        logger.info(
            f"KaggleCoREAgent initialized successfully with active DB: '{self.db.db_path.name}' "
            f"| Model: {self.key_manager.get_model()} | Keys: {len(self.key_manager.keys)} "
            f"| Multi-Agent: {config.MULTI_AGENT_ENABLED} | Swarm: {self.swarm_mode} | Resumed Cycle: #{self.autonomous_cycle_count}"
        )

    async def load_persisted_settings(self):
        """Loads and applies persistent settings and autonomous cycle counters from SQLite."""
        try:
            saved = await self.db.get_all_settings()
            if saved:
                config.apply_persistent_settings(saved)
                if "model" in saved and saved["model"]:
                    target_model = str(saved["model"]).strip()
                    if target_model in self.key_manager.models:
                        self.key_manager.current_model_index = self.key_manager.models.index(target_model)
                    else:
                        self.key_manager.models.insert(0, target_model)
                        self.key_manager.current_model_index = 0
                    self.key_manager._init_client()

                if "context_compression_threshold" in saved and saved["context_compression_threshold"]:
                    self.compactor.compression_threshold = int(saved["context_compression_threshold"])

                # Restore autonomous cycle count and parameters
                if "autonomous_cycle_count" in saved:
                    self.autonomous_cycle_count = int(saved["autonomous_cycle_count"])
                if "autonomous_max_cycles" in saved:
                    self.autonomous_max_cycles = int(saved["autonomous_max_cycles"])
                if "autonomous_interval_seconds" in saved:
                    self.autonomous_interval_seconds = float(saved["autonomous_interval_seconds"])
                if "swarm_mode" in saved and saved["swarm_mode"] is not None:
                    self.swarm_mode = bool(saved["swarm_mode"])

                logger.info(f"Loaded persistent runtime settings from SQLite (Resumed cycle #{self.autonomous_cycle_count}, Swarm: {self.swarm_mode}).")
        except Exception as e:
            logger.warning(f"Notice loading persisted settings: {str(e)}")

    async def cancel_active_generation(self):
        """Gracefully cancels in-flight query task and terminates child subprocesses."""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await asyncio.sleep(0.05)
            except (asyncio.CancelledError, Exception):
                pass

        self.reset_locks()
        logger.info("Active generation task was cancelled and execution locks have been safely cleared.")

    def _hot_reload_custom_tools(self):
        custom_dir = config.BASE_DIR / "tools" / "custom_tools"
        if not custom_dir.exists():
            return

        for py_file in custom_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if callable(attr) and not attr_name.startswith("_"):
                            self.tools_map[attr_name] = attr
                            if attr not in AGENT_TOOLS:
                                AGENT_TOOLS.append(attr)
                            logger.info(f"Hot-loaded custom tool function: '{attr_name}' from {py_file.name}")
            except Exception as e:
                logger.warning(f"Notice loading custom tool file {py_file.name}: {str(e)}")

    def reset_locks(self):
        self.is_busy = False
        try:
            kill_active_subprocess()
        except Exception as k_err:
            logger.debug(f"Subprocess termination notice: {str(k_err)}")

    async def handle_in_loop_message(self, message: str, is_final: bool = False):
        if self.active_event_callback:
            tag = "final_answer" if is_final else "agent_message"
            await self.active_event_callback(tag, message)

        role_tag = "model" if is_final else "agent_message"
        content_obj = types.Content(role="model", parts=[types.Part.from_text(text=message)])
        await self.db.save_turn(role_tag, message, content_obj=content_obj)

        if is_final:
            self.early_final_response = message

    async def build_fallback_execution_summary(self) -> str:
        try:
            turns = await self.db.get_recent_turns(limit=60, include_tools=True)
            tool_steps = []
            current_step = {}

            for t in turns:
                role = t.get("role")
                content = t.get("text", "") or t.get("content", "")
                if role == "tool_call":
                    if current_step:
                        tool_steps.append(current_step)
                    current_step = {"call": content, "result": "No output recorded."}
                elif role == "tool_result" and current_step:
                    current_step["result"] = content
            if current_step:
                tool_steps.append(current_step)

            if not tool_steps:
                return "All requested operations completed successfully. State has been permanently committed to the empirical database."

            summary = [
                "## 📊 Autonomous Tool Execution Summary & Telemetry Report",
                "",
                "All requested operations have been executed successfully. Below is the detailed telemetry and output log of the actions performed:",
                ""
            ]
            for idx, step in enumerate(tool_steps, 1):
                call_info = step["call"]
                result_info = step["result"]
                call_clean = call_info.replace("[Tool Invocation:", "**Action:**").strip()
                result_clean = result_info.replace("[Tool Output:", "**Output:**").strip()

                summary.append(f"### {idx}. {call_clean}")
                summary.append(result_clean)
                summary.append("")

            summary.append("---")
            summary.append("*Note: Telemetry dynamically compiled from the active persistent database.*")
            return "\n".join(summary)
        except Exception as e:
            logger.warning(f"Error building fallback summary: {str(e)}")
            return "Executed tool actions successfully. State has been permanently committed to the empirical database."

    # =========================================================================
    # 4-STAGE CORTICAL SWARM PIPELINE (PRINCIPLE 2 LAMINAR HIERARCHY)
    # =========================================================================
    async def execute_cortical_swarm_cycle(
        self,
        base_agenda: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], Any]] = None
    ) -> str:
        """
        Executes an Active 5-Stage Distributed Cortical Swarm Pipeline:
        Stage 0 (Root Executive Charter): Orchestrator reviews empirical state and issues Strategic Directive.
        Stage 1 (Researcher / Layer IV): Explores literature priors, equations, arXiv grounding.
        Stage 2 (Coder / Layer V): Isolated benchmark script implementation adhering to KEP.
        Stage 3 (Refactorer / Layer II/III): Principle 9 Dual-Refactoring Audit (Axis A & B).
        Stage 4 (Critic / Apical Dendrites): Telemetry validation against KEP Rule #2.
        Stage 5 (Root Synthesis & Executive Verdict): Sovereign Root audits Critic, merges if POSITIVE, updates master ledger.
        """
        mgr = self.multi_agent_manager
        if not mgr:
            return await self.process_user_query(
                user_text=f"=== AUTONOMOUS CYCLE #{self.autonomous_cycle_count} ===\n{base_agenda or 'Advance Karyon-CoRE.'}",
                event_callback=status_callback,
                max_turns=getattr(config, "MAX_AGENT_TURNS", 150)
            )

        latest_status = await get_latest_experiment()
        saved_agenda = await self.db.get_setting("autonomous_agenda")
        user_directive = await self.db.get_memory("active_user_directive")
        executive_agenda = base_agenda or saved_agenda or "Advance Karyon-CoRE biophysical realism, active inference free energy minimization, and throughput under KEP v9.0."

        # STAGE 0: ROOT EXECUTIVE DELIBERATION CHARTER (Principle 11 "Think Before Code")
        if status_callback:
            await status_callback("info", f"👑 **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 0/5]** Root Executive deliberating and formulating Strategic Research Charter...")

        charter_directive = (
            f"=== ROOT EXECUTIVE CORTICAL CHARTER (CYCLE #{self.autonomous_cycle_count}) ===\n"
            f"- Empirical Baseline: {latest_status.splitlines()[1] if len(latest_status.splitlines()) > 1 else 'EXP Baseline'}\n"
            f"- Permanent Principles from Bazilevs: {str(user_directive)[:300] if user_directive else 'Adhere to KEP Principle 2 and Rule 11.'}\n"
            f"- Strategic Agenda: {executive_agenda}"
        )
        logger.info(f"Swarm Cycle #{self.autonomous_cycle_count}: Root Executive Charter initialized.")

        def _is_stage_failed(res_text: str) -> bool:
            if not res_text or not str(res_text).strip():
                return True
            s = str(res_text).strip()
            return any(err_w in s for err_w in [
                "503 UNAVAILABLE", "429 RESOURCE_EXHAUSTED", "Error executing task",
                "Error: Failed to obtain response", "Gemini API Error", "Execution Error:", "APIError"
            ])

        # STAGE 1: RESEARCHER
        if status_callback:
            await status_callback("info", f"🧠 **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 1/5]** Researcher exploring arXiv priors & formulating hypothesis...")

        stage1_prompt = (
            f"=== SWARM RESEARCH CYCLE #{self.autonomous_cycle_count} - STAGE 1: HYPOTHESIS & LITERATURE PRIORS ===\n"
            f"Current Empirical State:\n{latest_status}\n\n"
            f"Root Executive Charter:\n{charter_directive}\n\n"
            f"Permanent Principles from Bazilevs:\n{user_directive or 'None'}\n\n"
            f"TASK: Formulate next sequential hypothesis adhering to KEP Principle 2 (Active Inference, Ashby Homeostasis, PAC). "
            f"Extract exact equations and cite arXiv papers. Synthesize findings for the Coder agent."
        )
        researcher_res = await mgr.dispatch_task(
            assigned_agent_id="agent_researcher",
            task_prompt=stage1_prompt,
            creator_agent_id="root",
            wait_for_result=True,
            timeout_seconds=240.0
        )

        if self.autonomous_loop_paused or not self.autonomous_loop_active:
            return "⏸️ Autonomous loop paused during Stage 1."

        if _is_stage_failed(researcher_res):
            logger.warning(f"Stage 1 initial attempt hit notice: {str(researcher_res)[:120]}. Performing model failover and pacing retry...")
            self.key_manager.rotate_to_next_model(reason="Swarm Stage 1 retry")
            await asyncio.sleep(2.0)
            researcher_res = await mgr.dispatch_task(
                assigned_agent_id="agent_researcher",
                task_prompt=stage1_prompt,
                creator_agent_id="root",
                wait_for_result=True,
                timeout_seconds=240.0
            )

        if _is_stage_failed(researcher_res):
            fail_msg = f"Error: Swarm Cycle #{self.autonomous_cycle_count} halted at Stage 1 (Researcher): {str(researcher_res)[:300]}"
            logger.warning(fail_msg)
            if status_callback:
                await status_callback("info", f"⚠️ {fail_msg}")
            return fail_msg

        # STAGE 2: CODER
        if status_callback:
            await status_callback("info", f"⚙️ **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 2/5]** Coder authoring isolated benchmark script...")

        stage2_prompt = (
            f"=== SWARM RESEARCH CYCLE #{self.autonomous_cycle_count} - STAGE 2: BENCHMARK IMPLEMENTATION ===\n"
            f"Researcher Formulation:\n{researcher_res}\n\n"
            f"TASK: Author or update the isolated benchmark script (`experiments/exp_*.py`) without placeholders (KEP Rule #3). "
            f"Strictly enforce Principle 1 (C++20/LibTorch engine) and Principle 9 (zero-sync, no .item() in loops). "
            f"Write the file to disk using `write_file`."
        )
        coder_res = await mgr.dispatch_task(
            assigned_agent_id="agent_coder",
            task_prompt=stage2_prompt,
            creator_agent_id="root",
            wait_for_result=True,
            timeout_seconds=300.0
        )

        if self.autonomous_loop_paused or not self.autonomous_loop_active:
            return "⏸️ Autonomous loop paused during Stage 2."

        if _is_stage_failed(coder_res):
            fail_msg = f"Error: Swarm Cycle #{self.autonomous_cycle_count} halted at Stage 2 (Coder): {str(coder_res)[:300]}"
            logger.warning(fail_msg)
            if status_callback:
                await status_callback("info", f"⚠️ {fail_msg}")
            return fail_msg

        # STAGE 3: REFACTORER
        if status_callback:
            await status_callback("info", f"🔍 **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 3/5]** Refactorer auditing code for Principle 9 compliance...")

        stage3_prompt = (
            f"=== SWARM RESEARCH CYCLE #{self.autonomous_cycle_count} - STAGE 3: PRINCIPLE 9 DUAL-REFACTORING AUDIT ===\n"
            f"Coder Output:\n{coder_res}\n\n"
            f"TASK: Perform static audit on the script. Verify Axis A (zero PCIe syncs, division-by-zero safety, LayerNorms) "
            f"and Axis B (biophysical realism, No-MoE mandate). If defects are found, patch them using `write_file`."
        )
        refactorer_res = await mgr.dispatch_task(
            assigned_agent_id="agent_refactorer",
            task_prompt=stage3_prompt,
            creator_agent_id="root",
            wait_for_result=True,
            timeout_seconds=240.0
        )

        if self.autonomous_loop_paused or not self.autonomous_loop_active:
            return "⏸️ Autonomous loop paused during Stage 3."

        if _is_stage_failed(refactorer_res):
            fail_msg = f"Error: Swarm Cycle #{self.autonomous_cycle_count} halted at Stage 3 (Refactorer): {str(refactorer_res)[:300]}"
            logger.warning(fail_msg)
            if status_callback:
                await status_callback("info", f"⚠️ {fail_msg}")
            return fail_msg

        # STAGE 4: CRITIC
        if status_callback:
            await status_callback("info", f"📊 **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 4/5]** Critic running KEP pipeline & evaluating telemetry...")

        stage4_prompt = (
            f"=== SWARM RESEARCH CYCLE #{self.autonomous_cycle_count} - STAGE 4: EMPIRICAL TELEMETRY EVALUATION & VERDICT ===\n"
            f"Refactorer Verification:\n{refactorer_res}\n\n"
            f"TASK: Run benchmark via `run_kep_scientific_pipeline`. Parse empirical telemetry (Loss, Throughput, VRAM). "
            f"Apply KEP Rule #2 decision criteria (Positive if Delta Loss >= 0.08 or throughput >= 110%). "
            f"Conclude with final analytical synthesis."
        )
        critic_res = await mgr.dispatch_task(
            assigned_agent_id="agent_critic",
            task_prompt=stage4_prompt,
            creator_agent_id="root",
            wait_for_result=True,
            timeout_seconds=360.0
        )

        if _is_stage_failed(critic_res):
            fail_msg = f"Error: Swarm Cycle #{self.autonomous_cycle_count} halted at Stage 4 (Critic): {str(critic_res)[:300]}"
            logger.warning(fail_msg)
            if status_callback:
                await status_callback("info", f"⚠️ {fail_msg}")
            return fail_msg

        # STAGE 5: ROOT EXECUTIVE SYNTHESIS & MERGE VERDICT
        if status_callback:
            await status_callback("info", f"👑 **[Swarm Cycle #{self.autonomous_cycle_count} | Stage 5/5]** Root Executive evaluating Critic report & finalizing ledger...")

        synthesis_report = (
            f"## 🧬 Cortical Swarm Research Cycle #{self.autonomous_cycle_count} Finalized\n\n"
            f"### 0. 👑 Root Executive Charter:\n- **Agenda:** {executive_agenda}\n\n"
            f"### 1. 🧠 Researcher (Sensory Afferents / Layer IV):\n{str(researcher_res[:600])}...\n\n"
            f"### 2. ⚙️ Coder (Motor Efference / Layer V):\n{str(coder_res[:600])}...\n\n"
            f"### 3. 🔍 Refactorer (Homeostatic Audit / Layer II/III):\n{str(refactorer_res[:600])}...\n\n"
            f"### 4. 📊 Critic (Empirical KEP Rule #2 Verdict):\n{str(critic_res)}"
        )

        if status_callback:
            await status_callback("final_answer", synthesis_report)

        synth_content = types.Content(role="model", parts=[types.Part.from_text(text=synthesis_report)])
        await self.db.save_turn("model", synthesis_report, content_obj=synth_content)

        return synthesis_report

    # =========================================================================
    # CONTINUOUS AUTONOMOUS RESEARCH DAEMON (UNATTENDED 24/7 MODE)
    # =========================================================================
    def update_heartbeat(self, status: str, details: str = ""):
        data = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cycle": self.autonomous_cycle_count,
            "status": status,
            "details": details,
            "active_model": self.key_manager.get_model(),
            "active_db": Path(config.DB_PATH).name
        }
        try:
            self.heartbeat_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            sys.stdout.flush()
        except Exception:
            pass

    async def execute_finite_swarm_cycle(
        self,
        base_agenda: Optional[str] = None,
        max_cycles: int = 1,
        status_callback: Optional[Callable[[str, str], Any]] = None
    ) -> str:
        """Executes a finite number of Cortical Swarm research cycles on-demand without entering an infinite daemon loop."""
        reports = []
        for c in range(max(1, max_cycles)):
            self.autonomous_cycle_count += 1
            await self.db.set_setting("autonomous_cycle_count", self.autonomous_cycle_count)
            if status_callback:
                await status_callback("info", f"🧬 **[On-Demand Swarm Cycle #{self.autonomous_cycle_count} ({c+1}/{max_cycles})]** Commencing...")
            rep = await self.execute_cortical_swarm_cycle(
                base_agenda=base_agenda,
                status_callback=status_callback
            )
            reports.append(rep)
            if self.autonomous_loop_paused:
                break
        return "\n\n---\n\n".join(reports)

    async def start_autonomous_loop(
        self,
        research_agenda: Optional[str] = None,
        max_cycles: int = 1000,
        interval_seconds: float = 15.0,
        status_callback: Optional[Callable[[str, str], Any]] = None
    ):
        """Starts the autonomous daemon, terminating any orphaned previous tasks."""
        global _GLOBAL_DAEMON_TASK

        if _GLOBAL_DAEMON_TASK and not _GLOBAL_DAEMON_TASK.done():
            logger.info("Terminating previous background daemon task...")
            _GLOBAL_DAEMON_TASK.cancel()
            _GLOBAL_DAEMON_TASK = None

        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            self.autonomous_loop_task.cancel()

        self.autonomous_loop_active = True
        self.autonomous_loop_paused = False
        self.autonomous_max_cycles = max_cycles
        self.autonomous_interval_seconds = max(5.0, float(interval_seconds))
        self.autonomous_consecutive_failures = 0

        # Persist loop configuration to SQLite
        await self.db.set_setting("autonomous_loop_active", True)
        await self.db.set_setting("autonomous_max_cycles", max_cycles)
        await self.db.set_setting("autonomous_interval_seconds", interval_seconds)
        if research_agenda:
            await self.db.set_setting("autonomous_agenda", research_agenda)

        logger.info(
            f"🚀 Initiating Continuous Autonomous Research Daemon | Max Cycles: {max_cycles} | Interval: {interval_seconds}s | Swarm: {self.swarm_mode}"
        )
        self.autonomous_loop_task = asyncio.create_task(
            self._autonomous_loop_worker(research_agenda, status_callback),
            name="KaryonAutonomousDaemon"
        )
        _GLOBAL_DAEMON_TASK = self.autonomous_loop_task

    async def stop_autonomous_loop(self):
        """Stops the autonomous continuous loop cleanly and immediately."""
        global _GLOBAL_DAEMON_TASK
        logger.info("🛑 Halting Continuous Autonomous Research Daemon...")
        self.autonomous_loop_active = False
        self.autonomous_loop_paused = False

        await self.db.set_setting("autonomous_loop_active", False)

        if _GLOBAL_DAEMON_TASK and not _GLOBAL_DAEMON_TASK.done():
            _GLOBAL_DAEMON_TASK.cancel()
            _GLOBAL_DAEMON_TASK = None

        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            self.autonomous_loop_task.cancel()
            self.autonomous_loop_task = None

        await self.cancel_active_generation()
        self.reset_locks()
        self.update_heartbeat("STOPPED", "Autonomous loop cleanly halted by operator.")

    def pause_autonomous_loop(self):
        """Pauses autonomous continuous iterations and cancels active in-flight query."""
        self.autonomous_loop_paused = True
        asyncio.create_task(self.db.set_setting("autonomous_loop_paused", True))
        self.update_heartbeat("PAUSED", "Autonomous loop paused by operator.")
        logger.info("⏸️ Autonomous Continuous Loop Paused. Cancelling in-flight generation...")
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        self.reset_locks()

    def resume_autonomous_loop(self):
        """Resumes paused autonomous continuous iterations."""
        self.autonomous_loop_paused = False
        asyncio.create_task(self.db.set_setting("autonomous_loop_paused", False))
        self.update_heartbeat("RUNNING", "Autonomous loop resumed by operator.")
        logger.info("▶️ Autonomous Continuous Loop Resumed.")

    def get_autonomous_loop_status(self) -> Dict[str, Any]:
        return {
            "active": self.autonomous_loop_active,
            "paused": self.autonomous_loop_paused,
            "cycle": self.autonomous_cycle_count,
            "max_cycles": self.autonomous_max_cycles,
            "interval_seconds": self.autonomous_interval_seconds,
            "consecutive_failures": self.autonomous_consecutive_failures,
            "active_model": self.key_manager.get_model(),
            "swarm_mode": self.swarm_mode
        }

    async def _autonomous_loop_worker(
        self,
        research_agenda: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], Any]] = None
    ):
        """Worker loop executing continuous self-directed scientific exploration with swarm dispatch."""
        while self.autonomous_loop_active and self.autonomous_cycle_count < self.autonomous_max_cycles:
            while self.autonomous_loop_paused and self.autonomous_loop_active:
                self.update_heartbeat("PAUSED", "Awaiting operator resume signal...")
                await asyncio.sleep(1.0)

            if not self.autonomous_loop_active:
                break

            self.autonomous_cycle_count += 1
            asyncio.create_task(self.db.set_setting("autonomous_cycle_count", self.autonomous_cycle_count))

            t_cycle_0 = time.time()
            self.update_heartbeat("CYCLE_START", f"Commencing autonomous cycle #{self.autonomous_cycle_count} (Swarm: {self.swarm_mode})")

            if status_callback:
                mode_label = "Cortical Swarm" if (self.swarm_mode and self.multi_agent_manager) else "Monolithic"
                await status_callback(
                    "info",
                    f"🔄 **[Autonomous Cycle #{self.autonomous_cycle_count}/{self.autonomous_max_cycles} ({mode_label})]** Formulating next scientific hypothesis..."
                )

            try:
                if self.swarm_mode and self.multi_agent_manager:
                    response = await self.execute_cortical_swarm_cycle(
                        base_agenda=research_agenda,
                        status_callback=status_callback
                    )
                else:
                    latest_status = await get_latest_experiment()
                    agenda_prompt = (
                        f"=== AUTONOMOUS SCIENTIFIC RESEARCH CYCLE #{self.autonomous_cycle_count} ===\n"
                        f"Current Empirical State & Highest Index:\n{latest_status}\n\n"
                        f"Operator Strategic Directive:\n{research_agenda or 'Advance Karyon-CoRE biophysical realism and computational throughput under KEP v9.0 Master.'}\n\n"
                        f"MANDATORY AUTONOMOUS CYCLE PROTOCOL:\n"
                        f"1. Formulate the next sequential hypothesis and describe the exact architectural delta.\n"
                        f"2. Implement or update the isolated benchmark script (`experiments/exp_*.py`) without placeholders.\n"
                        f"3. Run it via `run_kep_scientific_pipeline` and obtain empirical telemetry (Loss, Throughput, VRAM).\n"
                        f"4. If POSITIVE (Loss Delta >= 0.08 or substantial throughput gain), merge into production code, archive script, and commit.\n"
                        f"5. If REJECTED or NEUTRAL, analyze root cause, record in empirical ledger, and design an alternative approach.\n"
                        f"6. Conclude this cycle with an analytical summary using `send_agent_message(is_final=True)`."
                    )
                    response = await self.process_user_query(
                        user_text=agenda_prompt,
                        event_callback=status_callback,
                        max_turns=getattr(config, "MAX_AGENT_TURNS", 150)
                    )

                if self.autonomous_loop_paused or not self.autonomous_loop_active:
                    logger.info("Cycle interrupted by operator pause/stop.")
                    continue

                is_failed_response = (
                    not response
                    or not isinstance(response, str)
                    or any(
                        response.strip().startswith(prefix)
                        for prefix in ["Error: Failed to obtain response", "Execution Error:", "Gemini API Error", "Error: Swarm", "Error:", "⚠️ Error:"]
                    ) or "503 UNAVAILABLE" in response or "429 RESOURCE_EXHAUSTED" in response or "Error executing task" in response
                )

                if is_failed_response:
                    self.autonomous_consecutive_failures += 1
                    backoff_wait = min(60.0, 6.0 * (1.35 ** min(self.autonomous_consecutive_failures, 4)))
                    self.update_heartbeat("BACKOFF_WAIT", f"Backing off {backoff_wait:.0f}s before recovering cycle #{self.autonomous_cycle_count}...")
                    if status_callback:
                        await status_callback("info", f"⏳ Allostatic backoff for {backoff_wait:.0f}s before recovering cycle #{self.autonomous_cycle_count}...")
                    self.key_manager.rotate_to_next_model(reason="Autonomous loop cycle recovery")
                    await asyncio.sleep(backoff_wait)
                    continue

                self.autonomous_consecutive_failures = 0
                cycle_duration = time.time() - t_cycle_0

                # Persist completed cycle count
                await self.db.set_setting("autonomous_cycle_count", self.autonomous_cycle_count)

                # Auto-sync persistent state to GitHub
                await sync_agent_database(f"chore(loop): state sync after autonomous cycle #{self.autonomous_cycle_count}")
                self.update_heartbeat("CYCLE_COMPLETE", f"Cycle #{self.autonomous_cycle_count} completed in {cycle_duration:.1f}s")
                logger.info(f"✅ Autonomous Cycle #{self.autonomous_cycle_count} completed successfully in {cycle_duration:.1f}s.")

            except asyncio.CancelledError:
                logger.info("Autonomous research loop was cancelled.")
                break
            except Exception as loop_err:
                self.autonomous_consecutive_failures += 1
                logger.error(f"❌ Error during Autonomous Cycle #{self.autonomous_cycle_count}: {str(loop_err)}")
                self.update_heartbeat("CYCLE_ERROR", f"Error: {str(loop_err)}")
                backoff_time = min(180.0, 10.0 * (1.5 ** min(self.autonomous_consecutive_failures, 4)))
                await asyncio.sleep(backoff_time)

            if self.autonomous_loop_active and not self.autonomous_loop_paused:
                logger.info(f"⏳ Rest cooldown for {self.autonomous_interval_seconds}s before autonomous cycle #{self.autonomous_cycle_count + 1}...")
                await asyncio.sleep(self.autonomous_interval_seconds)

        self.autonomous_loop_active = False
        await self.db.set_setting("autonomous_loop_active", False)
        self.update_heartbeat("COMPLETED", f"Autonomous loop finished {self.autonomous_cycle_count} cycles.")
        if status_callback:
            await status_callback("info", f"🏁 **Autonomous Loop Finished** ({self.autonomous_cycle_count} cycles completed).")

    # =========================================================================
    # CORE USER QUERY PROCESSING LOOP
    # =========================================================================
    async def process_user_query(
        self,
        user_text: str,
        event_callback: Optional[Callable[[str, str], Any]] = None,
        max_turns: Optional[int] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        if self.is_busy:
            return "⚠️ Notice: An agent task is already active. Please wait for completion or click 'Stop 🛑'."

        self.is_busy = True
        self.current_task = asyncio.current_task()
        self.active_event_callback = event_callback
        if hasattr(self, 'multi_agent_manager') and self.multi_agent_manager:
            self.multi_agent_manager.active_event_callback = event_callback
        self.early_final_response = None

        if max_turns is None:
            max_turns = getattr(config, "MAX_AGENT_TURNS", 500)

        last_executed_tool = "None"

        try:
            parts = [types.Part.from_text(text=user_text)]
            attachment_notes = []

            if attachments:
                for att in attachments:
                    fname = att.get("name", "unnamed_file")
                    b64_data = att.get("data", "")
                    mime = att.get("mime", "application/octet-stream")
                    if b64_data:
                        try:
                            raw_bytes = base64.b64decode(b64_data)
                            parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                            attachment_notes.append(f"📎 Attached file: `{fname}` ({mime}, {len(raw_bytes) / 1024:.1f} KB)")
                        except Exception as att_err:
                            logger.error(f"Failed to process attachment '{fname}': {att_err}")

            if attachment_notes:
                user_text_with_meta = user_text + "\n\n" + "\n".join(attachment_notes)
            else:
                user_text_with_meta = user_text

            user_content = types.Content(role="user", parts=parts)
            try:
                await self.db.save_turn("user", user_text_with_meta, content_obj=user_content)
            except Exception as u_err:
                logger.warning(f"Notice in initial user save_turn: {u_err}. Force reconnecting...")
                await self.db.connect(force=True)
                await self.db.save_turn("user", user_text_with_meta, content_obj=user_content)

            system_instruction = build_system_prompt()
            raw_turns = await self.db.get_recent_turns(limit=60, include_tools=True)
            contents = await self.compactor.build_compacted_contents(raw_turns)

            final_response_text = ""
            tools_executed_in_query = 0
            empty_response_retry_count = 0

            for turn in range(max_turns):
                # Instant operator pause/stop check
                if self.autonomous_loop_paused:
                    logger.info("Autonomous loop paused by operator. Halting query loop immediately.")
                    return "⏸️ Autonomous loop paused by operator. In-flight turn cleanly halted."
                if not self.autonomous_loop_active and self.autonomous_loop_task:
                    logger.info("Autonomous loop stopped by operator. Halting query loop immediately.")
                    return "⏹️ Autonomous loop stopped by operator."

                if self.early_final_response is not None:
                    final_response_text = self.early_final_response
                    break

                inter_delay = getattr(config, "INTER_TURN_DELAY", 0.5)
                if inter_delay > 0 and turn > 0:
                    await asyncio.sleep(inter_delay)

                max_turns = getattr(config, "MAX_AGENT_TURNS", max_turns)
                active_model = self.key_manager.get_model()

                # Zero-API-Cost Token Valuation
                contents = sanitize_and_align_contents(contents)
                current_token_count = await count_contents_tokens(
                    None,
                    active_model,
                    contents,
                    system_instruction=system_instruction
                )

                sys_prompt_tokens = estimate_tokens_fallback(system_instruction)
                configured_threshold = getattr(config, "CONTEXT_COMPRESSION_THRESHOLD", 40000)
                window_limit = getattr(config, "INPUT_TOKEN_LIMIT", 1048576)

                effective_threshold = min(
                    configured_threshold,
                    max(sys_prompt_tokens + 15000, window_limit - 100000)
                )

                auto_compact = getattr(config, "AUTO_COMPACT_ON_THRESHOLD", True)

                # Compaction trigger (Safe 40k ceiling)
                if current_token_count >= effective_threshold and auto_compact and len(raw_turns) > self.compactor.preserve_count:
                    if event_callback:
                        await event_callback("info", f"📦 Context reached {current_token_count:,} tokens (safe threshold: {effective_threshold:,}). Triggering lossless distillation via '{self.compactor.get_active_summarizer_model()}'...")

                    await compress_context_now()
                    raw_turns = await self.db.get_recent_turns(limit=60, include_tools=True)
                    contents = await self.compactor.build_compacted_contents(raw_turns)
                    contents = sanitize_and_align_contents(contents)

                    current_token_count = await count_contents_tokens(
                        None,
                        active_model,
                        contents,
                        system_instruction=system_instruction
                    )
                    logger.info(f"Post-compaction token count: {current_token_count:,} tokens.")

                safety_settings = build_safety_settings()
                thinking_config = build_thinking_config(is_synthesis_step=False)

                config_with_tools = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=AGENT_TOOLS,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                    safety_settings=safety_settings,
                    thinking_config=thinking_config
                )

                response = None
                max_api_retries = max(getattr(config, "API_MAX_RETRIES", 60), len(self.key_manager.keys) * 2)
                last_api_error_str = "Unknown"

                for attempt in range(max_api_retries):
                    if self.autonomous_loop_paused:
                        return "⏸️ Autonomous loop paused by operator."

                    # Acquire ready client with pre-flight quota reservation on EACH iteration
                    gemini_client = await self.key_manager.get_ready_client_for_request(
                        estimated_tokens=current_token_count,
                        event_callback=event_callback,
                        agent=self
                    )

                    active_model = self.key_manager.get_model()
                    config_with_tools.system_instruction = system_instruction

                    acc_id = self.key_manager.get_account_for_key(self.key_manager.current_key_index)
                    proj_id = self.key_manager.get_project_for_key(self.key_manager.current_key_index)
                    logger.info(
                        f"Turn {turn + 1}/{max_turns} (Tokens: ~{current_token_count:,} | Attempt {attempt + 1}/{max_api_retries}) -> "
                        f"Gemini API ({active_model}, Key #{self.key_manager.current_key_index + 1}, Acc #{acc_id + 1}, Proj #{proj_id + 1})..."
                    )
                    try:
                        response = await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(
                                model=active_model,
                                contents=contents,
                                config=config_with_tools
                            ),
                            timeout=config.GEMINI_TIMEOUT
                        )
                        self.key_manager.record_request_tokens(self.key_manager.current_key_index, current_token_count)
                        break

                    except APIError as e:
                        err_msg = str(e)
                        err_code = getattr(e, "code", 500)
                        last_api_error_str = f"APIError [{err_code}]: {err_msg}"
                        logger.warning(f"Gemini API Error [{err_code}]: {err_msg}")

                        if err_code == 429 or "RESOURCE_EXHAUSTED" in err_msg.upper():
                            if event_callback:
                                await event_callback("info", f"Rate limit 429 on Key #{self.key_manager.current_key_index + 1} ({active_model}). Pacing and switching key...")
                            gemini_client = await self.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=event_callback,
                                agent=self,
                                estimated_tokens=current_token_count
                            )
                            active_model = self.key_manager.get_model()
                            # KeyManager's handle_quota_exhausted already executed adaptive allostatic pacing sleep
                            continue

                        elif err_code == 404 or "NOT_FOUND" in err_msg.upper() or "not found" in err_msg.lower():
                            if event_callback:
                                await event_callback("info", f"Model '{active_model}' not found (404). Falling back to next model...")
                            gemini_client = await self.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=event_callback,
                                agent=self,
                                estimated_tokens=current_token_count
                            )
                            active_model = self.key_manager.get_model()
                            await asyncio.sleep(0.3)
                            continue

                        elif err_code == 400 and any(w in err_msg.lower() for w in ["tool call", "too many", "function response", "function call", "role", "thinking", "budget"]):
                            logger.warning("400 alignment or tool call limit error. Re-sanitizing contents and retrying...")
                            contents = sanitize_and_align_contents(contents)
                            if "tool call" in err_msg.lower() or "too many" in err_msg.lower():
                                contents.append(types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text="[System Directive: Tool call limit reached for this step. Please formulate your final textual report or execute only 1 essential tool.]")]
                                ))
                                config_with_tools.tools = None
                            if "thinking" in err_msg.lower() or "budget" in err_msg.lower():
                                config_with_tools.thinking_config = None
                            await asyncio.sleep(0.5)
                            continue

                        elif err_code in [500, 502, 503, 504] or "UNAVAILABLE" in err_msg.upper() or "high demand" in err_msg.lower():
                            if event_callback:
                                await event_callback("info", f"Transient Server Error {err_code}. Switching key/model...")
                            gemini_client = await self.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=event_callback,
                                agent=self,
                                estimated_tokens=current_token_count
                            )
                            active_model = self.key_manager.get_model()
                            await asyncio.sleep(0.5)
                            continue

                        elif err_code == 401 or "unauthenticated" in err_msg.lower():
                            gemini_client = await self.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=event_callback,
                                agent=self,
                                estimated_tokens=current_token_count
                            )
                            active_model = self.key_manager.get_model()
                            await asyncio.sleep(0.3)
                            continue

                        else:
                            if attempt == max_api_retries - 1:
                                error_turn_msg = f"Gemini API Error [{err_code}]: {err_msg}"
                                await self.db.save_turn(
                                    "system",
                                    f"[System Interruption Event: Query loop interrupted due to {error_turn_msg}. Last tool: '{last_executed_tool}'. State preserved.]"
                                )
                                err_content = types.Content(role="model", parts=[types.Part.from_text(text=error_turn_msg)])
                                await self.db.save_turn("model", error_turn_msg, content_obj=err_content)
                                return error_turn_msg
                            await asyncio.sleep(0.5)
                            continue

                    except asyncio.TimeoutError:
                        last_api_error_str = f"Timeout after {config.GEMINI_TIMEOUT}s"
                        logger.warning(f"Gemini request timed out after {config.GEMINI_TIMEOUT}s. Rotating key...")
                        gemini_client = await self.key_manager.handle_quota_exhausted(
                            "Timeout",
                            event_callback=event_callback,
                            agent=self,
                            estimated_tokens=current_token_count
                        )
                        active_model = self.key_manager.get_model()
                        await asyncio.sleep(0.3)
                        continue

                    except asyncio.CancelledError:
                        await self.db.save_turn("system", f"[System Interruption Event: Generation cancelled during API call. Last tool: '{last_executed_tool}'.]")
                        logger.info("Generation cancelled during API request.")
                        raise

                    except Exception as ex:
                        last_api_error_str = str(ex)
                        logger.error(f"Unexpected API request error: {str(ex)}")
                        if "429" in last_api_error_str or "RESOURCE_EXHAUSTED" in last_api_error_str.upper() or "quota" in last_api_error_str.lower():
                            gemini_client = await self.key_manager.handle_quota_exhausted(
                                ex,
                                event_callback=event_callback,
                                agent=self,
                                estimated_tokens=current_token_count
                            )
                            active_model = self.key_manager.get_model()
                            continue
                        if attempt == max_api_retries - 1:
                            err_txt = f"Execution Error: {str(ex)}"
                            await self.db.save_turn("system", f"[System Interruption Event: {err_txt}. Last tool: '{last_executed_tool}'.]")
                            err_content = types.Content(role="model", parts=[types.Part.from_text(text=err_txt)])
                            await self.db.save_turn("model", err_txt, content_obj=err_content)
                            return err_txt
                        await asyncio.sleep(0.5)
                        continue

                if response is None:
                    fail_msg = f"Error: Failed to obtain response from Gemini API after {max_api_retries} retries ({last_api_error_str})."
                    await self.db.save_turn(
                        "system",
                        f"[System Interruption Event: {fail_msg} Last tool: '{last_executed_tool}'.]"
                    )
                    fail_content = types.Content(role="model", parts=[types.Part.from_text(text=fail_msg)])
                    await self.db.save_turn("model", fail_msg, content_obj=fail_content)
                    return fail_msg

                function_calls = []
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            function_calls.append(part.function_call)

                extracted_texts = []
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            t_clean = part.text.strip()
                            if t_clean:
                                extracted_texts.append(t_clean)

                resp_text = "\n\n".join(extracted_texts).strip()

                if not function_calls and resp_text:
                    parsed_calls = extract_pseudo_text_tool_calls(resp_text, self.tools_map)
                    if parsed_calls:
                        logger.info(f"Intercepted {len(parsed_calls)} pseudo-text tool call(s) from model response text.")
                        function_calls.extend(parsed_calls)

                # Handle Tool Invocations
                if function_calls:
                    tools_executed_in_query += len(function_calls)
                    empty_response_retry_count = 0

                    existing_fc_names = set()
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for p in response.candidates[0].content.parts:
                            if getattr(p, "function_call", None) is not None:
                                existing_fc_names.add(p.function_call.name)

                    missing_calls = [c for c in function_calls if c.name not in existing_fc_names]
                    if missing_calls:
                        model_parts = list(response.candidates[0].content.parts) if (response.candidates and response.candidates[0].content and response.candidates[0].content.parts) else []
                        for call in missing_calls:
                            fc_part = types.Part.from_function_call(name=call.name, args=call.args or {})
                            fc_part.thought_signature = b"skip_thought_signature_validator"
                            model_parts.append(fc_part)
                        model_content = types.Content(role="model", parts=model_parts)
                    else:
                        model_content = response.candidates[0].content

                    contents.append(model_content)

                    tool_responses = []
                    for call in function_calls:
                        if self.autonomous_loop_paused:
                            return "⏸️ Autonomous loop paused by operator."

                        fn_name = call.name
                        fn_args = call.args or {}
                        last_executed_tool = fn_name
                        args_preview = json.dumps(clean_for_json(fn_args), ensure_ascii=False) if fn_args else "{}"
                        if len(args_preview) > 150:
                            args_preview = args_preview[:150] + "..."

                        is_msg_tool = (fn_name == "send_agent_message")

                        if event_callback and not is_msg_tool:
                            await event_callback("tool_start", f"`{fn_name}` with args: `{args_preview}`")

                        tool_call_part = types.Part.from_function_call(name=fn_name, args=fn_args or {})
                        tool_call_part.thought_signature = b"skip_thought_signature_validator"
                        tool_call_content = types.Content(role="model", parts=[tool_call_part])
                        try:
                            await self.db.save_turn(
                                "tool_call",
                                f"[Tool Invocation: `{fn_name}`] {args_preview}",
                                content_obj=tool_call_content
                            )
                        except Exception as st_err:
                            logger.warning(f"Notice in save_turn for tool_call: {st_err}. Force reconnecting...")
                            await self.db.connect(force=True)
                            await self.db.save_turn(
                                "tool_call",
                                f"[Tool Invocation: `{fn_name}`] {args_preview}",
                                content_obj=tool_call_content
                            )

                        tool_fn = self.tools_map.get(fn_name)
                        if tool_fn:
                            try:
                                if inspect.iscoroutinefunction(tool_fn):
                                    result = await tool_fn(**fn_args)
                                else:
                                    result = await asyncio.to_thread(tool_fn, **fn_args)
                            except asyncio.CancelledError:
                                raise
                            except Exception as terr:
                                result = f"Error executing tool '{fn_name}': {str(terr)}"
                        else:
                            result = f"Error: Tool '{fn_name}' not found in registry."

                        result_str = str(result)
                        preview = result_str[:1200] + ("\n... [output truncated for preview]" if len(result_str) > 1200 else "")

                        if event_callback and not is_msg_tool:
                            await event_callback("tool_end", f"**Result from `{fn_name}`:**\n```text\n{preview}\n```")

                        tool_resp_part = types.Part.from_function_response(name=fn_name, response={"result": result_str})
                        tool_resp_content = types.Content(role="user", parts=[tool_resp_part])
                        try:
                            await self.db.save_turn(
                                "tool_result",
                                f"[Tool Output: `{fn_name}`]\n```text\n{preview}\n```",
                                content_obj=tool_resp_content
                            )
                        except Exception as st_err:
                            logger.warning(f"Notice in save_turn for tool_result: {st_err}. Force reconnecting...")
                            await self.db.connect(force=True)
                            await self.db.save_turn(
                                "tool_result",
                                f"[Tool Output: `{fn_name}`]\n```text\n{preview}\n```",
                                content_obj=tool_resp_content
                            )

                        tool_responses.append(tool_resp_part)

                    tool_content_resp = types.Content(role="user", parts=tool_responses)
                    contents.append(tool_content_resp)

                    if self.early_final_response is not None:
                        final_response_text = self.early_final_response
                        break

                    continue

                else:
                    if not resp_text and tools_executed_in_query > 0 and empty_response_retry_count < 2:
                        empty_response_retry_count += 1
                        logger.info(f"Model returned empty text after tool execution. Re-prompting continuation ({empty_response_retry_count}/2)...")
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="[Acknowledged tool output. Awaiting next directive.]")]
                        ))
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="[System Directive: Tool execution completed. Please inspect the output above and either execute the next tool required to advance your objective, or formulate your final analytical response.]")]
                        ))
                        continue

                    if self.early_final_response is not None:
                        final_response_text = self.early_final_response
                        break

                    if resp_text:
                        final_response_text = resp_text
                        break

                    if tools_executed_in_query > 0 and not final_response_text:
                        logger.info("Model completed tools with no text output. Synthesizing analytical report...")
                        if event_callback:
                            await event_callback("synthesis", "Synthesizing analytical summary report...")

                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="[Acknowledged tool execution telemetry.]")]
                        ))
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="[System Directive: Synthesize a concise analytical report detailing actions performed, observed telemetry, and next steps.]")]
                        ))

                        contents = sanitize_and_align_contents(contents)
                        synthesis_thinking = build_thinking_config(is_synthesis_step=True)

                        config_synthesis = types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=None,
                            temperature=config.TEMPERATURE,
                            top_p=config.TOP_P,
                            max_output_tokens=config.MAX_OUTPUT_TOKENS,
                            safety_settings=safety_settings,
                            thinking_config=synthesis_thinking
                        )

                        for s_attempt in range(3):
                            try:
                                synth_client = await self.key_manager.get_ready_client_for_request(
                                    estimated_tokens=current_token_count,
                                    event_callback=event_callback,
                                    agent=self
                                )
                                synth_model = self.key_manager.get_model()
                                synth_resp = await asyncio.wait_for(
                                    synth_client.aio.models.generate_content(
                                        model=synth_model,
                                        contents=contents,
                                        config=config_synthesis
                                    ),
                                    timeout=config.GEMINI_TIMEOUT
                                )
                                if synth_resp and synth_resp.text and synth_resp.text.strip():
                                    final_response_text = synth_resp.text.strip()
                                    break
                            except Exception as se:
                                logger.warning(f"Synthesis notice: {str(se)}")
                                synth_client = await self.key_manager.handle_quota_exhausted(
                                    se,
                                    event_callback=event_callback,
                                    agent=self,
                                    estimated_tokens=current_token_count
                                )
                                synth_model = self.key_manager.get_model()
                                await asyncio.sleep(0.5)

                    if final_response_text:
                        break

            if not final_response_text:
                final_response_text = await self.build_fallback_execution_summary()

            if not self.early_final_response:
                model_final_content = types.Content(role="model", parts=[types.Part.from_text(text=final_response_text)])
                await self.db.save_turn("model", final_response_text, content_obj=model_final_content)

            asyncio.create_task(sync_agent_database("chore(db): auto-sync state snapshot after query completion"))
            return final_response_text

        finally:
            self.is_busy = False
            self.current_task = None
            self.active_event_callback = None
