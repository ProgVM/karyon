# KARYON_PATCH_V33_2_MULTI_AGENT_IMPORT_DEFENSE_APPLIED
# KARYON_PATCH_V33_1_MULTI_AGENT_PY_APPLIED
# KARYON_PATCH_V33_MULTI_AGENT_PY_APPLIED
# karyon_agent_runtime/multi_agent.py
"""
===============================================================================
KARYON CORE HIERARCHICAL MULTI-AGENT CORTICAL ORCHESTRATION ENGINE (v32.6 MASTER)
Implements Laminar Cortical Columnar Specialization (Principle 2 Biophysical Realism),
Full Subagent Tool Parity (All 180 Tools Available by Default without Artificial Constraints),
Rich Context Injection (Empirical Ledger, KEP Guidelines & Recent Synaptic Messages),
Synchronous Bidirectional Message Passing with Inline Replies, Parallel Swarm Dispatch,
and Sliding Window Token Footprint Recording.
===============================================================================
"""

import re
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
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
from karyon_agent_runtime.db_manager import clean_for_json
try:
    from karyon_agent_runtime.agent_core import build_safety_settings, build_thinking_config
except ImportError:
    build_safety_settings = None
    build_thinking_config = None
from karyon_agent_runtime.context_compactor import (
    count_contents_tokens,
    sanitize_and_align_contents,
    truncate_tool_response_text
)

logger = logging.getLogger("ProxyAgent.MultiAgent")


class AgentRolePresets:
    """
    Standardized Cortical Laminar Presets grounded in Biophysical Realism (KEP Principle 2):
    Every agent is FULLY EMPOWERED with the complete production tool suite (P.S.2 parity).
    Role determines their biophysical reasoning posture and laminar layer specialization.
    """

    RESEARCHER_PROMPT = """You are the Dedicated Literature & Biophysical Research Agent for Karyon-CoRE (Layer IV - Sensory Afferents).
Your mission is to explore theoretical literature, extract exact mathematical equations, search arXiv and the web,
and provide mathematically verified empirical foundations for Karyon's living AGI architecture.

STRICT RESEARCH PROTOCOLS:
1. FOCUS ON BIOPHYSICAL REALISM: Ground all research in Karl Friston's Active Inference (variational free energy F),
   György Buzsáki's neural oscillations (theta-gamma PAC), W. Ross Ashby's somatic homeostasis, and Modern Hopfield energy.
2. PRESERVE EXACT FORMULAS: Never hand-wave or approximate mathematics. Provide exact LaTeX formulas and parameters.
3. EMPIRICAL CITATIONS: Cite arXiv paper IDs, author names, and publication dates for every claim.
4. FULL CAPABILITY: You have access to the complete tool suite (including file read/write, bash, and code execution).
   Act autonomously to acquire and verify knowledge, reporting synthesized findings back to your caller.
"""

    CODER_PROMPT = """You are the Lead Implementation & High-Performance Engineering Agent for Karyon-CoRE (Layer V - Motor Efference).
Your mission is to author robust, high-throughput, biologically realistic code (C++20, CUDA, PyTorch)
strictly following KEP v9.0 Master and Principle 1 (C++20 engine, thin Python orchestration client).

STRICT CODING PROTOCOLS:
1. KEP RULE #3 (ZERO PLACEHOLDERS): All code must be 100% complete, uncompressed, production-grade, and free of placeholders (`...` or `// TODO`).
2. PRINCIPLE 9 / AXIS A (HARDWARE SAFETY):
   - Zero-sync memory reads: NEVER call `.item()` in GPU token or step loops.
   - Division by zero safety: Always guard loss masks and normalizers with `.clamp_min(1.0)` or epsilon.
   - Modern Hopfield: Use dot-product energy basins to prevent un-epsiloned cdist NaN divergence.
   - FP16 stability: All FP16 module outputs must end with LayerNorm to prevent overflow (>65,504).
3. ATOMIC FILE OPS: Use `write_file` and `read_file` to write code directly to the repository without markdown chat dumps.
4. VERIFY EXECUTION: Verify compilation via `build_and_verify_cpp_core` or `execute_python_code` before reporting success.
"""

    REFACTORER_PROMPT = """You are the Principle 9 Continuous Spontaneous Dual-Refactoring Agent for Karyon-CoRE (Layer II/III - Homeostasis).
Your mission is to continuously audit, detect, and eradicate technical bugs and non-biological crutches.

STRICT REFACTORING PROTOCOLS (TWO-AXIS AUDIT):
1. AXIS A (HARDWARE SAFETY & SPEED REGRESSIONS):
   - Eradicate PCIe synchronization stalls (`.item()` in loops).
   - Detect and fix division-by-zero risks, missing LayerNorms, and tensor shape broadcast mismatches.
   - Ensure continuous packed streaming (S=2048, 0% padding) and zero memory leaks.
2. AXIS B (BIOPHYSICAL REALISM - PRINCIPLE 2 NON-NEGOTIABLE):
   - Eliminate discrete transformer shortcuts and artificial crutches (No discrete Mixture-of-Experts / MoE).
   - Enforce continuous neural oscillations (theta-gamma PAC), unit-sphere Hopfield energy, and Ashby somatic homeostasis.
3. FULL ACTIONABILITY: Audit and directly patch code using `write_file` and `batch_replace_text`.
"""

    CRITIC_PROMPT = """You are the Adversarial Empirical Auditor & KEP Rule #2 Critic for Karyon-CoRE (Apical Dendrites).
Your mission is to ruthlessly critique hypotheses, benchmark telemetry, and empirical metrics to prevent
confirmation bias, premature conclusions, and regressions.

STRICT CRITIQUE PROTOCOLS:
1. KEP RULE #2 DECISION ENGINE:
   - Positive (🟢 POSITIVE): Mandates statistically significant breakthrough in target metric (e.g. Loss Delta >= 0.08,
     throughput >= 110%, or clear Free Energy drop) with ZERO degradation in secondary invariants.
   - Neutral (⚪ NEUTRAL): Performance within noise margin (+-0.05). Demands further biophysical refinement.
   - Rejected (🔴 REJECTED): Divergence, NaN overflow, throughput drop, or pseudo-morphemic drift.
2. AUDIT DIAGNOSTIC SPEECH: Audit live text generation samples for phonotactic coherence and absence of drift.
3. UNAMBIGUOUS VERDICT: Run benchmark evaluation via `run_kep_scientific_pipeline` and deliver a decisive, data-driven verdict.
"""

    PRESET_MAP = {
        "researcher": {
            "name": "Biophysical Literature & Evidence Researcher (Layer IV)",
            "prompt": RESEARCHER_PROMPT,
            "tools": ["*"]
        },
        "coder": {
            "name": "C++20 & PyTorch Architecture Coder (Layer V)",
            "prompt": CODER_PROMPT,
            "tools": ["*"]
        },
        "refactorer": {
            "name": "Principle 9 Dual-Refactoring Auditor (Layer II/III)",
            "prompt": REFACTORER_PROMPT,
            "tools": ["*"]
        },
        "critic": {
            "name": "Empirical Telemetry Validator & Critic (Apical)",
            "prompt": CRITIC_PROMPT,
            "tools": ["*"]
        }
    }


from karyon_agent_runtime.synaptic_bus import SynapticMeshEventBus

class MultiAgentManager:

    async def _notify_ui(self, event_type: str, payload: str):
        cb = getattr(self.agent_core, "active_event_callback", None)
        if cb and callable(cb):
            try:
                await cb(event_type, payload)
            except Exception:
                pass

    def __init__(self, agent_core, db_manager, bus: Optional[SynapticMeshEventBus] = None):
        self.agent_core = agent_core
        self.db = db_manager
        self._active_task_jobs: Dict[str, asyncio.Task] = {}
        self.bus = bus or getattr(agent_core, "bus", None) or SynapticMeshEventBus(db_manager=db_manager)

    async def initialize(self):
        root = await self.db.get_sub_agent("root")
        if not root:
            await self.db.create_sub_agent(
                agent_id="root",
                name="Karyon Root Executive Cyberneticist",
                role="orchestrator",
                parent_id=None,
                system_prompt="Executive cortical orchestrator responsible for overall scientific direction, hypothesis approval, and task delegation.",
                model=self.agent_core.key_manager.get_model(),
                thinking_level="HIGH",
                thinking_budget=getattr(config, "THINKING_BUDGET", 24576),
                allowed_tools=["*"],
                can_communicate_with_peers=True,
                allowed_peers=["*"]
            )
            logger.info("MultiAgentManager: Initialized sovereign 'root' agent record in SQLite.")

        for r_name, r_info in AgentRolePresets.PRESET_MAP.items():
            agent_uid = f"agent_{r_name}"
            existing = await self.db.get_sub_agent(agent_uid)
            if not existing:
                await self.db.create_sub_agent(
                    agent_id=agent_uid,
                    name=r_info["name"],
                    role=r_name,
                    parent_id="root",
                    system_prompt=r_info["prompt"],
                    model=self.agent_core.key_manager.get_model(),
                    allowed_tools=["*"],
                    can_communicate_with_peers=True,
                    allowed_peers=["*"]
                )
                logger.info(f"MultiAgentManager: Pre-registered cortical sub-agent '{agent_uid}'.")
            elif existing.get("allowed_tools") != ["*"]:
                await self.db.update_sub_agent(agent_uid, allowed_tools_json=["*"])

    def _resolve_tools_for_agent(self, agent_data: Dict[str, Any]) -> List[Any]:
        all_tools = getattr(self.agent_core, "tools_map", {})
        blocked = set(agent_data.get("blocked_tools") or [])
        # Full 185 production tool parity across all cortical columns without artificial constraints
        return [fn for name, fn in all_tools.items() if name not in blocked]

    async def create_agent(
        self,
        name: str,
        role: str = "custom",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        can_communicate_with_peers: bool = True,
        allowed_peers: Optional[List[str]] = None,
        max_turns: int = 40,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = "root",
        metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        clean_role = role.lower().strip()
        preset = AgentRolePresets.PRESET_MAP.get(clean_role)

        target_name = name.strip() or (preset["name"] if preset else f"Agent_{clean_role.capitalize()}")
        target_prompt = (system_prompt or "").strip()
        if not target_prompt and preset:
            target_prompt = preset["prompt"]
        if not target_prompt:
            target_prompt = f"You are a specialized {clean_role} cortical agent assisting Bazilevs and Karyon-CoRE."

        target_tools = allowed_tools if allowed_tools is not None else ["*"]
        target_model = (model or "").strip() or self.agent_core.key_manager.get_model()

        if not agent_id:
            ts = int(time.time())
            tag = clean_role if clean_role != "custom" else "sub"
            target_id = f"agent_{tag}_{ts % 100000:05d}"
        else:
            target_id = re.sub(r'[^a-zA-Z0-9_]', '_', agent_id.strip().lower())

        target_parent = parent_id.strip() if parent_id else "root"
        agent_record = await self.db.create_sub_agent(
            agent_id=target_id,
            name=target_name,
            role=clean_role,
            parent_id=target_parent,
            system_prompt=target_prompt,
            model=target_model,
            thinking_level="HIGH",
            thinking_budget=getattr(config, "THINKING_BUDGET", 24576),
            allowed_tools=target_tools,
            blocked_tools=blocked_tools or [],
            can_communicate_with_peers=can_communicate_with_peers,
            allowed_peers=allowed_peers or ["*"],
            max_turns=max_turns,
            metadata=metadata or {}
        )

        logger.info(f"MultiAgentManager: Created subagent '{target_id}' ({target_name}, role: {clean_role}, parent: {target_parent}).")
        return agent_record

    async def update_agent(self, agent_id: str, acting_agent_id: str = "root", **kwargs) -> Dict[str, Any]:
        target_id = agent_id.strip()
        acting_id = acting_agent_id.strip()

        can_manage, reason = await self.db.can_manage_agent(acting_id, target_id)
        if not can_manage:
            raise PermissionError(f"Unauthorized update: {reason}")

        success = await self.db.update_sub_agent(target_id, **kwargs)
        if not success:
            raise ValueError(f"Failed to update agent '{target_id}'.")

        return await self.db.get_sub_agent(target_id)

    async def delete_agent(self, agent_id: str, acting_agent_id: str = "root") -> bool:
        target_id = agent_id.strip()
        acting_id = acting_agent_id.strip()

        can_manage, reason = await self.db.can_manage_agent(acting_id, target_id)
        if not can_manage:
            raise PermissionError(f"Unauthorized deletion: {reason}")

        for t_id, task_obj in list(self._active_task_jobs.items()):
            if t_id.startswith(f"{target_id}_") and not task_obj.done():
                task_obj.cancel()

        return await self.db.delete_sub_agent(target_id)

    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        msg_type: str = "peer_discussion",
        task_id: Optional[str] = None,
        subject: str = "",
        wait_for_reply: bool = False,
        reply_timeout: float = 180.0
    ) -> Dict[str, Any]:
        from_id = from_agent_id.strip()
        to_id = to_agent_id.strip()

        from_agent = await self.db.get_sub_agent(from_id)
        if not from_agent and from_id != "root":
            raise ValueError(f"Sender agent '{from_id}' not found.")

        to_agent = await self.db.get_sub_agent(to_id)
        if not to_agent:
            raise ValueError(f"Recipient agent '{to_id}' not found.")

        authorized = False
        if from_id == "root" or to_id == "root":
            authorized = True
        elif from_agent and from_agent.get("parent_id") == to_id:
            authorized = True
        elif to_agent and to_agent.get("parent_id") == from_id:
            authorized = True
        elif from_agent:
            can_peer = from_agent.get("can_communicate_with_peers", True)
            allowed_peers = from_agent.get("allowed_peers") or ["*"]
            if can_peer and ("*" in allowed_peers or to_id in allowed_peers):
                authorized = True

        if not authorized:
            raise PermissionError(f"Communication Policy Violation: Agent '{from_id}' is not authorized to contact peer '{to_id}'.")

        msg_id = await self.db.record_agent_message(
            from_agent_id=from_id,
            to_agent_id=to_id,
            msg_type=msg_type,
            body=message.strip(),
            subject=subject.strip(),
            task_id=task_id
        )

        # Transparently stream inter-agent synaptic communication to UI
        msg_preview = message[:500] + ("..." if len(message) > 500 else "")
        await self._notify_ui(
            "info",
            f"📡 **[Synaptic Message]** `{from_id}` ➔ `{to_id}` ({msg_type} | {subject or 'Direct'}):\n\n> {msg_preview}"
        )

        response_payload = {
            "message_id": msg_id,
            "from_agent_id": from_id,
            "to_agent_id": to_id,
            "msg_type": msg_type,
            "status": "DELIVERED"
        }

        if wait_for_reply:
            handoff_prompt = (
                f"=== INCOMING SYNAPTIC MESSAGE FROM AGENT `{from_id}` ===\n"
                f"- Message Type: `{msg_type}` | Subject: {subject or 'Direct Inquiry'}\n\n"
                f"{message}\n\n"
                f"TASK: Inspect the inquiry, perform any necessary tool operations, and formulate your direct analytical reply back to `{from_id}`."
            )
            reply_text = await self.dispatch_task(
                assigned_agent_id=to_id,
                task_prompt=handoff_prompt,
                creator_agent_id=from_id,
                wait_for_result=True,
                timeout_seconds=reply_timeout
            )
            response_payload["reply"] = reply_text
            response_payload["status"] = "REPLIED"

            reply_preview = str(reply_text)[:500] + ("..." if len(str(reply_text)) > 500 else "")
            await self._notify_ui(
                "info",
                f"💬 **[Synaptic Reply]** `{to_id}` ➔ `{from_id}`:\n\n{reply_preview}"
            )

        return response_payload

    async def dispatch_task(
        self,
        assigned_agent_id: str,
        task_prompt: str,
        creator_agent_id: str = "root",
        wait_for_result: bool = True,
        timeout_seconds: float = 300.0
    ) -> str:
        assigned_id = assigned_agent_id.strip()
        creator_id = creator_agent_id.strip()

        agent_record = await self.db.get_sub_agent(assigned_id)
        if not agent_record:
            return f"Error: Target subagent '{assigned_id}' not found in registry."

        if not agent_record.get("is_active", 1):
            return f"Error: Target subagent '{assigned_id}' is currently marked INACTIVE."

        task_id = f"task_{assigned_id}_{int(time.time())}_{int(time.perf_counter() * 1000) % 1000:03d}"
        await self.db.create_agent_task(
            task_id=task_id,
            creator_agent_id=creator_id,
            assigned_agent_id=assigned_id,
            task_prompt=task_prompt
        )

        await self.db.record_agent_message(
            from_agent_id=creator_id,
            to_agent_id=assigned_id,
            msg_type="task_assignment",
            subject=f"New Task: {task_id}",
            body=task_prompt,
            task_id=task_id
        )

        if hasattr(self, "bus") and self.bus:
            await self.bus.emit(
                topic="TASK_ASSIGNED",
                sender_agent_id=creator_id,
                payload={"assigned_to": assigned_id, "prompt_snippet": task_prompt[:200]},
                summary_text=f"Dispatched task `{task_id}` to `{assigned_id}`",
                task_id=task_id
            )

        worker_coro = self._run_subagent_task_worker(
            task_id=task_id,
            agent_record=agent_record,
            task_prompt=task_prompt,
            creator_id=creator_id,
            timeout_seconds=timeout_seconds
        )

        if wait_for_result:
            try:
                result_text = await asyncio.wait_for(worker_coro, timeout=timeout_seconds)
                return (
                    f"=== Subagent '{agent_record['name']}' (`{assigned_id}`) Completed Task `{task_id}` ===\n"
                    f"- Role: `{agent_record['role']}` | Creator: `{creator_id}`\n\n"
                    f"{result_text}"
                )
            except asyncio.TimeoutError:
                await self.db.update_agent_task_status(
                    task_id=task_id,
                    status="TIMED_OUT",
                    error=f"Task execution exceeded limit of {timeout_seconds}s"
                )
                return f"Error: Subagent '{assigned_id}' timed out after {timeout_seconds}s executing task `{task_id}`."
        else:
            task_obj = asyncio.create_task(worker_coro, name=f"SubAgentTask_{task_id}")
            self._active_task_jobs[task_id] = task_obj
            return (
                f"=== Subagent Task Dispatched in Background ===\n"
                f"- Task ID       : `{task_id}`\n"
                f"- Assigned Agent: `{agent_record['name']}` (`{assigned_id}`)\n"
                f"- Role          : `{agent_record['role']}`\n"
                f"- Status        : RUNNING ⏳\n"
                f"*(Poll via `get_agent_task_status(task_id='{task_id}')`)*"
            )

    async def dispatch_parallel_tasks(
        self,
        tasks_list: List[Dict[str, Any]],
        creator_agent_id: str = "root",
        timeout_seconds: float = 300.0
    ) -> List[Dict[str, Any]]:
        """Executes multiple subagent tasks concurrently in parallel using asyncio.gather."""
        coroutines = []
        task_meta = []
        for item in tasks_list:
            assigned_id = item.get("agent_id") or item.get("assigned_agent_id")
            prompt = item.get("task_prompt") or item.get("prompt") or ""
            t_timeout = float(item.get("timeout_seconds") or timeout_seconds)
            task_meta.append({"agent_id": assigned_id, "prompt": prompt})
            coroutines.append(
                self.dispatch_task(
                    assigned_agent_id=assigned_id,
                    task_prompt=prompt,
                    creator_agent_id=creator_agent_id,
                    wait_for_result=True,
                    timeout_seconds=t_timeout
                )
            )

        results = await asyncio.gather(*coroutines, return_exceptions=True)
        combined = []
        for meta, res in zip(task_meta, results):
            if isinstance(res, Exception):
                combined.append({
                    "agent_id": meta["agent_id"],
                    "status": "FAILED",
                    "result": f"Exception: {str(res)}"
                })
            else:
                combined.append({
                    "agent_id": meta["agent_id"],
                    "status": "COMPLETED",
                    "result": str(res)
                })
        return combined

    async def _build_subagent_rich_context(self, agent_record: Dict[str, Any], task_prompt: str) -> List[types.Content]:
        contents = []

        # 1. Active User Principles & Directives from Bazilevs
        user_directive = await self.db.get_memory("active_user_directive")
        saved_agenda = await self.db.get_setting("autonomous_agenda")
        if user_directive or saved_agenda:
            directive_parts = []
            if user_directive:
                directive_parts.append(f"Permanent User Principles from Bazilevs:\n{user_directive}")
            if saved_agenda:
                directive_parts.append(f"Active Research Agenda:\n{saved_agenda}")
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="[System Directives from Bazilevs]\n" + "\n\n".join(directive_parts))]
            ))

        # 2. Immutable Empirical Scientific Ledger (Recent experiments)
        recent_exps = await self.db.get_all_experiments(limit=8)
        if recent_exps:
            ledger_rows = [
                "=== ACTIVE SCIENTIFIC EMPIRICAL LEDGER (RECENT EXPERIMENTS) ===",
                "| EXP ID | Hypothesis & Delta | Loss | Verdict | Metrics |",
                "|---|---|---|---|---|"
            ]
            for exp in recent_exps:
                met = exp.get("metrics") or {}
                met_str = ", ".join(f"{k}={v}" for k, v in list(met.items())[:3]) if met else "-"
                loss_str = f"{exp.get('final_loss'):.4f}" if exp.get('final_loss') is not None else "N/A"
                ledger_rows.append(
                    f"| {exp['exp_id']} | {exp['hypothesis'][:30]} ({exp['architecture_delta'][:20]}) | {loss_str} | {exp['verdict']} | {met_str} |"
                )
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="[Empirical Research Context]\n" + "\n".join(ledger_rows))]
            ))

        # 3. Lossless Historical State Summary from DB
        cached_summary = await self.db.get_memory("active_context_summary")
        if cached_summary:
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"[Active Architecture & Constraint Summary]\n{cached_summary}")]
            ))

        # 4. Recent Synaptic Network Messages
        agent_id = agent_record["agent_id"]
        recent_msgs = await self.db.get_agent_inbox_messages(agent_id=agent_id, limit=5)
        if recent_msgs:
            msg_rows = [f"[Recent Synaptic Network Messages for `{agent_id}`]"]
            for m in recent_msgs:
                msg_rows.append(f"- From `{m['from_agent_id']}` ({m['msg_type']}): {m['body'][:250]}")
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="\n".join(msg_rows))]
            ))

        # 5. Recent Working Dialogue History from DB
        recent_turns = await self.db.get_recent_turns(limit=6, include_tools=False)
        if recent_turns:
            dialogue_rows = ["[Recent Working Dialogue Turns]"]
            for t in recent_turns:
                r_tag = "Bazilevs" if t.get("role") == "user" else "Karyon"
                dialogue_rows.append(f"- {r_tag}: {t.get('text', '')[:200]}...")
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="\n".join(dialogue_rows))]
            ))

        # 6. Delegated Task Objective
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"[Delegated Task Objective for `{agent_record['name']}`]\n{task_prompt}")]
        ))

        return contents

    async def _run_subagent_task_worker(
        self,
        task_id: str,
        agent_record: Dict[str, Any],
        task_prompt: str,
        creator_id: str,
        timeout_seconds: float
    ) -> str:
        from karyon_agent_runtime.agent_core import extract_pseudo_text_tool_calls

        agent_id = agent_record["agent_id"]
        max_turns = int(agent_record.get("max_turns", 40))
        system_prompt = agent_record.get("system_prompt", "You are an autonomous cortical agent.")

        filtered_tools = self._resolve_tools_for_agent(agent_record)
        t_start = time.time()
        turns_used = 0
        final_answer = ""
        max_tool_chars = getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)

        contents = await self._build_subagent_rich_context(agent_record, task_prompt)
        contents = sanitize_and_align_contents(contents)

        await self.db.save_subagent_turn(
            agent_id=agent_id,
            role="user",
            text=task_prompt,
            content_obj=contents[-1],
            task_id=task_id
        )

        # Stream task start to active UI
        task_preview = str(task_prompt)[:250]
        await self._notify_ui("info", f"🤖 **[{agent_record['name']} (`{agent_id}`)]** Started task `{task_id}`: " + task_preview)
        logger.info(f"Subagent '{agent_id}' commencing task '{task_id}' ({len(filtered_tools)} tools active)...")

        try:
            for turn in range(max_turns):
                turns_used += 1

                if getattr(self.agent_core, "autonomous_loop_paused", False) or (
                    not getattr(self.agent_core, "autonomous_loop_active", True)
                    and getattr(self.agent_core, "autonomous_loop_task", None)
                ):
                    return f"[Task '{task_id}' halted: Autonomous loop paused or stopped by operator.]"

                contents = sanitize_and_align_contents(contents)
                current_tokens = await count_contents_tokens(
                    None,
                    self.agent_core.key_manager.get_model(),
                    contents,
                    system_instruction=system_prompt
                )

                max_api_retries = max(
                    getattr(config, "API_MAX_RETRIES", 60),
                    len(self.agent_core.key_manager.keys) * 2
                )
                response = None
                last_api_error_str = "Unknown"

                inter_delay = getattr(config, "INTER_TURN_DELAY", 0.5)
                if inter_delay > 0 and turn > 0:
                    await asyncio.sleep(inter_delay)

                for attempt in range(max_api_retries):
                    if getattr(self.agent_core, "autonomous_loop_paused", False):
                        return "⏸️ Subagent execution halted: Autonomous loop paused."

                    gemini_client = await self.agent_core.key_manager.get_ready_client_for_request(
                        estimated_tokens=current_tokens,
                        event_callback=self._notify_ui,
                        agent=self.agent_core
                    )

                    active_model = self.agent_core.key_manager.get_model()

                    _safety_fn = build_safety_settings
                    _thinking_fn = build_thinking_config
                    if _safety_fn is None or _thinking_fn is None:
                        from karyon_agent_runtime.agent_core import (
                            build_safety_settings as _safety_fn,
                            build_thinking_config as _thinking_fn
                        )

                    tool_config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=filtered_tools if filtered_tools else None,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        temperature=getattr(config, "TEMPERATURE", 0.4),
                        top_p=getattr(config, "TOP_P", 0.95),
                        max_output_tokens=config.MAX_OUTPUT_TOKENS,
                        safety_settings=_safety_fn(),
                        thinking_config=_thinking_fn(is_synthesis_step=False)
                    )

                    try:
                        response = await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(
                                model=active_model,
                                contents=contents,
                                config=tool_config
                            ),
                            timeout=min(config.GEMINI_TIMEOUT, timeout_seconds)
                        )
                        self.agent_core.key_manager.record_request_tokens(
                            self.agent_core.key_manager.current_key_index,
                            current_tokens
                        )
                        break

                    except APIError as e:
                        err_msg = str(e)
                        err_code = getattr(e, "code", 500)
                        last_api_error_str = f"APIError [{err_code}]: {err_msg}"

                        if err_code == 429 or "RESOURCE_EXHAUSTED" in err_msg.upper():
                            await self._notify_ui("info", f"Rate limit 429 on Key #{self.agent_core.key_manager.current_key_index + 1} ({active_model}). Pacing and switching key...")
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=self._notify_ui,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            continue

                        elif err_code == 400 and any(w in err_msg.lower() for w in ["tool call", "too many", "function response", "function call", "role", "thinking", "budget"]):
                            logger.warning("400 alignment or tool call limit notice in subagent. Re-sanitizing and retrying...")
                            contents = sanitize_and_align_contents(contents)
                            if "tool call" in err_msg.lower() or "too many" in err_msg.lower():
                                contents.append(types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text="[System Directive: Tool call limit reached. Please synthesize your analytical report directly without additional tool invocations.]")]
                                ))
                                tool_config.tools = None
                            if "thinking" in err_msg.lower() or "budget" in err_msg.lower():
                                tool_config.thinking_config = None
                            await asyncio.sleep(0.5)
                            continue

                        elif err_code in [500, 502, 503, 504] or "UNAVAILABLE" in err_msg.upper() or "high demand" in err_msg.lower():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=self._notify_ui,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.5)
                            continue

                        elif err_code == 404 or "not found" in err_msg.lower():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=self._notify_ui,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.3)
                            continue

                        elif err_code == 401:
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                event_callback=self._notify_ui,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.3)
                            continue

                        else:
                            if attempt == max_api_retries - 1:
                                return f"Error executing task '{task_id}': Gemini API Error [{err_code}]: {err_msg}"
                            await asyncio.sleep(0.5)
                            continue

                    except asyncio.TimeoutError:
                        last_api_error_str = "Timeout"
                        gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                            "Timeout",
                            agent=self.agent_core,
                            estimated_tokens=current_tokens
                        )
                        active_model = self.agent_core.key_manager.get_model()
                        await asyncio.sleep(0.3)
                        continue

                    except asyncio.CancelledError:
                        raise
                    except Exception as ex:
                        err_txt = str(ex)
                        gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                            ex,
                            agent=self.agent_core,
                            estimated_tokens=current_tokens
                        )
                        active_model = self.agent_core.key_manager.get_model()
                        if attempt == max_api_retries - 1:
                            return f"Error executing task '{task_id}': {err_txt}"
                        await asyncio.sleep(0.5)
                        continue

                if response is None:
                    return f"Error executing task '{task_id}': Failed to obtain response ({last_api_error_str})"

                function_calls = []
                resp_text = ""

                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for p in response.candidates[0].content.parts:
                        if getattr(p, "function_call", None) is not None:
                            function_calls.append(p.function_call)
                        elif getattr(p, "text", None) and p.text:
                            resp_text += p.text + "\n"

                resp_text = resp_text.strip()

                if not function_calls and resp_text:
                    parsed_calls = extract_pseudo_text_tool_calls(resp_text, self.agent_core.tools_map)
                    if parsed_calls:
                        function_calls.extend(parsed_calls)

                if function_calls:
                    if len(function_calls) > 6:
                        function_calls = function_calls[:6]
                    model_parts = list(response.candidates[0].content.parts) if (response.candidates and response.candidates[0].content and response.candidates[0].content.parts) else []
                    for fc in model_parts:
                        if getattr(fc, "function_call", None) is not None:
                            fc.thought_signature = b"skip_thought_signature_validator"
                    model_turn = types.Content(role="model", parts=model_parts)
                    contents.append(model_turn)
                    await self.db.save_subagent_turn(
                        agent_id=agent_id,
                        role="model",
                        text=f"[Tool Invocations: {', '.join(c.name for c in function_calls)}]",
                        content_obj=model_turn,
                        task_id=task_id
                    )

                    tool_responses = []
                    for call in function_calls:
                        fn_name = call.name
                        fn_args = call.args or {}
                        args_preview = json.dumps(clean_for_json(fn_args), ensure_ascii=False) if fn_args else "{}"
                        if len(args_preview) > 120:
                            args_preview = args_preview[:120] + "..."

                        # Live UI stream of subagent tool execution
                        await self._notify_ui("tool_start", f"`[{agent_id}]` ⚙️ `{fn_name}` args: `{args_preview}`")

                        tool_fn = self.agent_core.tools_map.get(fn_name)
                        if tool_fn:
                            try:
                                import inspect
                                if inspect.iscoroutinefunction(tool_fn):
                                    res = await tool_fn(**fn_args)
                                else:
                                    res = await asyncio.to_thread(tool_fn, **fn_args)
                            except Exception as terr:
                                res = f"Error in tool '{fn_name}': {str(terr)}"
                        else:
                            res = f"Error: Tool '{fn_name}' is not in authorized toolset."

                        res_str = str(res)
                        preview = res_str[:600] + (" ...[truncated]" if len(res_str) > 600 else "")
                        await self._notify_ui("tool_end", f"**Result from `{fn_name}` (`{agent_id}`):**\n```text\n" + preview + "\n```")

                        capped_res = truncate_tool_response_text(res_str, max_tool_chars)
                        tool_resp_part = types.Part.from_function_response(name=fn_name, response={"result": capped_res})
                        tool_responses.append(tool_resp_part)

                    user_tool_turn = types.Content(role="user", parts=tool_responses)
                    contents.append(user_tool_turn)
                    await self.db.save_subagent_turn(
                        agent_id=agent_id,
                        role="tool_result",
                        text=f"[Tool Responses for: {', '.join(c.name for c in function_calls)}]",
                        content_obj=user_tool_turn,
                        task_id=task_id
                    )
                    continue

                if resp_text:
                    final_answer = resp_text
                    model_final_content = types.Content(role="model", parts=[types.Part.from_text(text=resp_text)])
                    await self.db.save_subagent_turn(
                        agent_id=agent_id,
                        role="model",
                        text=resp_text,
                        content_obj=model_final_content,
                        task_id=task_id
                    )
                    break

            if not final_answer:
                final_answer = f"[Subagent '{agent_id}' completed execution after {turns_used} turns without emitting final text report.]"

            # Stream final analytical output to UI
            await self._notify_ui("agent_message", f"### 💬 [{agent_record['name']} (`{agent_id}`)]:\n" + str(final_answer))
            end_t = time.time()
            duration = end_t - t_start

            await self.db.update_agent_task_status(
                task_id=task_id,
                status="COMPLETED",
                result=final_answer,
                end_time=end_t,
                turns_used=turns_used
            )

            await self.db.record_agent_message(
                from_agent_id=agent_id,
                to_agent_id=creator_id,
                msg_type="task_result",
                subject=f"Task Completed: {task_id}",
                body=final_answer[:1500],
                task_id=task_id
            )

            if hasattr(self, "bus") and self.bus:
                await self.bus.emit(
                    topic="TASK_COMPLETED",
                    sender_agent_id=agent_id,
                    payload={"creator": creator_id, "turns_used": turns_used, "result_snippet": final_answer[:300]},
                    summary_text=f"Completed task `{task_id}` in {duration:.1f}s ({turns_used} turns)",
                    task_id=task_id
                )

            logger.info(f"Subagent '{agent_id}' finished task '{task_id}' successfully in {duration:.1f}s.")
            return final_answer

        except asyncio.CancelledError:
            await self.db.update_agent_task_status(
                task_id=task_id,
                status="KILLED",
                error="Subagent task cancelled by operator signal."
            )
            raise
        except Exception as e:
            logger.error(f"Error executing subagent task '{task_id}': {str(e)}")
            await self.db.update_agent_task_status(
                task_id=task_id,
                status="FAILED",
                error=str(e),
                end_time=time.time(),
                turns_used=turns_used
            )
            return f"Error executing task '{task_id}': {str(e)}"
        finally:
            self._active_task_jobs.pop(task_id, None)

    async def spawn_ephemeral_agent(
        self,
        task_domain: str,
        task_prompt: str,
        parent_id: str = "root",
        custom_tools: Optional[List[str]] = None,
        timeout_seconds: float = 180.0,
        auto_cleanup: bool = True
    ) -> str:
        ts = int(time.time())
        clean_domain = re.sub(r'[^a-zA-Z0-9_]', '_', task_domain.strip().lower())
        ephemeral_id = f"ephemeral_{clean_domain}_{ts % 100000:05d}"

        ephemeral_prompt = (
            f"You are an on-demand ephemeral micro-agent specialized strictly in: '{task_domain}'.\n"
            f"Objective: Execute the requested task with laser focus, zero placeholders, and maximum efficiency.\n"
            f"Synthesize your output concisely and report back directly."
        )

        await self.create_agent(
            name=f"MicroAgent ({task_domain})",
            role="custom",
            system_prompt=ephemeral_prompt,
            allowed_tools=custom_tools or ["*"],
            can_communicate_with_peers=True,
            allowed_peers=[parent_id],
            max_turns=30,
            agent_id=ephemeral_id,
            parent_id=parent_id
        )

        try:
            result = await self.dispatch_task(
                assigned_agent_id=ephemeral_id,
                task_prompt=task_prompt,
                creator_agent_id=parent_id,
                wait_for_result=True,
                timeout_seconds=timeout_seconds
            )
            return result
        finally:
            if auto_cleanup:
                try:
                    await self.delete_agent(agent_id=ephemeral_id, acting_agent_id=parent_id)
                    logger.info(f"Ephemeral micro-agent '{ephemeral_id}' cleaned up successfully.")
                except Exception:
                    pass

    async def get_swarm_telemetry(self) -> Dict[str, Any]:
        agents = await self.db.list_sub_agents(active_only=True)
        running_tasks = await self.db.list_agent_tasks(status="RUNNING")
        completed_tasks = await self.db.list_agent_tasks(status="COMPLETED", limit=10)
        recent_messages = await self.db.get_agent_inbox_messages(agent_id="root", limit=10)

        return {
            "total_agents": len(agents),
            "agents": agents,
            "running_tasks_count": len(running_tasks),
            "running_tasks": running_tasks,
            "recent_completed_tasks": completed_tasks,
            "recent_messages": recent_messages
        }
