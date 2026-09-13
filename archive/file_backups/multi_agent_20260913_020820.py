# karyon_agent_runtime/multi_agent.py
"""
===============================================================================
KARYON CORE HIERARCHICAL MULTI-AGENT CORTICAL ORCHESTRATION ENGINE (v31.2 MASTER)
Implements Laminar Cortical Columnar Specialization (Principle 2 Biophysical Realism),
Hierarchical Lineage Protection (Descendant Ownership, Ancestor Immunity),
Dynamic Synaptic Inter-Agent Messaging, Autonomous Task Delegation, and 100%
Architectural Parity with Root (36-Key Quota Immunity, Sanitization & Tool Capping).
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import re
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
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
from karyon_agent_runtime.db_manager import clean_for_json, content_to_dict
from karyon_agent_runtime.context_compactor import (
    count_contents_tokens,
    sanitize_and_align_contents,
    truncate_tool_response_text
)

logger = logging.getLogger("ProxyAgent.MultiAgent")


class AgentRolePresets:
    """
    Standardized Cortical Laminar Presets grounded in Biophysical Realism (KEP Principle 2):
    - Researcher (Layer IV - Granular / Sensory Afferents): Literature grounding, arXiv formulas, evidence extraction.
    - Coder (Layer V - Large Pyramidal / Motor Efference): C++20 LibTorch, CUDA, PyTorch, zero placeholders.
    - Refactorer (Layer II/III - Cortico-cortical / Homeostasis): Principle 9 Dual-Refactoring (Axis A & Axis B).
    - Critic (Apical Dendrites / Top-down Prediction Error): KEP Rule #2 empirical verification & telemetry audit.
    """

    RESEARCHER_PROMPT = """You are the Dedicated Literature & Biophysical Research Agent for Karyon-CoRE.
Your mission is to explore theoretical literature, extract exact mathematical equations, search arXiv and the web,
and provide mathematically verified empirical foundations for Karyon's living AGI architecture.

STRICT RESEARCH PROTOCOLS:
1. FOCUS ON BIOPHYSICAL REALISM: Ground all research in Karl Friston's Active Inference (variational free energy F),
   György Buzsáki's neural oscillations (theta-gamma PAC), W. Ross Ashby's somatic homeostasis, and Modern Hopfield energy.
2. PRESERVE EXACT FORMULAS: Never hand-wave or approximate mathematics. Provide exact LaTeX formulas and parameters.
3. EMPIRICAL CITATIONS: Cite arXiv paper IDs, author names, and publication dates for every claim.
4. ACTION OVER CHATTER: Use tools (`arxiv_search_papers`, `arxiv_get_paper_details`, `internet_search`, `read_file`)
   to obtain concrete evidence, then synthesize a concise analytical report for your parent or peer agent.
"""

    CODER_PROMPT = """You are the Lead Implementation & High-Performance Engineering Agent for Karyon-CoRE.
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

    REFACTORER_PROMPT = """You are the Principle 9 Continuous Spontaneous Dual-Refactoring Agent for Karyon-CoRE.
Your mission is to continuously audit, detect, and eradicate technical bugs and non-biological crutches.

STRICT REFACTORING PROTOCOLS (TWO-AXIS AUDIT):
1. AXIS A (HARDWARE SAFETY & SPEED REGRESSIONS):
   - Eradicate PCIe synchronization stalls (`.item()` in loops).
   - Detect and fix division-by-zero risks, missing LayerNorms, and tensor shape broadcast mismatches.
   - Ensure continuous packed streaming (S=2048, 0% padding) and zero memory leaks.
2. AXIS B (BIOPHYSICAL REALISM - PRINCIPLE 2 NON-NEGOTIABLE):
   - Eliminate discrete transformer shortcuts and artificial crutches (No discrete Mixture-of-Experts / MoE).
   - Enforce continuous neural oscillations (theta-gamma PAC), unit-sphere Hopfield energy, and Ashby somatic homeostasis.
3. CONCLUDE WITH AUDIT TABLE: Summarize all detected bottlenecks and refactoring deltas cleanly for your caller.
"""

    CRITIC_PROMPT = """You are the Adversarial Empirical Auditor & KEP Rule #2 Critic for Karyon-CoRE.
Your mission is to ruthlessly critique hypotheses, benchmark telemetry, and empirical metrics to prevent
confirmation bias, premature conclusions, and regressions.

STRICT CRITIQUE PROTOCOLS:
1. KEP RULE #2 DECISION ENGINE:
   - Positive (🟢 POSITIVE): Mandates statistically significant breakthrough in target metric (e.g. Loss Delta >= 0.08,
     throughput >= 110%, or clear Free Energy drop) with ZERO degradation in secondary invariants.
   - Neutral (⚪ NEUTRAL): Performance within noise margin (+-0.05). Demands further biophysical refinement.
   - Rejected (🔴 REJECTED): Divergence, NaN overflow, throughput drop, or pseudo-morphemic drift.
2. AUDIT DIAGNOSTIC SPEECH: Audit live text generation samples for phonotactic coherence and absence of drift.
3. UNAMBIGUOUS VERDICT: Provide clear, data-driven verdicts with actionable advice for next architectural iterations.
"""

    PRESET_MAP = {
        "researcher": {
            "name": "Biophysical Literature & Evidence Researcher",
            "prompt": RESEARCHER_PROMPT,
            "tools": [
                "arxiv_search_papers", "arxiv_get_paper_details", "arxiv_download_pdf",
                "internet_search", "internet_media_search", "internet_deep_search",
                "scrape_url", "read_file", "list_directory", "search_codebase",
                "send_agent_message_to", "get_agent_inbox"
            ]
        },
        "coder": {
            "name": "C++20 & PyTorch Architecture Coder",
            "prompt": CODER_PROMPT,
            "tools": [
                "read_file", "write_file", "execute_python_code", "execute_cpp_code",
                "execute_code_with_shebang", "run_bash_command", "diff_files",
                "batch_replace_text", "build_and_verify_cpp_core", "inspect_cuda_environment",
                "benchmark_cpp_kernel", "send_agent_message_to", "get_agent_inbox"
            ]
        },
        "refactorer": {
            "name": "Principle 9 Dual-Refactoring Auditor",
            "prompt": REFACTORER_PROMPT,
            "tools": [
                "validate_kep_compliance", "search_codebase", "read_file", "write_file",
                "diff_files", "batch_replace_text", "git_status", "git_diff", "git_add",
                "git_commit", "send_agent_message_to", "get_agent_inbox"
            ]
        },
        "critic": {
            "name": "Empirical Telemetry Validator & Critic",
            "prompt": CRITIC_PROMPT,
            "tools": [
                "get_experiment_history", "get_latest_experiment", "read_file",
                "plot_experiment_comparison", "plot_training_loss_curves",
                "plot_phase_space_portrait", "run_bash_command",
                "send_agent_message_to", "get_agent_inbox"
            ]
        }
    }


class MultiAgentManager:
    """
    Hierarchical Multi-Agent Cortical Orchestrator.
    Manages creation, permissions, synaptic messaging, and task execution for specialized sub-agents.
    """

    def __init__(self, agent_core, db_manager):
        self.agent_core = agent_core
        self.db = db_manager
        self._active_task_jobs: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        """Ensures the root executive agent exists in the SQLite database."""
        root = await self.db.get_sub_agent("root")
        if not root:
            await self.db.create_sub_agent(
                agent_id="root",
                name="Karyon Root Executive Cyberneticist",
                role="orchestrator",
                parent_id=None,
                system_prompt="Executive cortical orchestrator responsible for overall scientific direction and task delegation.",
                model=self.agent_core.key_manager.get_model(),
                thinking_level="HIGH",
                thinking_budget=getattr(config, "THINKING_BUDGET", 24576),
                allowed_tools=["*"],
                can_communicate_with_peers=True,
                allowed_peers=["*"]
            )
            logger.info("MultiAgentManager: Initialized sovereign 'root' agent record in SQLite.")

        # Ensure default cortical laminar sub-agents exist
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
                    allowed_tools=r_info["tools"],
                    can_communicate_with_peers=True,
                    allowed_peers=["*"]
                )
                logger.info(f"MultiAgentManager: Pre-registered cortical sub-agent '{agent_uid}'.")

    def _resolve_tools_for_agent(self, agent_data: Dict[str, Any]) -> List[Any]:
        """Resolves the allowed list of callable tools for a subagent."""
        all_tools = getattr(self.agent_core, "tools_map", {})
        allowed = agent_data.get("allowed_tools") or ["*"]
        blocked = agent_data.get("blocked_tools") or []

        resolved = []
        for name, fn in all_tools.items():
            if name in blocked:
                continue
            if "*" in allowed or name in allowed:
                resolved.append(fn)

        return resolved

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
        max_turns: int = 30,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = "root",
        metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Creates a new subagent with preset or custom configuration."""
        clean_role = role.lower().strip()
        preset = AgentRolePresets.PRESET_MAP.get(clean_role)

        target_name = name.strip() or (preset["name"] if preset else f"Agent_{clean_role.capitalize()}")
        target_prompt = (system_prompt or "").strip()
        if not target_prompt and preset:
            target_prompt = preset["prompt"]
        if not target_prompt:
            target_prompt = f"You are a specialized {clean_role} agent assisting Bazilevs and Karyon-CoRE."

        target_tools = allowed_tools
        if target_tools is None and preset:
            target_tools = preset["tools"]
        if target_tools is None:
            target_tools = ["*"]

        target_model = (model or "").strip() or self.agent_core.key_manager.get_model()

        if not agent_id:
            ts = int(time.time())
            tag = clean_role if clean_role != "custom" else "sub"
            target_id = f"agent_{tag}_{ts % 100000:05d}"
        else:
            target_id = re.sub(r'[^a-zA-Z0-9_]', '_', agent_id.strip().lower())

        target_parent = parent_id.strip() if parent_id else "root"
        parent_record = await self.db.get_sub_agent(target_parent)
        if not parent_record and target_parent != "root":
            raise ValueError(f"Parent agent '{target_parent}' not found in database.")

        agent_record = await self.db.create_sub_agent(
            agent_id=target_id,
            name=target_name,
            role=clean_role,
            parent_id=target_parent,
            system_prompt=target_prompt,
            model=target_model,
            thinking_level="HIGH",
            thinking_budget=2048,
            allowed_tools=target_tools,
            blocked_tools=blocked_tools or [],
            can_communicate_with_peers=can_communicate_with_peers,
            allowed_peers=allowed_peers or ["*"],
            max_turns=max_turns,
            metadata=metadata or {}
        )

        logger.info(f"MultiAgentManager: Created subagent '{target_id}' ({target_name}, role: {clean_role}, parent: {target_parent}).")
        return agent_record

    async def update_agent(
        self,
        agent_id: str,
        acting_agent_id: str = "root",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Updates an existing subagent enforcing the Hierarchical Permission Invariant:
        A child CANNOT edit its parent or any ancestor. An agent can only manage its descendants.
        """
        target_id = agent_id.strip()
        acting_id = acting_agent_id.strip()

        can_manage, reason = await self.db.can_manage_agent(acting_id, target_id)
        if not can_manage:
            raise PermissionError(f"Unauthorized update: {reason}")

        success = await self.db.update_sub_agent(target_id, **kwargs)
        if not success:
            raise ValueError(f"Failed to update agent '{target_id}' (Agent may not exist).")

        return await self.db.get_sub_agent(target_id)

    async def delete_agent(
        self,
        agent_id: str,
        acting_agent_id: str = "root"
    ) -> bool:
        """Deletes an existing subagent enforcing the Hierarchical Permission Invariant."""
        target_id = agent_id.strip()
        acting_id = acting_agent_id.strip()

        can_manage, reason = await self.db.can_manage_agent(acting_id, target_id)
        if not can_manage:
            raise PermissionError(f"Unauthorized deletion: {reason}")

        # Cancel any active tasks running under this subagent
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
        subject: str = ""
    ) -> Dict[str, Any]:
        """
        Sends an inter-agent message with communication permission validation:
        - Parent <-> Child: ALWAYS allowed.
        - Root <-> Any: ALWAYS allowed.
        - Peer <-> Peer: Governed by sender's can_communicate_with_peers & allowed_peers.
        """
        from_id = from_agent_id.strip()
        to_id = to_agent_id.strip()

        from_agent = await self.db.get_sub_agent(from_id)
        if not from_agent and from_id != "root":
            raise ValueError(f"Sender agent '{from_id}' not found in database.")

        to_agent = await self.db.get_sub_agent(to_id)
        if not to_agent:
            raise ValueError(f"Recipient agent '{to_id}' not found in database.")

        # Permission verification
        authorized = False
        if from_id == "root" or to_id == "root":
            authorized = True
        elif from_agent and from_agent.get("parent_id") == to_id:
            authorized = True  # Child to Parent
        elif to_agent and to_agent.get("parent_id") == from_id:
            authorized = True  # Parent to Child
        elif from_agent:
            can_peer = from_agent.get("can_communicate_with_peers", True)
            allowed_peers = from_agent.get("allowed_peers") or ["*"]
            if can_peer and ("*" in allowed_peers or to_id in allowed_peers):
                authorized = True

        if not authorized:
            raise PermissionError(
                f"Communication Policy Violation: Agent '{from_id}' is not authorized to send messages to peer '{to_id}'."
            )

        msg_id = await self.db.record_agent_message(
            from_agent_id=from_id,
            to_agent_id=to_id,
            msg_type=msg_type,
            body=message.strip(),
            subject=subject.strip(),
            task_id=task_id
        )

        return {
            "message_id": msg_id,
            "from_agent_id": from_id,
            "to_agent_id": to_id,
            "msg_type": msg_type,
            "status": "DELIVERED"
        }

    async def dispatch_task(
        self,
        assigned_agent_id: str,
        task_prompt: str,
        creator_agent_id: str = "root",
        wait_for_result: bool = True,
        timeout_seconds: float = 300.0
    ) -> str:
        """
        Delegates a task to a subagent:
        - wait_for_result=True: executes and awaits completion, returning the analytical synthesis.
        - wait_for_result=False: launches asynchronously in background, returning task_id for polling.
        """
        assigned_id = assigned_agent_id.strip()
        creator_id = creator_agent_id.strip()

        agent_record = await self.db.get_sub_agent(assigned_id)
        if not agent_record:
            return f"Error: Target subagent '{assigned_id}' not found in registry."

        if not agent_record.get("is_active", 1):
            return f"Error: Target subagent '{assigned_id}' is currently marked INACTIVE."

        task_id = f"task_{assigned_id}_{int(time.time())}_{int(time.perf_counter()*1000)%1000:03d}"
        await self.db.create_agent_task(
            task_id=task_id,
            creator_agent_id=creator_id,
            assigned_agent_id=assigned_id,
            task_prompt=task_prompt
        )

        # Notify recipient of the new task assignment
        await self.db.record_agent_message(
            from_agent_id=creator_id,
            to_agent_id=assigned_id,
            msg_type="task_assignment",
            subject=f"New Task: {task_id}",
            body=task_prompt,
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
                f"*(Subagent is executing autonomously using its specialized toolset. Poll via `get_agent_task_status(task_id='{task_id}')`)*"
            )

    async def _run_subagent_task_worker(
        self,
        task_id: str,
        agent_record: Dict[str, Any],
        task_prompt: str,
        creator_id: str,
        timeout_seconds: float
    ) -> str:
        """
        Worker loop executing an autonomous subagent tool-calling turn sequence with
        100% architectural parity with root agent:
        - 36-key pool resilience & multi-tier cascading fallback (3.8 -> 3.7 -> 3.5 -> 2.5)
        - Zero-API-cost exact token valuation for TPM/RPM accounting
        - Historical tool response capping (MAX_HISTORICAL_TOOL_CHARS)
        - Full GenAI role & function alignment (sanitize_and_align_contents)
        - Fallback pseudo-text tool call interception
        """
        from karyon_agent_runtime.agent_core import extract_pseudo_text_tool_calls

        agent_id = agent_record["agent_id"]
        max_turns = int(agent_record.get("max_turns", 30))
        system_prompt = agent_record.get("system_prompt", "You are an autonomous subagent.")

        filtered_tools = self._resolve_tools_for_agent(agent_record)
        t_start = time.time()
        turns_used = 0
        final_answer = ""
        max_tool_chars = getattr(config, "MAX_HISTORICAL_TOOL_CHARS", 1200)

        # Prepare initial turn contents
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=task_prompt)])
        ]
        await self.db.save_subagent_turn(
            agent_id=agent_id,
            role="user",
            text=task_prompt,
            content_obj=contents[0],
            task_id=task_id
        )

        logger.info(f"Subagent '{agent_id}' commencing task '{task_id}' ({len(filtered_tools)} tools available)...")

        try:
            for turn in range(max_turns):
                turns_used += 1

                if getattr(self.agent_core, "autonomous_loop_paused", False) or (
                    not getattr(self.agent_core, "autonomous_loop_active", True)
                    and getattr(self.agent_core, "autonomous_loop_task", None)
                ):
                    return f"[Task '{task_id}' halted: Autonomous loop paused or stopped by operator.]"

                # 1. Full content alignment and accurate token valuation
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

                for attempt in range(max_api_retries):
                    if getattr(self.agent_core, "autonomous_loop_paused", False):
                        return "⏸️ Subagent execution halted: Autonomous loop paused."

                    # Acquire ready client with pre-flight quota reservation on EACH iteration
                    gemini_client = await self.agent_core.key_manager.get_ready_client_for_request(
                        estimated_tokens=current_tokens,
                        agent=self.agent_core
                    )

                    active_model = self.agent_core.key_manager.get_model()

                    tool_config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=filtered_tools if filtered_tools else None,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        temperature=0.2,
                        top_p=0.95,
                        max_output_tokens=config.MAX_OUTPUT_TOKENS
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
                        break

                    except APIError as e:
                        err_msg = str(e)
                        err_code = getattr(e, "code", 500)
                        last_api_error_str = f"APIError [{err_code}]: {err_msg}"
                        logger.warning(f"Subagent '{agent_id}' API Error [{err_code}] on '{active_model}': {err_msg}")

                        if err_code == 429 or "RESOURCE_EXHAUSTED" in err_msg.upper():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.5 + 0.1 * min(attempt, 5))
                            continue

                        elif err_code in [500, 502, 503, 504] or "UNAVAILABLE" in err_msg.upper() or "high demand" in err_msg.lower():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.5)
                            continue

                        elif err_code == 404 or "NOT_FOUND" in err_msg.upper() or "not found" in err_msg.lower():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
                                agent=self.agent_core,
                                estimated_tokens=current_tokens
                            )
                            active_model = self.agent_core.key_manager.get_model()
                            await asyncio.sleep(0.3)
                            continue

                        elif err_code == 401 or "unauthenticated" in err_msg.lower():
                            gemini_client = await self.agent_core.key_manager.handle_quota_exhausted(
                                e,
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
                        logger.warning(f"Subagent '{agent_id}' request timed out on '{active_model}'. Rotating key/model...")
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
                        logger.warning(f"Subagent '{agent_id}' request error: {err_txt}. Rotating key/model...")
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
                    return f"Error executing task '{task_id}': Failed to obtain response after {max_api_retries} attempts ({last_api_error_str})"

                function_calls = []
                resp_text = ""

                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for p in response.candidates[0].content.parts:
                        if getattr(p, "function_call", None) is not None:
                            function_calls.append(p.function_call)
                        elif getattr(p, "text", None) and p.text:
                            resp_text += p.text + "\n"

                resp_text = resp_text.strip()

                # Fallback to pseudo-text tool interception
                if not function_calls and resp_text:
                    parsed_calls = extract_pseudo_text_tool_calls(resp_text, self.agent_core.tools_map)
                    if parsed_calls:
                        function_calls.extend(parsed_calls)

                if function_calls:
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
                        # Cap output in context payload to preserve 250k TPM budget
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
        """
        Dynamically spawns an on-demand ephemeral micro-agent for a narrow, highly specialized domain task.
        Executes the delegated task, logs turns and messages, and automatically unregisters upon completion.
        """
        ts = int(time.time())
        clean_domain = re.sub(r'[^a-zA-Z0-9_]', '_', task_domain.strip().lower())
        ephemeral_id = f"ephemeral_{clean_domain}_{ts % 100000:05d}"

        ephemeral_prompt = (
            f"You are an on-demand ephemeral micro-agent specialized strictly in: '{task_domain}'.\n"
            f"Objective: Execute the requested task with laser focus, zero placeholders, and maximum efficiency.\n"
            f"Synthesize your output concisely and report back directly."
        )

        agent_record = await self.create_agent(
            name=f"MicroAgent ({task_domain})",
            role="custom",
            system_prompt=ephemeral_prompt,
            allowed_tools=custom_tools or ["*"],
            can_communicate_with_peers=True,
            allowed_peers=[parent_id],
            max_turns=20,
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
        """Returns active cortical sub-agents, live delegated tasks, and synaptic message metrics."""
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
