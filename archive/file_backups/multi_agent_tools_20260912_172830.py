# karyon_agent_runtime/tools/multi_agent_tools.py
"""
===============================================================================
MULTI-AGENT COLLABORATION & CORTICAL HIERARCHY TOOLKIT (v30.1 MASTER)
Allows Agent and Operator to Author, Configure, Delegate Tasks, and Message
Specialized Cortical Sub-Agents (Researcher, Coder, Refactorer, Critic, Custom)
with Strict Hierarchical Permission Enforcement and SQLite Persistence.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("ProxyAgent.MultiAgentTools")


def _get_multi_agent_manager():
    """Lazily resolves the MultiAgentManager to eliminate circular import races."""
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    if not agent or not getattr(agent, "multi_agent_manager", None):
        raise RuntimeError("MultiAgentManager is not initialized on active agent instance.")
    return agent.multi_agent_manager


# === 1. CREATE SUB-AGENT ===
async def create_subagent(
    name: str,
    role: str = "custom",
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools_json: Optional[str] = None,
    blocked_tools_json: Optional[str] = None,
    can_communicate_with_peers: bool = True,
    allowed_peers_json: Optional[str] = None,
    max_turns: int = 30,
    agent_id: Optional[str] = None,
    parent_id: Optional[str] = "root"
) -> str:
    """
    Creates and registers a new specialized cortical subagent in SQLite storage.

    Args:
        name: Human-readable display name for the subagent (e.g. 'PAC Literature Researcher').
        role: Specialization preset: 'researcher', 'coder', 'refactorer', 'critic', or 'custom'.
        system_prompt: Custom system prompt instructions. If omitted for presets, automatically applies KEP battle-tested prompts.
        model: Target Gemini model identifier (default: active model).
        allowed_tools_json: Optional JSON list of allowed tool names (e.g. '["read_file", "write_file"]'). Use '["*"]' for all.
        blocked_tools_json: Optional JSON list of explicitly forbidden tool names.
        can_communicate_with_peers: If True, permits horizontal communication with peer subagents.
        allowed_peers_json: JSON list of peer agent IDs this subagent is authorized to contact.
        max_turns: Maximum allowed tool-calling turns per delegated task (default: 30).
        agent_id: Optional custom alphanumeric ID (e.g. 'agent_researcher_01'). If None, generated automatically.
        parent_id: ID of the creator agent enforcing lineage protection (default: 'root').
    """
    mgr = _get_multi_agent_manager()

    try:
        allowed_tools = json.loads(allowed_tools_json) if allowed_tools_json else None
    except Exception:
        allowed_tools = ["*"]

    try:
        blocked_tools = json.loads(blocked_tools_json) if blocked_tools_json else None
    except Exception:
        blocked_tools = []

    try:
        allowed_peers = json.loads(allowed_peers_json) if allowed_peers_json else None
    except Exception:
        allowed_peers = ["*"]

    try:
        record = await mgr.create_agent(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            can_communicate_with_peers=can_communicate_with_peers,
            allowed_peers=allowed_peers,
            max_turns=max_turns,
            agent_id=agent_id,
            parent_id=parent_id
        )

        tools_preview = ", ".join(record.get("allowed_tools", ["*"])[:6])
        return (
            f"=== Subagent Successfully Created ===\n"
            f"- Agent ID     : `{record['agent_id']}`\n"
            f"- Display Name : {record['name']}\n"
            f"- Role Preset  : `{record['role']}`\n"
            f"- Parent Agent : `{record['parent_id']}`\n"
            f"- Model        : `{record['model']}`\n"
            f"- Peer Comm    : {bool(record['can_communicate_with_peers'])}\n"
            f"- Active Tools : `{tools_preview}...`\n"
            f"- Max Turns    : {record['max_turns']}"
        )
    except Exception as e:
        return f"Error creating subagent: {str(e)}"


# === 2. UPDATE SUB-AGENT ===
async def update_subagent(
    agent_id: str,
    name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools_json: Optional[str] = None,
    blocked_tools_json: Optional[str] = None,
    can_communicate_with_peers: Optional[bool] = None,
    allowed_peers_json: Optional[str] = None,
    max_turns: Optional[int] = None,
    is_active: Optional[bool] = None,
    acting_agent_id: str = "root"
) -> str:
    """
    Updates the configuration of an existing subagent.
    ENFORCES LINEAGE PROTECTION: A child agent CANNOT modify its parent or any ancestor.

    Args:
        agent_id: ID of the subagent to update.
        name: New display name.
        system_prompt: New system instruction text.
        model: New model identifier.
        allowed_tools_json: JSON list of allowed tools.
        blocked_tools_json: JSON list of blocked tools.
        can_communicate_with_peers: Peer communication flag.
        allowed_peers_json: JSON list of permitted peer IDs.
        max_turns: New max turns limit.
        is_active: Toggle active/inactive state.
        acting_agent_id: ID of the agent requesting the modification (default: 'root').
    """
    mgr = _get_multi_agent_manager()
    kwargs = {}
    if name is not None: kwargs["name"] = name.strip()
    if system_prompt is not None: kwargs["system_prompt"] = system_prompt.strip()
    if model is not None: kwargs["model"] = model.strip()
    if max_turns is not None: kwargs["max_turns"] = int(max_turns)
    if is_active is not None: kwargs["is_active"] = bool(is_active)
    if can_communicate_with_peers is not None: kwargs["can_communicate_with_peers"] = bool(can_communicate_with_peers)

    if allowed_tools_json is not None:
        try: kwargs["allowed_tools_json"] = json.loads(allowed_tools_json)
        except Exception: pass
    if blocked_tools_json is not None:
        try: kwargs["blocked_tools_json"] = json.loads(blocked_tools_json)
        except Exception: pass
    if allowed_peers_json is not None:
        try: kwargs["allowed_peers_json"] = json.loads(allowed_peers_json)
        except Exception: pass

    try:
        updated = await mgr.update_agent(agent_id=agent_id, acting_agent_id=acting_agent_id, **kwargs)
        return (
            f"=== Subagent Updated Successfully ===\n"
            f"- Agent ID : `{updated['agent_id']}` ({updated['name']})\n"
            f"- Role     : `{updated['role']}` | Active: {bool(updated['is_active'])}\n"
            f"- Model    : `{updated['model']}`"
        )
    except PermissionError as p_err:
        return f"❌ Permission Denied: {str(p_err)}"
    except Exception as e:
        return f"Error updating subagent '{agent_id}': {str(e)}"


# === 3. DELETE SUB-AGENT ===
async def delete_subagent(agent_id: str, acting_agent_id: str = "root") -> str:
    """
    Deletes a subagent from SQLite storage.
    ENFORCES LINEAGE PROTECTION: A child agent CANNOT delete its parent or any ancestor.

    Args:
        agent_id: ID of the subagent to delete.
        acting_agent_id: ID of the agent issuing the deletion (default: 'root').
    """
    mgr = _get_multi_agent_manager()
    try:
        success = await mgr.delete_agent(agent_id=agent_id, acting_agent_id=acting_agent_id)
        if success:
            return f"Success: Subagent `{agent_id}` has been deleted from the registry."
        return f"Notice: Subagent `{agent_id}` was not found."
    except PermissionError as p_err:
        return f"❌ Permission Denied: {str(p_err)}"
    except Exception as e:
        return f"Error deleting subagent '{agent_id}': {str(e)}"


# === 4. LIST SUB-AGENTS ===
async def list_subagents(parent_id: Optional[str] = None, active_only: bool = True) -> str:
    """
    Lists all registered cortical subagents, their roles, models, and hierarchy.

    Args:
        parent_id: Optional filter for a specific parent agent ID.
        active_only: If True, lists only active subagents (default: True).
    """
    mgr = _get_multi_agent_manager()
    agents = await mgr.db.list_sub_agents(parent_id=parent_id, active_only=active_only)
    if not agents:
        return f"No registered subagents found (Filter: parent={parent_id or 'ALL'}, active_only={active_only})."

    lines = [f"=== REGISTERED CORTICAL SUB-AGENTS ({len(agents)} agents) ==="]
    lines.append("| Agent ID | Name | Role | Parent | Model | Peers | Active |")
    lines.append("|---|---|---|---|---|---|---|")

    for a in agents:
        parent_tag = f"`{a['parent_id']}`" if a.get("parent_id") else "*[ROOT]*"
        active_tag = "✅ Yes" if a.get("is_active", 1) else "❌ Inactive"
        peer_tag = "🌐 Yes" if a.get("can_communicate_with_peers", 1) else "🔒 No"
        lines.append(
            f"| `{a['agent_id']}` | **{a['name']}** | `{a['role']}` | {parent_tag} | `{a.get('model', 'default')}` | {peer_tag} | {active_tag} |"
        )

    return "\n".join(lines)


# === 5. GET SUB-AGENT INFO ===
async def get_subagent_info(agent_id: str) -> str:
    """
    Retrieves full details, system prompt instructions, and tool whitelists for a subagent.

    Args:
        agent_id: The unique identifier of the subagent.
    """
    mgr = _get_multi_agent_manager()
    record = await mgr.db.get_sub_agent(agent_id)
    if not record:
        return f"Error: Subagent '{agent_id}' not found in registry."

    tools_str = ", ".join(f"`{t}`" for t in record.get("allowed_tools", ["*"]))
    blocked_str = ", ".join(f"`{t}`" for t in record.get("blocked_tools", [])) or "None"

    return (
        f"=== Subagent Profile: `{record['agent_id']}` ===\n"
        f"- Display Name     : {record['name']}\n"
        f"- Cortical Role    : `{record['role']}`\n"
        f"- Parent Lineage   : `{record.get('parent_id') or '[SOVEREIGN ROOT]'}`\n"
        f"- Inference Model  : `{record.get('model')}`\n"
        f"- Thinking Budget  : {record.get('thinking_budget')} tokens\n"
        f"- Peer Comm Policy : {'Enabled (Peer-to-Peer Authorized)' if record.get('can_communicate_with_peers') else 'Restricted (Parent-Only)'}\n"
        f"- Allowed Tools    : {tools_str}\n"
        f"- Blocked Tools    : {blocked_str}\n"
        f"- Max Turns Limit  : {record.get('max_turns')} turns\n\n"
        f"--- System Prompt Instructions ---\n```markdown\n{record.get('system_prompt', '')}\n```"
    )


# === 6. DISPATCH AGENT TASK ===
async def dispatch_agent_task(
    agent_id: str,
    task_prompt: str,
    wait_for_result: bool = True,
    timeout_seconds: float = 300.0,
    creator_agent_id: str = "root"
) -> str:
    """
    Delegates an empirical task, literature inquiry, or coding request to a specialized subagent.

    Args:
        agent_id: ID of the subagent to execute the task (e.g. 'agent_researcher_01').
        task_prompt: Concrete instruction or objective for the subagent.
        wait_for_result: If True, waits for completion and returns the synthesized response. If False, runs in background and returns task_id.
        timeout_seconds: Execution timeout in seconds (default: 300.0s).
        creator_agent_id: ID of the delegating agent (default: 'root').
    """
    mgr = _get_multi_agent_manager()
    return await mgr.dispatch_task(
        assigned_agent_id=agent_id,
        task_prompt=task_prompt,
        creator_agent_id=creator_agent_id,
        wait_for_result=wait_for_result,
        timeout_seconds=timeout_seconds
    )


# === 7. SYNAPTIC INTER-AGENT MESSAGING ===
async def send_agent_message_to(
    to_agent_id: str,
    message: str,
    msg_type: str = "peer_discussion",
    task_id: Optional[str] = None,
    from_agent_id: str = "root",
    subject: Optional[str] = None
) -> str:
    """
    Transmits an inter-agent message across the synaptic multi-agent bus.
    Enforces communication policies (Parent-child always allowed; peer-to-peer governed by settings).

    Args:
        to_agent_id: Recipient agent ID.
        message: The message body or data exchange.
        msg_type: Message classification ('peer_discussion', 'task_assignment', 'critique', 'report').
        task_id: Optional associated task ID.
        from_agent_id: Sender agent ID (default: 'root').
        subject: Optional message subject header.
    """
    mgr = _get_multi_agent_manager()
    try:
        res = await mgr.send_message(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message=message,
            msg_type=msg_type,
            task_id=task_id,
            subject=subject or ""
        )
        return f"Success: Message #{res['message_id']} delivered from `{from_agent_id}` to `{to_agent_id}` (Type: {msg_type})."
    except PermissionError as p_err:
        return f"❌ Permission Denied: {str(p_err)}"
    except Exception as e:
        return f"Error sending inter-agent message: {str(e)}"


# === 8. GET AGENT INBOX ===
async def get_agent_inbox(agent_id: str = "root", unread_only: bool = False, limit: int = 20) -> str:
    """
    Inspects messages received by an agent.

    Args:
        agent_id: Target agent whose inbox to inspect (default: 'root').
        unread_only: If True, filters for unread messages only (default: False).
        limit: Max messages to return (default: 20).
    """
    mgr = _get_multi_agent_manager()
    messages = await mgr.db.get_agent_inbox_messages(agent_id=agent_id, unread_only=unread_only, limit=limit)
    if not messages:
        return f"Inbox for agent '{agent_id}' is empty."

    lines = [f"=== INBOX FOR AGENT `{agent_id}` ({len(messages)} messages) ==="]
    for m in messages:
        status_tag = "✉️ UNREAD" if m.get("status") == "DELIVERED" else "📭 READ"
        lines.append(
            f"### Message #{m['id']} [{status_tag}] From: `{m['from_agent_id']}` | Type: `{m['msg_type']}` ({m['created_at']})\n"
            f"- **Subject:** {m.get('subject') or 'No Subject'}\n"
            f"```text\n{m['body']}\n```\n"
        )
        # Mark as read
        await mgr.db.mark_message_read(m["id"])

    return "\n".join(lines)


# === 9. GET AGENT DIALOGUE HISTORY ===
async def get_agent_dialogue_history(agent_id: str, task_id: Optional[str] = None, limit: int = 30) -> str:
    """
    Reads the autonomous tool-execution and dialogue trace for a specific subagent.

    Args:
        agent_id: Subagent identifier.
        task_id: Optional task ID filter.
        limit: Maximum turns to retrieve (default: 30).
    """
    mgr = _get_multi_agent_manager()
    turns = await mgr.db.get_subagent_turns(agent_id=agent_id, task_id=task_id, limit=limit)
    if not turns:
        return f"No dialogue turns recorded for subagent '{agent_id}' (Task: {task_id or 'ALL'})."

    lines = [f"=== Dialogue Trace for Subagent `{agent_id}` ({len(turns)} turns) ==="]
    for t in turns:
        role_label = f"🤖 [{t['role'].upper()}]" if t['role'] == "model" else f"⚙️ [{t['role'].upper()}]"
        lines.append(f"{role_label} ({t['timestamp']}):\n{t['text']}\n---")

    return "\n".join(lines)


# === 10. GET AGENT TASK STATUS ===
async def get_agent_task_status(task_id: str) -> str:
    """
    Polls the status and output of an asynchronous background subagent task.

    Args:
        task_id: The unique task ID (e.g. 'task_agent_researcher_...').
    """
    mgr = _get_multi_agent_manager()
    task = await mgr.db.get_agent_task(task_id)
    if not task:
        return f"Error: Task `{task_id}` not found in registry."

    status = task.get("status", "UNKNOWN")
    duration = task.get("duration")
    dur_str = f"{duration:.1f}s" if duration else "In-flight"

    lines = [
        f"=== Delegated Agent Task Telemetry: `{task_id}` ===",
        f"- Status         : **{status}**",
        f"- Assigned Agent : `{task['assigned_agent_id']}`",
        f"- Creator Agent  : `{task['creator_agent_id']}`",
        f"- Execution Time : {dur_str}",
        f"- Turns Used     : {task.get('turns_used', 0)} turns",
        f"- Initial Prompt : {task.get('task_prompt')[:200]}..."
    ]

    if task.get("result"):
        lines.append(f"\n--- Synthesized Task Output ---\n{task['result']}")
    if task.get("error"):
        lines.append(f"\n❌ **Error Encountered:** `{task['error']}`")

    return "\n".join(lines)


# === 11. DYNAMIC EPHEMERAL MICRO-AGENT SPAWN ===
async def spawn_ephemeral_subagent(
    task_domain: str,
    task_prompt: str,
    parent_id: str = "root",
    custom_tools_json: Optional[str] = None,
    timeout_seconds: float = 180.0,
    auto_cleanup: bool = True
) -> str:
    """
    Dynamically spawns an on-demand ephemeral micro-agent for a narrow specialized task
    (e.g. 'cuda_ptx_optimizer', 'arxiv_math_extractor', 'dataset_streaming_validator'),
    executes the task, returns the synthesized output, and automatically cleans up the agent.

    Args:
        task_domain: Narrow domain description (e.g. 'cuda_kernel', 'literature_search').
        task_prompt: Concrete instruction to execute.
        parent_id: Creator agent ID (default: 'root').
        custom_tools_json: Optional JSON list of tools to grant to this micro-agent.
        timeout_seconds: Execution timeout in seconds (default: 180.0s).
        auto_cleanup: If True, automatically deletes the agent from registry upon completion.
    """
    mgr = _get_multi_agent_manager()
    try:
        custom_tools = json.loads(custom_tools_json) if custom_tools_json else None
    except Exception:
        custom_tools = None

    try:
        res = await mgr.spawn_ephemeral_agent(
            task_domain=task_domain,
            task_prompt=task_prompt,
            parent_id=parent_id,
            custom_tools=custom_tools,
            timeout_seconds=timeout_seconds,
            auto_cleanup=auto_cleanup
        )
        return f"=== Ephemeral Micro-Agent Task Complete ({task_domain}) ===\n\n{res}"
    except Exception as e:
        return f"Error executing ephemeral micro-agent: {str(e)}"


# === 12. GET SWARM TELEMETRY ===
async def get_swarm_telemetry() -> str:
    """
    Returns deep real-time telemetry of the Cortical Sub-Agents Swarm:
    active agents, live background tasks, turn counts, and inter-agent synaptic message traffic.
    """
    mgr = _get_multi_agent_manager()
    data = await mgr.get_swarm_telemetry()

    lines = [
        "=== CORTICAL SUB-AGENTS SWARM TELEMETRY ===",
        f"- Total Active Agents : {data['total_agents']}",
        f"- In-Flight Tasks     : {data['running_tasks_count']}",
        f"- Swarm Mode Enabled  : {getattr(config, 'SWARM_MODE', True)}\n",
        "| Agent ID | Role | Display Name | Model | Parent | Turns |",
        "|---|---|---|---|---|---|"
    ]

    for a in data["agents"]:
        p_str = a.get("parent_id") or "[ROOT]"
        lines.append(f"| `{a['agent_id']}` | `{a['role']}` | {a['name']} | `{a.get('model')}` | `{p_str}` | {a.get('max_turns')} |")

    if data["running_tasks"]:
        lines.append("\n--- Live In-Flight Swarm Tasks ---")
        for t in data["running_tasks"]:
            lines.append(f"• **Task `{t['task_id']}`** -> Assigned to `{t['assigned_agent_id']}` (Prompt: {t['task_prompt'][:80]}...)")

    return "\n".join(lines)
