# karyon_agent_runtime/tools/multi_agent_tools.py
"""
===============================================================================
MULTI-AGENT COLLABORATION & CORTICAL HIERARCHY TOOLKIT (v32.6 MASTER)
Allows Agent and Operator to Author, Configure, Delegate Tasks, and Message
Specialized Cortical Sub-Agents with Synchronous Turn Handoff (transfer_turn_to_agent,
wait_for_reply), Parallel Swarm Batch Execution (dispatch_parallel_agent_tasks),
and Full Production Tool Parity (All 180 Tools Available to Every Agent).
===============================================================================
"""

import json
import logging
from typing import Optional
from karyon_agent_runtime import config

logger = logging.getLogger("ProxyAgent.MultiAgentTools")


def _get_multi_agent_manager():
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    if agent and getattr(agent, "multi_agent_manager", None):
        return agent.multi_agent_manager

    from karyon_agent_runtime.multi_agent import MultiAgentManager
    from karyon_agent_runtime.db_manager import AgentDBManager
    from karyon_agent_runtime.key_manager import GeminiKeyManager

    db = agent.db if (agent and getattr(agent, "db", None)) else AgentDBManager()
    km = agent.key_manager if (agent and getattr(agent, "key_manager", None)) else GeminiKeyManager(db)
    bus = agent.bus if (agent and getattr(agent, "bus", None)) else None

    mgr = MultiAgentManager(db=db, key_manager=km, bus=bus)
    if agent:
        agent.multi_agent_manager = mgr
        agent.db = db
        agent.key_manager = km
        if not getattr(agent, "bus", None):
            agent.bus = bus
    return mgr


async def create_subagent(
    name: str,
    role: str = "custom",
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools_json: Optional[str] = None,
    blocked_tools_json: Optional[str] = None,
    can_communicate_with_peers: bool = True,
    allowed_peers_json: Optional[str] = None,
    max_turns: int = 40,
    agent_id: Optional[str] = None,
    parent_id: Optional[str] = "root"
) -> str:
    mgr = _get_multi_agent_manager()

    try:
        allowed_tools = json.loads(allowed_tools_json) if allowed_tools_json else ["*"]
    except Exception:
        allowed_tools = ["*"]

    try:
        blocked_tools = json.loads(blocked_tools_json) if blocked_tools_json else []
    except Exception:
        blocked_tools = []

    try:
        allowed_peers = json.loads(allowed_peers_json) if allowed_peers_json else ["*"]
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
            f"- Active Tools : `{tools_preview}...` (Full 180 tool access active)\n"
            f"- Max Turns    : {record['max_turns']}"
        )
    except Exception as e:
        return f"Error creating subagent: {str(e)}"


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
    mgr = _get_multi_agent_manager()
    kwargs = {}
    if name is not None:
        kwargs["name"] = name.strip()
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt.strip()
    if model is not None:
        kwargs["model"] = model.strip()
    if max_turns is not None:
        kwargs["max_turns"] = int(max_turns)
    if is_active is not None:
        kwargs["is_active"] = bool(is_active)
    if can_communicate_with_peers is not None:
        kwargs["can_communicate_with_peers"] = bool(can_communicate_with_peers)

    if allowed_tools_json is not None:
        try:
            kwargs["allowed_tools_json"] = json.loads(allowed_tools_json)
        except Exception:
            pass
    if blocked_tools_json is not None:
        try:
            kwargs["blocked_tools_json"] = json.loads(blocked_tools_json)
        except Exception:
            pass
    if allowed_peers_json is not None:
        try:
            kwargs["allowed_peers_json"] = json.loads(allowed_peers_json)
        except Exception:
            pass

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


async def delete_subagent(agent_id: str, acting_agent_id: str = "root") -> str:
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


async def list_subagents(parent_id: Optional[str] = None, active_only: bool = True) -> str:
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


async def get_subagent_info(agent_id: str) -> str:
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


async def dispatch_agent_task(
    agent_id: str,
    task_prompt: str,
    wait_for_result: bool = True,
    timeout_seconds: Optional[float] = None,
    creator_agent_id: str = "root"
) -> str:
    mgr = _get_multi_agent_manager()
    return await mgr.dispatch_task(
        assigned_agent_id=agent_id,
        task_prompt=task_prompt,
        creator_agent_id=creator_agent_id,
        wait_for_result=wait_for_result,
        timeout_seconds=timeout_seconds
    )


async def send_agent_message_to(
    to_agent_id: str,
    message: str,
    msg_type: str = "peer_discussion",
    task_id: Optional[str] = None,
    from_agent_id: str = "root",
    subject: Optional[str] = None,
    wait_for_reply: bool = False,
    reply_timeout: float = 180.0
) -> str:
    mgr = _get_multi_agent_manager()
    try:
        res = await mgr.send_message(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message=message,
            msg_type=msg_type,
            task_id=task_id,
            subject=subject or "",
            wait_for_reply=wait_for_reply,
            reply_timeout=reply_timeout
        )
        if wait_for_reply and "reply" in res:
            return (
                f"=== Synaptic Exchange: Message #{res['message_id']} Replied ===\n"
                f"- Sender   : `{from_agent_id}`\n"
                f"- Recipient: `{to_agent_id}`\n\n"
                f"--- Reply from `{to_agent_id}` ---\n"
                f"{res['reply']}"
            )
        return f"Success: Message #{res['message_id']} delivered from `{from_agent_id}` to `{to_agent_id}` (Type: {msg_type})."
    except PermissionError as p_err:
        return f"❌ Permission Denied: {str(p_err)}"
    except Exception as e:
        return f"Error sending inter-agent message: {str(e)}"


async def transfer_turn_to_agent(
    target_agent_id: str,
    message_or_objective: str,
    return_control: bool = True,
    acting_agent_id: str = "root",
    timeout_seconds: float = 240.0
) -> str:
    """
    Transfers the conversational and computational turn synchronously to a specialized cortical agent.
    Awaits their full execution and returns their analytical response directly into the calling turn context.
    """
    mgr = _get_multi_agent_manager()
    try:
        res = await mgr.send_message(
            from_agent_id=acting_agent_id,
            to_agent_id=target_agent_id,
            message=message_or_objective,
            msg_type="turn_handoff",
            subject=f"Synchronous Turn Transfer from {acting_agent_id}",
            wait_for_reply=return_control,
            reply_timeout=timeout_seconds
        )
        if return_control and "reply" in res:
            return (
                f"=== Synchronous Turn Handoff Completed ===\n"
                f"- From: `{acting_agent_id}` ➔ To: `{target_agent_id}`\n\n"
                f"{res['reply']}"
            )
        return f"Turn transferred asynchronously to `{target_agent_id}` (Message #{res['message_id']})."
    except Exception as e:
        return f"Error during turn transfer to `{target_agent_id}`: {str(e)}"


async def dispatch_parallel_agent_tasks(
    tasks_json: str,
    creator_agent_id: str = "root",
    timeout_seconds: Optional[float] = None
) -> str:
    """
    Executes multiple subagent tasks concurrently in parallel across specialized cortical columns.
    Awaits all agents and synthesizes their combined outputs.

    Args:
        tasks_json: JSON list of objects with 'agent_id' and 'task_prompt'.
                    Example: '[{"agent_id": "agent_architect", "task_prompt": "..."}, {"agent_id": "agent_engineer", "task_prompt": "..."}]'
        creator_agent_id: ID of the delegating agent (default: 'root').
        timeout_seconds: Maximum wait timeout per task (default: 300.0s).
    """
    mgr = _get_multi_agent_manager()
    try:
        tasks_list = json.loads(tasks_json) if isinstance(tasks_json, str) else tasks_json
        if not isinstance(tasks_list, list) or not tasks_list:
            return "Error: tasks_json must be a non-empty JSON list of task objects."
    except Exception as j_err:
        return f"Error parsing tasks_json: {str(j_err)}"

    try:
        results = await mgr.dispatch_parallel_tasks(
            tasks_list=tasks_list,
            creator_agent_id=creator_agent_id,
            timeout_seconds=timeout_seconds
        )
        report_blocks = [f"=== Parallel Swarm Dispatch Results ({len(results)} agents executed concurrently) ==="]
        for r in results:
            report_blocks.append(
                f"### 🤖 Agent `{r['agent_id']}` [{r['status']}]:\n{r['result']}\n"
            )
        return "\n---\n".join(report_blocks)
    except Exception as e:
        return f"Error executing parallel agent tasks: {str(e)}"


async def get_agent_inbox(agent_id: str = "root", unread_only: bool = False, limit: int = 20) -> str:
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
        await mgr.db.mark_message_read(m["id"])

    return "\n".join(lines)


async def get_agent_dialogue_history(agent_id: str, task_id: Optional[str] = None, limit: int = 30) -> str:
    mgr = _get_multi_agent_manager()
    turns = await mgr.db.get_subagent_turns(agent_id=agent_id, task_id=task_id, limit=limit)
    if not turns:
        return f"No dialogue turns recorded for subagent '{agent_id}' (Task: {task_id or 'ALL'})."

    lines = [f"=== Dialogue Trace for Subagent `{agent_id}` ({len(turns)} turns) ==="]
    for t in turns:
        role_label = f"🤖 [{t['role'].upper()}]" if t['role'] == "model" else f"⚙️ [{t['role'].upper()}]"
        lines.append(f"{role_label} ({t['timestamp']}):\n{t['text']}\n---")

    return "\n".join(lines)


async def get_agent_task_status(task_id: str) -> str:
    mgr = _get_multi_agent_manager()
    task = await mgr.db.get_agent_task(task_id)
    if not task:
        return f"Error: Task `{task_id}` not found in registry."

    status = task.get("status", "UNKNOWN")
    duration = task.get("duration")
    dur_str = f"{duration:.1f}s" if duration else "In-flight"

    lines = [
        f"=== Delegated Agent Task Telemetry: `{task_id}` ===\n",
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


async def spawn_ephemeral_subagent(
    task_domain: str,
    task_prompt: str,
    parent_id: str = "root",
    custom_tools_json: Optional[str] = None,
    timeout_seconds: float = 180.0,
    auto_cleanup: bool = True
) -> str:
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


async def get_swarm_telemetry() -> str:
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


async def publish_synaptic_event(
    topic: str,
    payload_json: str,
    summary_text: str = "",
    task_id: Optional[str] = None,
    sender_agent_id: str = "root"
) -> str:
    """
    Publishes an asynchronous event to the SynapticMeshEventBus, broadcasting across
    all subscribed cortical columns and streaming the spike to the interactive user interface.

    Args:
        topic: Event classification topic (e.g. 'HYPOTHESIS_FORMULATED', 'CODE_VERIFIED', 'BENCHMARK_RESULT').
        payload_json: JSON string payload containing structured data or telemetry dict.
        summary_text: Human-readable analytical headline summarizing the event.
        task_id: Optional associated task identifier.
        sender_agent_id: Identifier of emitting agent (default: 'root').
    """
    mgr = _get_multi_agent_manager()
    try:
        payload = json.loads(payload_json) if isinstance(payload_json, str) else (payload_json or {})
    except Exception as je:
        payload = {"raw_payload": str(payload_json), "error": str(je)}

    try:
        ev = await mgr.bus.emit(
            topic=topic.strip(),
            sender_agent_id=sender_agent_id.strip(),
            payload=payload,
            summary_text=summary_text.strip(),
            task_id=task_id
        )
        return (
            f"=== Synaptic Spike Dispatched ===\n"
            f"- Topic  : `{ev['topic']}`\n"
            f"- Sender : `{ev['sender']}`\n"
            f"- Summary: {ev['summary'][:300]}\n"
            f"- Status : Propagated to Synaptic Mesh Bus & Subscribers"
        )
    except Exception as e:
        return f"Error publishing synaptic event: {str(e)}"
