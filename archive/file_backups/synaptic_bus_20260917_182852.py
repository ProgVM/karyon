# karyon_agent_runtime/synaptic_bus.py
"""
===============================================================================
KARYON SYNAPTIC MESH EVENT BUS (v33.0 MASTER)
Asynchronous, Reactive Event Distribution Substrate for Cortical Subagents.
Connects Researcher, Coder, Refactorer, and Critic in a continuous biophysical loop.
===============================================================================
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable

logger = logging.getLogger("ProxyAgent.SynapticBus")


class SynapticMeshEventBus:
    """
    In-memory asynchronous pub/sub event bus paired with persistent SQLite telemetry.
    Enables spontaneous inter-agent reaction, concurrent streaming, and peer-to-peer spikes.
    """

    def __init__(self, db_manager=None, ui_notifier: Optional[Callable[[str, str], Awaitable[None]]] = None):
        self.db = db_manager
        self.ui_notifier = ui_notifier
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    def set_ui_notifier(self, notifier: Optional[Callable[[str, str], Awaitable[None]]]):
        self.ui_notifier = notifier

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribes an asynchronous callback to a specific synaptic topic or '*' for all."""
        topic = topic.strip()
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if handler not in self._subscribers[topic]:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        topic = topic.strip()
        if topic in self._subscribers and handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)

    async def emit(
        self,
        topic: str,
        sender_agent_id: str,
        payload: Dict[str, Any],
        summary_text: str = "",
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Emits a spontaneous synaptic event.
        Dispatches concurrently to all topic subscribers and logs to persistent DB.
        """
        event = {
            "topic": topic,
            "sender": sender_agent_id,
            "payload": payload,
            "summary": summary_text or str(payload)[:300],
            "task_id": task_id,
            "timestamp": time.time()
        }

        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > 200:
                self._event_history = self._event_history[-100:]

        # 1. UI Notification
        if self.ui_notifier:
            try:
                icon = "⚡"
                if "CODE" in topic:
                    icon = "⚙️"
                elif "AUDIT" in topic or "LINT" in topic:
                    icon = "🔍"
                elif "BENCHMARK" in topic or "TELEMETRY" in topic:
                    icon = "📊"
                elif "HYPOTHESIS" in topic or "CHARTER" in topic:
                    icon = "🧬"

                short_sum = (summary_text or str(payload))[:250]
                await self.ui_notifier(
                    "info",
                    f"{icon} **[Synaptic Mesh Event: `{topic}`]** `{sender_agent_id}`: {short_sum}"
                )
            except Exception as e:
                logger.debug(f"UI notification error in SynapticMeshEventBus: {e}")

        # 2. Persist to SQLite agent_messages as broadcast if DB is available
        if self.db:
            try:
                await self.db.record_agent_message(
                    from_agent_id=sender_agent_id,
                    to_agent_id="*",
                    msg_type=f"event_{topic}",
                    body=summary_text or str(payload),
                    subject=f"Synaptic Event: {topic}",
                    task_id=task_id
                )
            except Exception as dbe:
                logger.debug(f"DB recording error in SynapticMeshEventBus: {dbe}")

        # 3. Trigger Subscribers Concurrently
        handlers_to_call = list(self._subscribers.get(topic, [])) + list(self._subscribers.get("*", []))
        if handlers_to_call:
            tasks = [asyncio.create_task(h(event)) for h in handlers_to_call]
            # Gather with exception swallowing so one agent crash does not kill the bus
            await asyncio.gather(*tasks, return_exceptions=True)

        return event

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._event_history[-limit:]
