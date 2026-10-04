# karyon_agent_runtime/tools/control_tools.py
"""
===============================================================================
AGENT FLOW CONTROL & DELAY TOOLS (v26.0 MASTER)
Features Explicit No-Op (Ignore/Pass) with Turn Conclusion Flag and
Asynchronous Non-Blocking Sleep Delay Execution Engine.
===============================================================================
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("ProxyAgent.ControlTools")


async def ignore_or_noop(reason: str = "No action required", conclude_cycle: bool = False) -> str:
    """
    A no-operation (ignore/pass) tool. Performs no state modifications.
    Allows the model to explicitly skip actions or formally conclude the current turn loop.

    Args:
        reason: Explanation of why no action is needed (e.g. 'Waiting for training epoch', 'Analysis finished').
        conclude_cycle: If True, concludes the current turn loop immediately with this reason. If False, keeps the tool loop active.
    """
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()

    if conclude_cycle and agent:
        msg = f"Turn successfully concluded via ignore_or_noop. Reason: {reason}"
        await agent.handle_in_loop_message(msg, is_final=True)
        return msg

    return f"Action ignored (no-op). Cycle continues without state modification. Reason: {reason}"


async def sleep_delay(seconds: float = 5.0, reason: Optional[str] = None) -> str:
    """
    Pauses agent execution asynchronously for the specified number of seconds before continuing the tool cycle.
    Useful for giving background jobs, compilations, or file writes time to produce logs without spinning tight loops.

    Args:
        seconds: Number of seconds to pause (min: 0.1s, max: 3600.0s).
        reason: Optional reason for the pause (e.g. 'Waiting 10s for CUDA training to flush log file').
    """
    pause_time = max(0.1, min(float(seconds), 3600.0))
    logger.info(f"Pausing execution for {pause_time:.1f}s (Reason: {reason or 'Scheduled wait'})...")
    await asyncio.sleep(pause_time)
    reason_str = f" Reason: {reason}" if reason else ""
    return f"=== Delay Completed ===\nSuccessfully paused execution for {pause_time:.1f}s.{reason_str}\nThe tool execution cycle is now continuing."
