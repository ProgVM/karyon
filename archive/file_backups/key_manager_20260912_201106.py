# karyon_agent_runtime/key_manager.py
"""
===============================================================================
36-KEY DYNAMIC POOL MANAGER & 250K TPM 36-PROJECT CROSS-ROTATION ENGINE (v31.0)
Features 36 Independent Project Quota Scopes (6 Accounts x 6 Projects), Exact
Google API RPC Error & Daily Quota Parser, Multi-Tier Cascading Model Failover
(3.8 -> 3.7 -> 3.5 -> 2.5 -> 2.0), Per-(Key, Model) Cooldowns, Preemptive
Sliding-Window RPM Throttling, and Model-Wide Midnight Daily Lockouts.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import time
import json
import re
import asyncio
import datetime
import logging
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Union
from google import genai
from google.genai import types
import karyon_agent_runtime.config as config
from karyon_agent_runtime.proxy_manager import proxy_rotator

logger = logging.getLogger("ProxyAgent.KeyManager")


def get_seconds_until_pacific_midnight() -> int:
    """Calculates remaining seconds until Pacific Midnight (PDT/PST) for daily quota reset."""
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        is_pdt = False
        if 3 < now_utc.month < 11:
            is_pdt = True
        elif now_utc.month == 3 and now_utc.day >= 14:
            is_pdt = True
        elif now_utc.month == 11 and now_utc.day < 7:
            is_pdt = True
        offset = config.PACIFIC_DAYLIGHT_TIME_OFFSET if is_pdt else config.PACIFIC_STANDARD_TIME_OFFSET
        pacific_now = now_utc + datetime.timedelta(hours=offset)
        tomorrow_pacific = datetime.datetime(
            year=pacific_now.year, month=pacific_now.month, day=pacific_now.day, hour=0, minute=0, second=0
        ) + datetime.timedelta(days=1)
        return max(60, int((tomorrow_pacific - pacific_now).total_seconds()))
    except Exception as e:
        logger.warning(f"Pacific midnight calculation notice: {str(e)}")
    return 86400


def parse_gemini_error_cooldown(ex_or_str: Any) -> Tuple[float, bool]:
    """
    Parses Google API errors to extract the EXACT required retry delay and quota type:
    1. Deep inspects structured RPC details (QuotaFailure.violations, quotaId, quotaMetric).
    2. Identifies daily caps (GenerateRequestsPerDay, RPD=20, Daily, PerDay) -> Pacific Midnight reset.
    3. Identifies transient rate limits (RPM/TPM, retryDelay).
    4. Enforces a 15.0s minimum cooldown for short-term rate limits to satisfy free-tier pacing.
    """
    ex_str = str(ex_or_str)
    err_lower = ex_str.lower()
    is_daily = False
    retry_delay_val: Optional[float] = None

    # 1. Deep inspect structured RPC error details
    details = getattr(ex_or_str, "details", None) or []
    for detail in details:
        if isinstance(detail, dict):
            # Inspect quota violations
            violations = detail.get("violations", [])
            for v in violations:
                if isinstance(v, dict):
                    q_id = str(v.get("quotaId", "")).lower()
                    q_metric = str(v.get("quotaMetric", "")).lower()
                    q_val = str(v.get("quotaValue", ""))
                    if any(w in q_id for w in ["perday", "daily", "per_day", "day-freetier"]):
                        is_daily = True
                    if any(w in q_metric for w in ["perday", "daily", "per_day"]):
                        is_daily = True
                    if q_val in ["20", "50"] and any(w in q_id or w in q_metric for w in ["request", "content"]):
                        is_daily = True

            # Extract retryDelay
            if "retryDelay" in detail:
                rd_str = str(detail["retryDelay"]).strip().rstrip("s")
                try:
                    retry_delay_val = float(rd_str)
                except ValueError:
                    pass

        elif hasattr(detail, "violations"):
            for v in getattr(detail, "violations", []):
                q_id = str(getattr(v, "quota_id", "") or getattr(v, "quotaId", "")).lower()
                if any(w in q_id for w in ["perday", "daily", "per_day", "day-freetier"]):
                    is_daily = True

    # 2. Textual daily quota indicator check
    daily_indicators = [
        "perday", "daily", "per_day", "requests per day", "requests_per_day",
        "day-freetier", "limit: 20", "quota metric 'generatecontentrequestsperday",
        "generatecontentrequestsperday", "generaterequestsperday"
    ]
    if any(ind in err_lower for ind in daily_indicators):
        is_daily = True

    if is_daily:
        cooldown = float(get_seconds_until_pacific_midnight())
        logger.warning(f"Daily quota reached (RPD exhausted). Cooldown until Pacific Midnight: {cooldown:.0f}s.")
        return cooldown, True

    # 3. Structured retryDelay if present
    if retry_delay_val is not None:
        return max(5.0, retry_delay_val + 1.0), False

    # 4. Regex parsing for retryDelay or textual patterns
    retry_delay_match = re.search(
        r'["\']?retryDelay["\']?\s*:\s*["\']?([0-9]+(?:\.[0-9]+)?)s?["\']?',
        ex_str,
        re.IGNORECASE
    )
    if retry_delay_match:
        try:
            val = float(retry_delay_match.group(1))
            return max(5.0, val + 1.0), False
        except ValueError:
            pass

    text_matches = re.findall(
        r'(?:retry|wait|try again)\s+(?:in|after)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|second|seconds)\b',
        ex_str,
        re.IGNORECASE
    )
    if text_matches:
        try:
            val = float(text_matches[0])
            return max(5.0, val + 1.0), False
        except ValueError:
            pass

    # 5. Default fallback for standard RPM rate limits (15.0s ensures clean 5 RPM recovery)
    return 15.0, False


class GeminiKeyManager:
    """
    36-Project Key Pool Coordinator with Multi-Tier Model Failover and RPM Throttling.
    Tracks quotas per (key_index, model) pair and enforces model-wide daily lockouts.
    """

    def __init__(self, db_manager=None):
        self.keys = list(dict.fromkeys([k.strip() for k in config.GEMINI_API_KEYS if k.strip()]))
        self.models = [m.strip() for m in config.GEMINI_MODELS if m.strip()]
        self.invalid_models = set()
        self.model_daily_exhaustion: Dict[str, float] = {}  # {model_name: reset_timestamp}
        self.current_key_index = 0
        self.current_model_index = 0
        self.db = db_manager

        # 6 Accounts Architecture (6 keys per account, each key is an independent project)
        self.num_accounts = getattr(config, "GEMINI_NUM_ACCOUNTS", 6)
        self.keys_per_account = max(1, len(self.keys) // self.num_accounts) if len(self.keys) >= self.num_accounts else 1

        self.interleaved_indices = self._build_interleaved_schedule()
        self.schedule_cursor = 0

        # State Telemetry: Cooldowns are strictly per (key_idx, model)!
        self.exhausted_keys: Dict[Tuple[int, str], float] = {}  # {(key_idx, model): cooldown_until_timestamp}
        self.last_used_timestamps: Dict[int, float] = {i: 0.0 for i in range(len(self.keys))}
        self.key_tpm_history: Dict[int, deque] = {i: deque() for i in range(len(self.keys))}
        self.key_rpm_history: Dict[int, deque] = {i: deque() for i in range(len(self.keys))}

        self._client = None
        self._init_client()

    def _build_interleaved_schedule(self) -> List[int]:
        """Builds interleaved schedule: cycles through accounts before repeating projects."""
        schedule = []
        for p_idx in range(self.keys_per_account):
            for acc_idx in range(self.num_accounts):
                flat_idx = acc_idx * self.keys_per_account + p_idx
                if flat_idx < len(self.keys):
                    schedule.append(flat_idx)
        return schedule if schedule else list(range(len(self.keys)))

    def get_account_for_key(self, key_idx: int) -> int:
        return min(self.num_accounts - 1, key_idx // self.keys_per_account)

    def get_project_for_key(self, key_idx: int) -> int:
        return key_idx % self.keys_per_account

    def _init_client(self):
        active_key = self.get_active_key()
        proxy_url = proxy_rotator.get_proxy()
        http_opts = None
        if proxy_url:
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            os.environ["ALL_PROXY"] = proxy_url
            os.environ["all_proxy"] = proxy_url
            http_opts = types.HttpOptions(
                client_args={"proxy": proxy_url},
                async_client_args={"proxy": proxy_url}
            )

        self._client = genai.Client(api_key=active_key, http_options=http_opts)
        acc_id = self.get_account_for_key(self.current_key_index)
        proj_id = self.get_project_for_key(self.current_key_index)
        logger.info(
            f"Active Credentials: Key #{self.current_key_index+1}/{len(self.keys)} "
            f"(Account #{acc_id+1}/{self.num_accounts}, Project #{proj_id+1}/{self.keys_per_account}) "
            f"[{active_key[:10]}...] | Model: {self.get_model()}"
        )

    def get_client(self) -> genai.Client:
        return self._client

    def _prune_sliding_windows(self, now: float):
        """Cleans rolling token & request records older than 60s and expires key-model cooldowns."""
        for i in range(len(self.keys)):
            # Prune TPM history
            hist_tpm = self.key_tpm_history[i]
            while hist_tpm and now - hist_tpm[0][0] > 60.0:
                hist_tpm.popleft()

            # Prune RPM history
            hist_rpm = self.key_rpm_history[i]
            while hist_rpm and now - hist_rpm[0] > 60.0:
                hist_rpm.popleft()

        # Expire key cooldowns
        expired_keys = [k for k, exp_ts in self.exhausted_keys.items() if now >= exp_ts]
        for k in expired_keys:
            del self.exhausted_keys[k]

        # Expire model daily exhaustion
        expired_models = [m for m, exp_ts in self.model_daily_exhaustion.items() if now >= exp_ts]
        for m in expired_models:
            del self.model_daily_exhaustion[m]

    def record_request_tokens(self, key_idx: int, tokens: int):
        """Records token and request counts for sliding-window RPM and TPM protection."""
        now = time.time()
        self._prune_sliding_windows(now)
        self.last_used_timestamps[key_idx] = now
        self.key_tpm_history[key_idx].append((now, tokens))
        self.key_rpm_history[key_idx].append(now)

    def get_key_60s_tokens(self, key_idx: int, now: Optional[float] = None) -> int:
        t_now = now or time.time()
        return sum(tok for ts, tok in self.key_tpm_history[key_idx] if t_now - ts <= 60.0)

    def get_key_60s_requests(self, key_idx: int, now: Optional[float] = None) -> int:
        t_now = now or time.time()
        return sum(1 for ts in self.key_rpm_history[key_idx] if t_now - ts <= 60.0)

    def is_key_exhausted(self, key_idx: int, model: Optional[str] = None) -> bool:
        now = time.time()
        m = model or self.get_model()
        key_tuple = (key_idx, m)
        if key_tuple in self.exhausted_keys:
            if now < self.exhausted_keys[key_tuple]:
                return True
            else:
                del self.exhausted_keys[key_tuple]
        return False

    def rotate_to_next_model(self, reason: str = "") -> str:
        """Advances to the next viable fallback model in the cascade and re-inits client."""
        old_model = self.get_model()
        valid_models = [
            m for m in self.models
            if m not in self.invalid_models and m not in self.model_daily_exhaustion
        ]
        if not valid_models:
            # If all models hit daily, clear daily exhaustion to allow minimum retry
            self.model_daily_exhaustion.clear()
            valid_models = [m for m in self.models if m not in self.invalid_models]

        if len(valid_models) > 1:
            curr_idx = valid_models.index(old_model) if old_model in valid_models else 0
            new_model = valid_models[(curr_idx + 1) % len(valid_models)]
            self.current_model_index = self.models.index(new_model)
            self.schedule_cursor = 0  # Reset key schedule to start cleanly from Account 0 Project 0
            logger.warning(f"🔄 Model auto-rotation: '{old_model}' ➔ '{new_model}' (Reason: {reason})")
            self._init_client()
            if self.db:
                asyncio.create_task(self.db.set_setting("model", new_model))
            return new_model
        return old_model

    def select_best_key(self, estimated_tokens: int = 35000, model: Optional[str] = None) -> int:
        """
        Preemptive Interleaved Key Selector:
        Validates both TPM headroom AND RPM headroom (5-15 RPM limits) before selecting key.
        """
        now = time.time()
        self._prune_sliding_windows(now)
        target_model = model or self.get_model()

        tpm_limit = getattr(config, "FREE_TIER_TPM_LIMIT", 250000)
        tpm_ceiling = int(tpm_limit * getattr(config, "PREEMPTIVE_TPM_SAFETY_MARGIN", 0.85))

        # Model-specific RPM cap (3.8-flash has 5 RPM; others have 15 RPM)
        base_rpm = 5 if "3.8" in target_model else getattr(config, "FREE_TIER_RPM_LIMIT", 15)
        rpm_ceiling = max(1, int(base_rpm * getattr(config, "PREEMPTIVE_TPM_SAFETY_MARGIN", 0.85)))

        # 1. Primary pass: key not in cooldown with both TPM and RPM headroom
        for step in range(len(self.interleaved_indices)):
            cand_idx = self.interleaved_indices[(self.schedule_cursor + step) % len(self.interleaved_indices)]

            if (cand_idx, target_model) in self.exhausted_keys:
                continue

            key_tokens = self.get_key_60s_tokens(cand_idx, now)
            if key_tokens + estimated_tokens > tpm_ceiling:
                continue

            key_reqs = self.get_key_60s_requests(cand_idx, now)
            if key_reqs >= rpm_ceiling:
                continue

            self.schedule_cursor = (self.schedule_cursor + step + 1) % len(self.interleaved_indices)
            self.current_key_index = cand_idx
            return cand_idx

        # 2. Secondary pass: any key not in cooldown for target_model with TPM headroom
        for step in range(len(self.interleaved_indices)):
            cand_idx = self.interleaved_indices[(self.schedule_cursor + step) % len(self.interleaved_indices)]
            if (cand_idx, target_model) not in self.exhausted_keys:
                key_tokens = self.get_key_60s_tokens(cand_idx, now)
                if key_tokens + estimated_tokens <= tpm_ceiling:
                    self.schedule_cursor = (self.schedule_cursor + step + 1) % len(self.interleaved_indices)
                    self.current_key_index = cand_idx
                    return cand_idx

        # 3. Tertiary pass: any key not in active cooldown
        for step in range(len(self.interleaved_indices)):
            cand_idx = self.interleaved_indices[(self.schedule_cursor + step) % len(self.interleaved_indices)]
            if (cand_idx, target_model) not in self.exhausted_keys:
                self.schedule_cursor = (self.schedule_cursor + step + 1) % len(self.interleaved_indices)
                self.current_key_index = cand_idx
                return cand_idx

        return -1  # All keys on cooldown for target_model

    def get_active_key(self) -> str:
        if not self.keys:
            raise ValueError("No GEMINI_API_KEYS found in configuration or secrets!")

        best_idx = self.select_best_key()
        if best_idx != -1:
            return self.keys[best_idx]

        curr_m = self.get_model()
        matching_cooldowns = {k[0]: ts for k, ts in self.exhausted_keys.items() if k[1] == curr_m}
        earliest_idx = min(matching_cooldowns.keys(), key=lambda k: matching_cooldowns[k], default=0)
        self.current_key_index = earliest_idx
        return self.keys[self.current_key_index]

    def get_model(self) -> str:
        now = time.time()
        self._prune_sliding_windows(now)
        valid_models = [
            m for m in self.models
            if m not in self.invalid_models and m not in self.model_daily_exhaustion
        ]
        if not valid_models:
            valid_models = [m for m in self.models if m not in self.invalid_models]
        if not valid_models:
            return "gemini-3.7-flash"
        return valid_models[self.current_model_index % len(valid_models)]

    def mark_model_invalid(self, model_name: str):
        logger.warning(f"Excluding model '{model_name}' from active rotation pool (404/Unsupported).")
        self.invalid_models.add(model_name)
        self.current_model_index = 0

    def reset_all_cooldowns(self) -> int:
        count = len(self.exhausted_keys)
        self.exhausted_keys.clear()
        self.model_daily_exhaustion.clear()
        self.invalid_models.clear()
        for i in range(len(self.keys)):
            self.key_tpm_history[i].clear()
            self.key_rpm_history[i].clear()
        self._init_client()
        logger.info(f"All {count} key-model cooldown locks cleared. Pool 100% active.")
        return len(self.keys)

    def get_pool_status(self) -> Dict[str, Any]:
        now = time.time()
        self._prune_sliding_windows(now)
        curr_m = self.get_model()
        active_count = sum(1 for idx in range(len(self.keys)) if (idx, curr_m) not in self.exhausted_keys)
        acc_id = self.get_account_for_key(self.current_key_index)
        proj_id = self.get_project_for_key(self.current_key_index)
        key_tpm = self.get_key_60s_tokens(self.current_key_index, now)
        key_rpm = self.get_key_60s_requests(self.current_key_index, now)

        return {
            "total_keys": len(self.keys),
            "num_accounts": self.num_accounts,
            "keys_per_account": self.keys_per_account,
            "active_ready_keys": active_count,
            "cooldown_keys": len(self.keys) - active_count,
            "current_key_index": self.current_key_index + 1,
            "current_account_index": acc_id + 1,
            "current_project_index": proj_id + 1,
            "current_key_60s_tokens": key_tpm,
            "current_key_60s_requests": key_rpm,
            "current_model": curr_m,
            "valid_models": [m for m in self.models if m not in self.invalid_models and m not in self.model_daily_exhaustion]
        }

    async def get_ready_client_for_request(
        self,
        estimated_tokens: int = 35000,
        event_callback=None,
        agent=None
    ) -> genai.Client:
        best_idx = self.select_best_key(estimated_tokens=estimated_tokens)
        if best_idx != -1:
            self._init_client()
            return self._client

        return await self.handle_quota_exhausted(
            "Preemptive 250k TPM / RPM limit protection",
            event_callback=event_callback,
            agent=agent,
            estimated_tokens=estimated_tokens
        )

    async def handle_quota_exhausted(
        self,
        error_msg_or_ex: Any,
        event_callback=None,
        agent=None,
        estimated_tokens: int = 35000
    ) -> genai.Client:
        """
        Handles 429 Quota Exhaustion with Multi-Tier Cascading Failover:
        1. Identifies exact cooldown & whether the limit is Daily (RPD) vs Transient (RPM/TPM).
        2. Locks ONLY (key_idx, current_model).
        3. If daily or multi-key 429 occurs, auto-falls back to next model with fresh quotas!
        4. Smooth pacing between key switches prevents IP-level rate limit storms.
        """
        err_str = str(error_msg_or_ex)
        err_lower = err_str.lower()

        # 1. Model Not Found (404)
        if any(w in err_lower for w in ["404", "not found", "unsupported model"]):
            bad_model = self.get_model()
            self.mark_model_invalid(bad_model)
            new_model = self.get_model()
            logger.info(f"Model unavailable ('{bad_model}') -> advancing to '{new_model}'.")
            self._init_client()
            return self._client

        # 2. Server Overload (500/502/503/504)
        if any(w in err_lower for w in ["503", "overloaded", "high demand", "unavailable", "500", "502", "504"]):
            valid_models = [m for m in self.models if m not in self.invalid_models]
            if len(valid_models) > 1:
                prev_model = self.get_model()
                self.current_model_index = (self.current_model_index + 1) % len(valid_models)
                logger.info(f"Server transient error on '{prev_model}' -> fallback to '{self.get_model()}'.")

            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            self._init_client()
            return self._client

        # 3. 429 Rate Limit (RPM / TPM / RPD Quota)
        now = time.time()
        cooldown_sec, is_daily = parse_gemini_error_cooldown(error_msg_or_ex)
        prev_key_idx = self.current_key_index
        curr_model = self.get_model()

        # Lock ONLY this key for this specific model
        self.exhausted_keys[(prev_key_idx, curr_model)] = now + cooldown_sec

        # Count keys exhausted for THIS model
        exhausted_for_model = sum(
            1 for (k_i, m_n), exp_t in self.exhausted_keys.items()
            if m_n == curr_model and exp_t > now
        )

        acc_id = self.get_account_for_key(prev_key_idx)
        proj_id = self.get_project_for_key(prev_key_idx)

        rate_type_str = "Daily RPD Cap" if is_daily else f"RPM/TPM ({cooldown_sec:.0f}s)"
        logger.warning(
            f"Key #{prev_key_idx+1}/{len(self.keys)} (Account #{acc_id+1}, Project #{proj_id+1}) "
            f"hit 429 [{rate_type_str}] on '{curr_model}'."
        )

        # CASCADING MODEL FAILOVER: If multiple keys hit 429 on this model, or if daily cap is hit
        model_switch_threshold = 3 if is_daily else min(6, len(self.keys))
        if exhausted_for_model >= model_switch_threshold:
            if is_daily:
                # Mark model daily exhausted until Pacific midnight
                self.model_daily_exhaustion[curr_model] = now + float(get_seconds_until_pacific_midnight())

            valid_models = [
                m for m in self.models
                if m not in self.invalid_models and m not in self.model_daily_exhaustion
            ]
            if len(valid_models) > 1:
                old_m = curr_model
                new_m = self.rotate_to_next_model(
                    reason=f"{exhausted_for_model} keys rate-limited on '{old_m}' ({rate_type_str})"
                )
                if new_m != old_m:
                    failover_msg = (
                        f"🔄 Model '{old_m}' quota saturated across projects ({exhausted_for_model} keys locked). "
                        f"Auto-falling back to next model '{new_m}' with fresh key quotas..."
                    )
                    logger.warning(failover_msg)
                    if event_callback:
                        await event_callback("info", failover_msg)
                    if agent and hasattr(agent, "update_heartbeat"):
                        agent.update_heartbeat("MODEL_FAILOVER", f"Switched: {old_m} -> {new_m}")

                    next_k_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=new_m)
                    if next_k_idx != -1:
                        self.current_key_index = next_k_idx
                        self._init_client()
                        return self._client

        # Try next ready key for the current model
        next_key_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=curr_model)
        if next_key_idx != -1:
            self.current_key_index = next_key_idx
            self._init_client()
            next_acc = self.get_account_for_key(next_key_idx)
            next_proj = self.get_project_for_key(next_key_idx)
            logger.info(
                f"⚡ Failover: Activated Key #{next_key_idx+1} "
                f"(Account #{next_acc+1}, Project #{next_proj+1}, Model: {self.get_model()})."
            )
            return self._client

        # If all keys exhausted for current model, force advance to next model
        valid_models = [
            m for m in self.models
            if m not in self.invalid_models and m not in self.model_daily_exhaustion
        ]
        if len(valid_models) > 1:
            old_m = curr_model
            new_m = self.rotate_to_next_model(reason=f"All keys exhausted for model '{old_m}'")
            if new_m != old_m:
                msg = f"🔄 All keys exhausted on '{old_m}'. Cascading to fallback model '{new_m}'..."
                if event_callback:
                    await event_callback("info", msg)
                next_k_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=new_m)
                if next_k_idx != -1:
                    self.current_key_index = next_k_idx
                    self._init_client()
                    return self._client

        # === 4. ALL KEYS AND MODELS IN COOLDOWN: MINIMAL RESILIENT ASYNC SLEEP ===
        relevant_cooldowns = [
            exp_t for (k_i, m_n), exp_t in self.exhausted_keys.items()
            if m_n == self.get_model()
        ]
        min_cooldown_until = min(relevant_cooldowns) if relevant_cooldowns else (now + 15.0)
        wait_needed = max(1.0, min_cooldown_until - time.time() + 0.5)

        wait_msg = (
            f"⏳ All keys temporarily rate-limited for '{self.get_model()}'. "
            f"Pausing pipeline for {wait_needed:.1f}s until quota resets. "
            f"Will resume automatically without dropping turns..."
        )
        logger.warning(wait_msg)

        if event_callback:
            try:
                await event_callback("info", wait_msg)
            except Exception:
                pass

        if agent and hasattr(agent, "update_heartbeat"):
            agent.update_heartbeat("QUOTA_WAIT", f"Sleeping {wait_needed:.1f}s for quota reset...")

        elapsed = 0.0
        while elapsed < wait_needed:
            chunk = min(1.0, wait_needed - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk
            if agent and hasattr(agent, "update_heartbeat"):
                agent.update_heartbeat("QUOTA_WAIT", f"Quota reset in progress ({elapsed:.0f}/{wait_needed:.0f}s)...")

        now_after = time.time()
        self._prune_sliding_windows(now_after)
        self.select_best_key(estimated_tokens=estimated_tokens)
        self._init_client()
        logger.info(f"✅ Cooldown expired. Key #{self.current_key_index+1} refreshed. Resuming pipeline immediately.")
        return self._client
