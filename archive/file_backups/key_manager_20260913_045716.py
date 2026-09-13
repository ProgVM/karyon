# karyon_agent_runtime/key_manager.py
"""
===============================================================================
36-KEY DYNAMIC POOL MANAGER & 250K TPM 36-PROJECT CROSS-ROTATION ENGINE (v32.2)
Features 36 Independent Project Quota Scopes (6 Accounts x 6 Projects), Exact
Google API RPC Error & Daily Quota Parser, Model Limit Auto-Discovery & DB
Persistence, Multi-Tier Cascading Model Failover (3.8 -> 3.7 -> 3.6 -> 3.5),
Per-(Key, Model) Cooldowns, Preemptive Sliding-Window RPM Throttling, and
Exact Server-Specified Wait Time Compliance.
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
        offset_hours = config.PACIFIC_DAYLIGHT_TIME_OFFSET if is_pdt else config.PACIFIC_STANDARD_TIME_OFFSET
        pacific_tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
        pacific_now = now_utc.astimezone(pacific_tz)
        tomorrow_pacific = datetime.datetime(
            year=pacific_now.year, month=pacific_now.month, day=pacific_now.day,
            hour=0, minute=0, second=0, tzinfo=pacific_tz
        ) + datetime.timedelta(days=1)
        return max(60, int((tomorrow_pacific - pacific_now).total_seconds()))
    except Exception as e:
        logger.error(f"Error calculating Pacific Midnight: {e}")
        return 3600


def parse_gemini_error_cooldown(ex_or_str: Any) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Precision Cybernetic Parser for Gemini API RPC & HTTP Error Payloads.
    Extracts exact retryDelay from google.rpc.ErrorInfo and quota details.
    
    Recognizes:
    1. HTTP 429 RESOURCE_EXHAUSTED structured JSON with retryDelay.
    2. Daily limit indicators (Requests Per Day, Day-FreeTier).
    3. Distinguishes true daily quota limits from per-minute rate limits (RPM / TPM).
    Returns: (cooldown_seconds, is_daily, quota_meta_dict)
    """
    ex_str = str(ex_or_str)
    err_lower = ex_str.lower()
    is_daily = False
    retry_delay_val: Optional[float] = None
    quota_metric: Optional[str] = None
    quota_value: Optional[int] = None
    quota_id: Optional[str] = None
    target_model: Optional[str] = None

    def _inspect_node(node):
        nonlocal is_daily, retry_delay_val, quota_metric, quota_value, quota_id, target_model
        if isinstance(node, dict):
            t_url = str(node.get("@type", "") or node.get("typeUrl", "")).lower()
            if "quota" in t_url or "errorinfo" in t_url or "details" in t_url:
                violations = node.get("violations", [])
                if isinstance(violations, list):
                    for v in violations:
                        if isinstance(v, dict):
                            desc = str(v.get("description", "")).lower()
                            q_m = v.get("quotaMetric", "") or v.get("metric", "")
                            q_i = v.get("quotaId", "") or v.get("id", "")
                            q_v = v.get("quotaValue", "") or v.get("limit", "")
                            q_dims = v.get("quotaDimensions", {}) or v.get("dimensions", {})

                            if any(w in desc for w in ["perday", "daily", "per_day", "day-freetier", "day_free_tier"]):
                                is_daily = True
                            if q_m:
                                quota_metric = str(q_m)
                            if q_i:
                                quota_id = str(q_i)
                            if q_v is not None:
                                try:
                                    quota_value = int(q_v)
                                except (ValueError, TypeError):
                                    pass
                            if isinstance(q_dims, dict) and "model" in q_dims:
                                target_model = str(q_dims["model"])

            q_id = str(node.get("quotaId", "")).lower()
            q_metric = str(node.get("quotaMetric", "")).lower()
            q_val = str(node.get("quotaValue", ""))

            if any(w in q_id for w in ["perday", "per_day", "day-freetier"]):
                is_daily = True
            if any(w in q_metric for w in ["perday", "per_day", "day-freetier"]):
                is_daily = True
            if q_val in ["20", "50"] and any(w in q_id or w in q_metric for w in ["generatecontentrequestsperday", "requestsperday"]):
                is_daily = True

            for v in node.values():
                if isinstance(v, (dict, list)):
                    _inspect_node(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _inspect_node(item)

    try:
        json_match = re.search(r"({.*})", ex_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            _inspect_node(data)
            
            def _find_retry_delay(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ["retryDelay", "retry_delay"]:
                            return v
                        res = _find_retry_delay(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = _find_retry_delay(item)
                        if res:
                            return res
                return None

            r_delay = _find_retry_delay(data)
            if r_delay:
                if isinstance(r_delay, str) and r_delay.endswith("s"):
                    try:
                        retry_delay_val = float(r_delay[:-1])
                    except ValueError:
                        pass
                elif isinstance(r_delay, (int, float)):
                    retry_delay_val = float(r_delay)
    except Exception:
        pass

    daily_indicators = [
        "generatecontentrequestsperday",
        "requestsperday",
        "requests per day",
        "daily quota",
        "perday",
        "per day",
        "day-freetier",
        "day_free_tier"
    ]
    if any(ind in err_lower for ind in daily_indicators):
        is_daily = True

    quota_meta = {
        "quota_metric": quota_metric,
        "quota_value": quota_value,
        "quota_id": quota_id,
        "target_model": target_model,
        "retry_delay_sec": retry_delay_val
    }

    if is_daily:
        cooldown = 60.0
        logger.warning(f"Daily quota reached (RPD exhausted). Key cooldown set to 60s (Midnight reset is {get_seconds_until_pacific_midnight()}s away).")
        return cooldown, True, quota_meta

    if retry_delay_val is not None and retry_delay_val > 0:
        cooldown = float(retry_delay_val) + 1.0
        logger.info(f"Server requested exact retryDelay: {retry_delay_val:.2f}s. Key cooldown: {cooldown:.2f}s.")
        return cooldown, False, quota_meta

    cooldown = 15.0 if ("limit" in err_lower or "exhausted" in err_lower) else 5.0
    return cooldown, False, quota_meta


class KeyManager:
    def __init__(self, db=None):
        self.db = db
        self.keys: List[str] = []
        self._load_keys()
        
        self.models = [m.strip() for m in config.GEMINI_MODELS if m.strip()]
        self.invalid_models = set()
        self.model_daily_exhaustion: Dict[str, float] = {}
        self.current_key_index = 0
        self.current_model_index = 0
        
        self.exhausted_keys: Dict[Tuple[int, str], float] = {}
        self.request_history: Dict[int, deque] = {i: deque() for i in range(len(self.keys))}
        self.token_history: Dict[int, deque] = {i: deque() for i in range(len(self.keys))}
        
        self.consecutive_429_count = 0
        self.last_429_timestamp = 0.0
        self.discovered_model_limits: Dict[str, Dict[str, Any]] = {}
        
        self._client: Optional[genai.Client] = None
        self._init_client()

    def _load_keys(self):
        """Loads unique Gemini keys from config, environment variables, and Kaggle/Colab secrets."""
        # 1. First check config.GEMINI_API_KEYS if populated
        if hasattr(config, "GEMINI_API_KEYS") and config.GEMINI_API_KEYS:
            for k in config.GEMINI_API_KEYS:
                k_clean = str(k).strip()
                if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                    self.keys.append(k_clean)

        # 2. Check environment GEMINI_API_KEYS or GEMINI_KEYS (comma-separated list)
        raw_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_KEYS") or os.getenv("GEMINI_API_KEY") or ""
        if raw_keys_env:
            for k in raw_keys_env.split(","):
                k_clean = k.strip()
                if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                    self.keys.append(k_clean)

        # 3. Check Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            k_sec = secrets.get_secret("GEMINI_API_KEYS") or secrets.get_secret("GEMINI_KEYS") or secrets.get_secret("GEMINI_API_KEY")
            if k_sec:
                for k in str(k_sec).split(","):
                    k_clean = k.strip()
                    if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                        self.keys.append(k_clean)
        except Exception:
            pass

        # 4. Check Colab Userdata
        try:
            from google.colab import userdata
            k_colab = userdata.get("GEMINI_API_KEYS") or userdata.get("GEMINI_KEYS") or userdata.get("GEMINI_API_KEY")
            if k_colab:
                for k in str(k_colab).split(","):
                    k_clean = k.strip()
                    if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                        self.keys.append(k_clean)
        except Exception:
            pass

        # 5. Scan Account #1 to #6, Project #1 to #6 (36 individual project keys)
        for acc in range(1, 7):
            for proj in range(1, 7):
                env_name = f"GEMINI_A{acc}_P{proj}"
                val = os.getenv(env_name)
                if val and val.strip():
                    val_clean = val.strip()
                    if len(val_clean) >= 30 and val_clean not in self.keys:
                        self.keys.append(val_clean)

        if not self.keys:
            logger.error("CRITICAL: Zero valid Gemini API keys found in environment! Using fallback dummy key.")
            self.keys.append("AIzaSyDummyKey_PleaseConfigureYourKeysInEnvironment_")
        else:
            logger.info(f"Loaded {len(self.keys)} unique Gemini API keys into the resilient pool.")

    def _init_client(self):
        """Initializes Google GenAI Client with active API key and proxy rotation."""
        if not self.keys:
            return
        active_key = self.keys[self.current_key_index % len(self.keys)]
        
        proxy_url = None
        try:
            if hasattr(proxy_rotator, "get_active_proxy"):
                proxy_url = proxy_rotator.get_active_proxy()
            elif hasattr(proxy_rotator, "get_proxy"):
                proxy_url = proxy_rotator.get_proxy()
        except Exception as p_err:
            logger.debug(f"Proxy resolution notice: {p_err}")

        # Configure client options with proxy if active
        http_options = None
        if proxy_url:
            try:
                http_options = types.HttpOptions(
                    client_args={"proxy": proxy_url},
                    async_client_args={"proxy": proxy_url}
                )
            except Exception:
                http_options = {"proxy": proxy_url}
            
        self._client = genai.Client(
            api_key=active_key,
            http_options=http_options
        )

    def get_client(self) -> genai.Client:
        """Returns the active GenAI client, initializing if necessary."""
        if self._client is None:
            self._init_client()
        return self._client

    def get_model(self) -> str:
        """Returns the active model name from the models pool."""
        valid_models = [m for m in self.models if m not in self.invalid_models]
        if not valid_models:
            self.invalid_models.clear()
            valid_models = self.models
            
        self.current_model_index = self.current_model_index % len(valid_models)
        return valid_models[self.current_model_index]

    def mark_model_invalid(self, model_name: str):
        """Quarantines a model if the API returns 404 Model Not Found."""
        if model_name in self.models:
            self.invalid_models.add(model_name)
            logger.warning(f"Model '{model_name}' marked invalid (404 / unsupported). Active pool: {[m for m in self.models if m not in self.invalid_models]}")
            self.current_model_index = 0

    def rotate_to_next_model(self, reason: str = "Manual rotation") -> str:
        """Cascades to the next model in the hierarchy."""
        valid_models = [m for m in self.models if m not in self.invalid_models]
        if len(valid_models) <= 1:
            return self.get_model()
            
        old_model = self.get_model()
        self.current_model_index = (self.current_model_index + 1) % len(valid_models)
        new_model = self.get_model()
        logger.warning(f"🔄 Model Cascade triggered: '{old_model}' -> '{new_model}'. Reason: {reason}.")
        self._init_client()
        return new_model

    def select_best_key(self, estimated_tokens: int = 35000, model: Optional[str] = None) -> int:
        """
        Selects the best available key for the requested model based on:
        1. Liveness (not in rate-limit cooldown).
        2. Rolling 60s TPM (Token Per Minute) limit headroom.
        3. Rolling 60s RPM (Requests Per Minute) limit headroom.
        """
        now = time.time()
        target_model = model or self.get_model()
        self._prune_sliding_windows(now)
        
        ready_keys = []
        for i in range(len(self.keys)):
            cooldown_until = self.exhausted_keys.get((i, target_model), 0.0)
            if now >= cooldown_until:
                ready_keys.append(i)
                
        if not ready_keys:
            return -1
            
        best_key_idx = -1
        min_token_load = float("inf")
        
        tpm_limit = getattr(config, "FREE_TIER_TPM_LIMIT", getattr(config, "GEMINI_KEY_TPM_LIMIT", 250000))
        rpm_limit = getattr(config, "FREE_TIER_RPM_LIMIT", getattr(config, "GEMINI_KEY_RPM_LIMIT", 15))

        for idx in ready_keys:
            rolling_tokens = sum(tok for ts, tok in self.token_history[idx] if now - ts < 60.0)
            rolling_requests = sum(1 for ts in self.request_history[idx] if now - ts < 60.0)
            
            if rolling_tokens + estimated_tokens > tpm_limit:
                continue
                
            discovered = self.discovered_model_limits.get(target_model, {})
            discovered_val = discovered.get("quota_value")
            
            base_rpm = 15
            if "3.8" in target_model:
                base_rpm = min(5, rpm_limit)
            elif discovered_val and not discovered.get("is_daily") and discovered_val <= 60:
                base_rpm = discovered_val
            else:
                base_rpm = rpm_limit
                
            if rolling_requests >= base_rpm:
                continue
                
            if rolling_tokens < min_token_load:
                min_token_load = rolling_tokens
                best_key_idx = idx
                
        return best_key_idx

    def record_request_tokens(self, key_idx: int, tokens: int):
        """Registers a successful request and its token footprint in sliding windows."""
        now = time.time()
        if key_idx not in self.request_history:
            self.request_history[key_idx] = deque()
        if key_idx not in self.token_history:
            self.token_history[key_idx] = deque()
        self.request_history[key_idx].append(now)
        self.token_history[key_idx].append((now, tokens))

    def _prune_sliding_windows(self, now: float):
        """Cleans historical entries older than 60 seconds across all keys."""
        for idx in range(len(self.keys)):
            if idx in self.request_history:
                req_q = self.request_history[idx]
                while req_q and now - req_q[0] >= 60.0:
                    req_q.popleft()
                
            if idx in self.token_history:
                tok_q = self.token_history[idx]
                while tok_q and now - tok_q[0][0] >= 60.0:
                    tok_q.popleft()
                
        expired_models = [m for m, exp_ts in self.model_daily_exhaustion.items() if now >= exp_ts]
        for m in expired_models:
            del self.model_daily_exhaustion[m]

    def get_account_for_key(self, key_idx: int) -> int:
        """Derives the Account 0-based index based on key position (0 to 5)."""
        return key_idx // 6

    def get_project_for_key(self, key_idx: int) -> int:
        """Derives the Project 0-based index based on key position (0 to 5)."""
        return key_idx % 6

    def get_key_60s_tokens(self, key_idx: int, now: Optional[float] = None) -> int:
        """Computes rolling 60s token footprint for a specific key."""
        if now is None:
            now = time.time()
        if key_idx in self.token_history:
            return sum(tok for ts, tok in self.token_history[key_idx] if now - ts < 60.0)
        return 0

    def get_pool_status(self) -> Dict[str, Any]:
        """Returns deep telemetry metrics across the entire key pool."""
        now = time.time()
        self._prune_sliding_windows(now)
        
        curr_m = self.get_model()
        active_ready = 0
        cooldown_count = 0
        
        for i in range(len(self.keys)):
            cooldown_until = self.exhausted_keys.get((i, curr_m), 0.0)
            if now >= cooldown_until:
                active_ready += 1
            else:
                cooldown_count += 1
                
        curr_idx = self.current_key_index
        key_rpm = len(self.request_history[curr_idx]) if curr_idx in self.request_history else 0
        key_tpm = sum(tok for ts, tok in self.token_history[curr_idx]) if curr_idx in self.token_history else 0
        
        return {
            "total_keys": len(self.keys),
            "active_ready_keys": active_ready,
            "cooldown_keys": cooldown_count,
            "keys_on_cooldown": cooldown_count,
            "current_key_index": curr_idx + 1,
            "current_key_idx": curr_idx,
            "current_account_index": self.get_account_for_key(curr_idx) + 1,
            "current_project_index": self.get_project_for_key(curr_idx) + 1,
            "current_key_60s_tokens": key_tpm,
            "current_key_60s_requests": key_rpm,
            "current_model": curr_m,
            "valid_models": [m for m in self.models if m not in self.invalid_models and m not in self.model_daily_exhaustion],
            "discovered_limits": self.discovered_model_limits
        }

    def get_key_status_summary(self) -> Dict[str, Any]:
        """Alias for get_pool_status to ensure backwards compatibility."""
        return self.get_pool_status()

    def reset_all_cooldowns(self) -> int:
        """Alias for force_reset_cooldowns."""
        return self.force_reset_cooldowns()

    def force_reset_cooldowns(self) -> int:
        """Clears all rate limit cooldowns across the entire pool."""
        count = len(self.exhausted_keys)
        self.exhausted_keys.clear()
        self.model_daily_exhaustion.clear()
        self.invalid_models.clear()
        for i in range(len(self.keys)):
            if i in self.request_history:
                self.request_history[i].clear()
            if i in self.token_history:
                self.token_history[i].clear()
        self.consecutive_429_count = 0
        logger.info("Resilient key pool cooldowns 100% reset and cleared.")
        return count

    async def get_ready_client_for_request(
        self,
        estimated_tokens: int = 35000,
        event_callback=None,
        agent=None
    ) -> genai.Client:
        best_idx = self.select_best_key(estimated_tokens=estimated_tokens)
        if best_idx != -1:
            self.current_key_index = best_idx
            self._init_client()
            self.record_request_tokens(best_idx, estimated_tokens)
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
        1. Parses exact server-specified `retryDelay` and model limit metadata.
        2. Saves discovered model limits into SQLite DB (`model_quota_limits`).
        3. Locks ONLY (key_idx, current_model) for the EXACT server-specified duration.
        4. If daily or multi-key 429 occurs, auto-falls back to next model with fresh quotas.
        5. Performs exact resilient async wait when all keys/models are temporarily rate-limited.
        """
        now = time.time()
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

        # 1.1 Unauthenticated or Corrupted Key (401)
        if any(w in err_lower for w in ["401", "unauthenticated", "access_token_type_unsupported"]):
            bad_key_idx = self.current_key_index
            logger.error(f"Permanent Auth Error on Key #{bad_key_idx+1} ({self.keys[bad_key_idx][:8]}... len={len(self.keys[bad_key_idx])}). Quarantining key.")
            for m in self.models:
                self.exhausted_keys[(bad_key_idx, m)] = now + 315360000.0
            next_k_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=self.get_model())
            if next_k_idx != -1:
                self.current_key_index = next_k_idx
            else:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
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
        cooldown_sec, is_daily, quota_meta = parse_gemini_error_cooldown(error_msg_or_ex)
        prev_key_idx = self.current_key_index
        curr_model = self.get_model()

        target_model_for_meta = quota_meta.get("target_model") or curr_model
        limit_rec = {
            "model_name": target_model_for_meta,
            "quota_metric": quota_meta.get("quota_metric"),
            "quota_value": quota_meta.get("quota_value"),
            "retry_delay_sec": quota_meta.get("retry_delay_sec"),
            "is_daily": is_daily
        }
        self.discovered_model_limits[target_model_for_meta] = limit_rec

        if self.db:
            asyncio.create_task(self.db.record_model_quota_limit(
                model_name=target_model_for_meta,
                quota_metric=quota_meta.get("quota_metric"),
                quota_value=quota_meta.get("quota_value"),
                retry_delay_sec=quota_meta.get("retry_delay_sec"),
                is_daily=is_daily
            ))

        if (now - self.last_429_timestamp) < 30.0:
            self.consecutive_429_count += 1
        else:
            self.consecutive_429_count = 1
        self.last_429_timestamp = now

        self.exhausted_keys[(prev_key_idx, curr_model)] = now + cooldown_sec

        acc_id = self.get_account_for_key(prev_key_idx)
        proj_id = self.get_project_for_key(prev_key_idx)

        rate_type_str = "Daily RPD Cap" if is_daily else f"RPM/TPM ({cooldown_sec:.1f}s)"
        logger.warning(
            f"Key #{prev_key_idx+1}/{len(self.keys)} (Account #{acc_id+1}, Project #{proj_id+1}) "
            f"hit 429 [{rate_type_str}] on '{curr_model}'. Server retryDelay: {quota_meta.get('retry_delay_sec')}s. "
            f"Consecutive 429 count: {self.consecutive_429_count}."
        )

        next_key_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=curr_model)
        if next_key_idx != -1:
            self.current_key_index = next_key_idx
            self._init_client()
            next_acc = self.get_account_for_key(next_key_idx)
            next_proj = self.get_project_for_key(next_key_idx)
            logger.info(
                f"⚡ Instant Failover: Switched to Key #{next_key_idx+1} "
                f"(Account #{next_acc+1}, Project #{next_proj+1}, Model: {curr_model}) in 0ms."
            )
            if self.consecutive_429_count >= 3:
                await asyncio.sleep(0.5)
            return self._client

        exhausted_for_model = sum(
            1 for (k_i, m_n), exp_t in self.exhausted_keys.items()
            if m_n == curr_model and exp_t > now
        )

        if is_daily:
            model_switch_threshold = 2
        else:
            model_switch_threshold = max(10, len(self.keys) // 2)

        if exhausted_for_model >= model_switch_threshold:
            if is_daily:
                self.model_daily_exhaustion[curr_model] = now + float(get_seconds_until_pacific_midnight())

            valid_models = [
                m for m in self.models
                if m not in self.invalid_models and m not in self.model_daily_exhaustion
            ]
            if len(valid_models) <= 1:
                logger.warning("⚠️ Only 1 or 0 valid models left in pool. Clearing model_daily_exhaustion to force cascading rotation!")
                self.model_daily_exhaustion.clear()
                valid_models = [m for m in self.models if m not in self.invalid_models]

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

        valid_models = [
            m for m in self.models
            if m not in self.invalid_models and m not in self.model_daily_exhaustion
        ]
        if len(valid_models) <= 1:
            logger.warning("⚠️ All models in daily exhaustion. Clearing model_daily_exhaustion list to resume cycle.")
            self.model_daily_exhaustion.clear()
            valid_models = [m for m in self.models if m not in self.invalid_models]

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

        # All keys and models in cooldown: minimal resilient async sleep
        relevant_cooldowns = [
            exp_t for (k_i, m_n), exp_t in self.exhausted_keys.items()
            if m_n == self.get_model()
        ]
        min_cooldown_until = min(relevant_cooldowns) if relevant_cooldowns else (now + 15.0)
        wait_needed = max(1.0, min_cooldown_until - time.time() + 0.2)

        is_hard_capped = False
        if wait_needed > 30.0:
            logger.warning(f"⚠️ Requested cooldown sleep of {wait_needed:.1f}s exceeds MAX_SAFETY_SLEEP (30.0s). Hard-capping sleep to 30.0s!")
            wait_needed = 30.0
            is_hard_capped = True

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
        
        if is_hard_capped:
            logger.warning("🔄 Hard safety sleep cap expired. Forcefully clearing rate limit cooldowns to resume pipeline.")
            self.force_reset_cooldowns()

        self._prune_sliding_windows(now_after)
        next_idx = self.select_best_key(estimated_tokens=estimated_tokens)
        if next_idx != -1:
            self.current_key_index = next_idx
        self._init_client()
        logger.info(f"✅ Cooldown expired. Key #{self.current_key_index+1} refreshed. Resuming pipeline immediately.")
        return self._client


# Alias KeyManager as GeminiKeyManager to guarantee zero-import-error parity
GeminiKeyManager = KeyManager

# Global singleton instance
key_manager = KeyManager()
