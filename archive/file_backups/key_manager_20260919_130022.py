# KARYON_PATCH_V33_1_KEY_MANAGER_PY_APPLIED
# KARYON_PATCH_V33_KEY_MANAGER_PY_APPLIED
# karyon_agent_runtime/key_manager.py
"""
===============================================================================
36-KEY DYNAMIC POOL MANAGER & 250K TPM 36-PROJECT CROSS-ROTATION ENGINE (v33.5)
Features 36 Independent Project Quota Scopes (6 Accounts x 6 Projects), Exact
Google API RPC Error & Daily Quota Parser, Immediate Model Cascading on Daily Caps,
IP-Burst / Throttling Detection with Cooldown Unlocking, Proxy Failover with
Geo-Quarantine Safeguards, and Strictly Flagship 3.x Model Cascade.
===============================================================================
"""

import os
import time
import json
import ast
import re
import asyncio
import datetime
import logging
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from google import genai
from google.genai import types
import karyon_agent_runtime.config as config
from karyon_agent_runtime.proxy_manager import proxy_rotator

logger = logging.getLogger("ProxyAgent.KeyManager")


def get_seconds_until_pacific_midnight() -> int:
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
            if any(w in t_url for w in ["quota", "errorinfo", "details", "quotafailure", "retryinfo"]):
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
                                if any(w in q_i.lower() for w in ["perday", "day", "daily"]):
                                    is_daily = True
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

            if any(w in q_id for w in ["perday", "per_day", "day-freetier", "generaterequestsperday"]):
                is_daily = True
            if any(w in q_metric for w in ["perday", "per_day", "day-freetier", "requestsperday"]):
                is_daily = True
            if q_val in ["20", "50"] and ("perday" in q_id or "day" in q_id or "requestsperday" in q_metric):
                is_daily = True

            rd = node.get("retryDelay") or node.get("retry_delay")
            if rd is not None:
                try:
                    if isinstance(rd, str) and rd.endswith("s"):
                        retry_delay_val = float(rd[:-1])
                    elif isinstance(rd, (int, float)):
                        retry_delay_val = float(rd)
                except Exception:
                    pass

            for v in node.values():
                if isinstance(v, (dict, list)):
                    _inspect_node(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _inspect_node(item)

    dict_match = re.search(r"({[\s\S]*})", ex_str)
    if dict_match:
        payload_str = dict_match.group(1)
        parsed_obj = None
        try:
            parsed_obj = json.loads(payload_str)
        except Exception:
            try:
                parsed_obj = ast.literal_eval(payload_str)
            except Exception:
                pass

        if isinstance(parsed_obj, dict):
            _inspect_node(parsed_obj)

    if retry_delay_val is None:
        delay_patterns = [
            r'[\'"]?retryDelay[\'"]?\s*[:=]\s*[\'"]?([0-9]+(?:\.[0-9]+)?)\s*s?[\'"]?',
            r'retry\s*(?:after|in)\s*([0-9]+(?:\.[0-9]+)?)\s*s(?:econds?)?',
            r'wait\s*([0-9]+(?:\.[0-9]+)?)\s*s(?:econds?)?',
            r'try\s*again\s*in\s*([0-9]+(?:\.[0-9]+)?)\s*s(?:econds?)?',
            r'quota\s*will\s*reset\s*in\s*([0-9]+(?:\.[0-9]+)?)\s*s(?:econds?)?'
        ]
        for pat in delay_patterns:
            m = re.search(pat, ex_str, re.IGNORECASE)
            if m:
                try:
                    retry_delay_val = float(m.group(1))
                    break
                except Exception:
                    pass

    if not is_daily:
        daily_indicators = [
            "generaterequestsperday",
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

    if quota_value is None:
        m_val = re.search(r'[\'"]?quotaValue[\'"]?\s*[:=]\s*[\'"]?(\d+)[\'"]?', ex_str)
        if m_val:
            try:
                quota_value = int(m_val.group(1))
            except Exception:
                pass

    if not target_model:
        m_model = re.search(r'[\'"]?model[\'"]?\s*[:=]\s*[\'"]?([a-zA-Z0-9\.\-_]+)[\'"]?', ex_str)
        if m_model:
            target_model = m_model.group(1)

    quota_meta = {
        "quota_metric": quota_metric,
        "quota_value": quota_value,
        "quota_id": quota_id,
        "target_model": target_model,
        "retry_delay_sec": retry_delay_val
    }

    if is_daily:
        cooldown = float(get_seconds_until_pacific_midnight())
        logger.warning(f"Daily quota reached ({quota_id or 'RPD'}). Key cooldown set to midnight reset ({cooldown:.0f}s).")
        return cooldown, True, quota_meta

    if retry_delay_val is not None and retry_delay_val > 0:
        cooldown = float(retry_delay_val) + 1.0
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
        self.daily_request_counts: Dict[Tuple[int, str], int] = {}
        self.daily_request_day: int = datetime.datetime.now(datetime.timezone.utc).day

        self.consecutive_429_count = 0
        self.last_429_timestamp = 0.0
        self.consecutive_geo_blocked_count = 0
        self.discovered_model_limits: Dict[str, Dict[str, Any]] = {}

        self._client: Optional[genai.Client] = None
        self._init_client()

    def _load_keys(self):
        if hasattr(config, "GEMINI_API_KEYS") and config.GEMINI_API_KEYS:
            for k in config.GEMINI_API_KEYS:
                k_clean = str(k).strip()
                if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                    self.keys.append(k_clean)

        raw_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_KEYS") or os.getenv("GEMINI_API_KEY") or ""
        if raw_keys_env:
            for k in raw_keys_env.split(","):
                k_clean = k.strip()
                if k_clean and len(k_clean) >= 30 and k_clean not in self.keys:
                    self.keys.append(k_clean)

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
        if not self.keys:
            return
        active_key = self.keys[self.current_key_index % len(self.keys)]

        proxy_url = None
        try:
            proxy_url = proxy_rotator.get_healthy_proxy()
        except Exception as p_err:
            logger.debug(f"Proxy resolution notice: {p_err}")

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
        if self._client is None:
            self._init_client()
        return self._client

    def get_model(self) -> str:
        valid_models = [m for m in self.models if m not in self.invalid_models and m not in self.model_daily_exhaustion]
        if not valid_models:
            self.model_daily_exhaustion.clear()
            valid_models = [m for m in self.models if m not in self.invalid_models]
            if not valid_models:
                self.invalid_models.clear()
                valid_models = self.models

        self.current_model_index = self.current_model_index % len(valid_models)
        return valid_models[self.current_model_index]

    def mark_model_invalid(self, model_name: str):
        if model_name in self.models:
            self.invalid_models.add(model_name)
            logger.warning(f"Model '{model_name}' marked invalid (404). Active pool: {[m for m in self.models if m not in self.invalid_models]}")
            self.current_model_index = 0

    def rotate_to_next_model(self, reason: str = "Manual rotation") -> str:
        valid_models = [m for m in self.models if m not in self.invalid_models and m not in self.model_daily_exhaustion]
        if not valid_models:
            self.model_daily_exhaustion.clear()
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
        now = time.time()
        target_model = model or self.get_model()
        self._prune_sliding_windows(now)

        ready_keys = []
        for i in range(len(self.keys)):
            cooldown_until = self.exhausted_keys.get((i, target_model), 0.0)
            if now >= cooldown_until:
                discovered = self.discovered_model_limits.get(target_model, {})
                if discovered.get("is_daily") and discovered.get("quota_value"):
                    daily_cap = discovered["quota_value"]
                    if self.daily_request_counts.get((i, target_model), 0) >= daily_cap:
                        continue
                ready_keys.append(i)

        if not ready_keys:
            return -1

        best_key_idx = -1
        min_token_load = float("inf")

        tpm_limit = getattr(config, "FREE_TIER_TPM_LIMIT", 250000)
        rpm_limit = getattr(config, "FREE_TIER_RPM_LIMIT", 15)

        for idx in ready_keys:
            rolling_tokens = sum(tok for ts, tok in self.token_history[idx] if now - ts < 60.0)
            rolling_requests = sum(1 for ts in self.request_history[idx] if now - ts < 60.0)

            if rolling_tokens + estimated_tokens > tpm_limit:
                continue

            discovered = self.discovered_model_limits.get(target_model, {})
            discovered_val = discovered.get("quota_value")

            base_rpm = rpm_limit
            if "3.8" in target_model:
                base_rpm = min(5, rpm_limit)
            elif discovered_val and not discovered.get("is_daily") and discovered_val <= 60:
                base_rpm = min(discovered_val, rpm_limit)

            if rolling_requests >= base_rpm:
                continue

            if rolling_tokens < min_token_load:
                min_token_load = rolling_tokens
                best_key_idx = idx

        return best_key_idx

    def record_request_tokens(self, key_idx: int, tokens: int):
        now = time.time()
        self.consecutive_429_count = 0
        self.consecutive_geo_blocked_count = 0

        curr_day = datetime.datetime.now(datetime.timezone.utc).day
        if curr_day != self.daily_request_day:
            self.daily_request_counts.clear()
            self.daily_request_day = curr_day

        target_model = self.get_model()
        self.daily_request_counts[(key_idx, target_model)] = self.daily_request_counts.get((key_idx, target_model), 0) + 1

        if key_idx not in self.request_history:
            self.request_history[key_idx] = deque()
        if key_idx not in self.token_history:
            self.token_history[key_idx] = deque()
        self.request_history[key_idx].append(now)
        self.token_history[key_idx].append((now, tokens))

    def _prune_sliding_windows(self, now: float):
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
        return key_idx // 6

    def get_project_for_key(self, key_idx: int) -> int:
        return key_idx % 6

    def get_key_60s_tokens(self, key_idx: int, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        if key_idx in self.token_history:
            return sum(tok for ts, tok in self.token_history[key_idx] if now - ts < 60.0)
        return 0

    def get_pool_status(self) -> Dict[str, Any]:
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
            "discovered_limits": self.discovered_model_limits,
            "consecutive_429": self.consecutive_429_count,
            "has_healthy_proxy": proxy_rotator.has_any_healthy_proxy()
        }

    def get_key_status_summary(self) -> Dict[str, Any]:
        return self.get_pool_status()

    def reset_all_cooldowns(self) -> int:
        return self.force_reset_cooldowns()

    def force_reset_cooldowns(self) -> int:
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
        self.consecutive_geo_blocked_count = 0
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
            # Do not record tokens or reset consecutive_429_count preemptively.
            # Token consumption and success metrics must strictly be recorded upon a verified successful response.
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
            logger.error(f"Permanent Auth Error on Key #{bad_key_idx + 1}. Quarantining key.")
            for m in self.models:
                self.exhausted_keys[(bad_key_idx, m)] = now + 315360000.0
            next_k_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=self.get_model())
            if next_k_idx != -1:
                self.current_key_index = next_k_idx
            else:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            self._init_client()
            return self._client

        # 1.2 Geo-Blocking / Region Restriction ("location is not supported" / 403)
        if any(w in err_lower for w in ["user location is not supported", "location is not supported", "country is not supported", "403 forbidden"]):
            self.consecutive_geo_blocked_count += 1
            geo_msg = f"⚠️ Gemini API Region Notice ({self.consecutive_geo_blocked_count}/3): User location not supported."
            logger.warning(geo_msg)
            if event_callback:
                await event_callback("info", geo_msg)

            if proxy_rotator.has_any_healthy_proxy():
                current_p = proxy_rotator.get_healthy_proxy()
                proxy_rotator.mark_proxy_geo_blocked(current_p)
                next_p = proxy_rotator.get_healthy_proxy()
                if next_p:
                    logger.info(f"Switching to next healthy proxy: '{next_p}'")
                    self._init_client()
                    return self._client

            if self.consecutive_geo_blocked_count < 3:
                await asyncio.sleep(2.0)
                return self._client
            else:
                block_alert = "🔴 CRITICAL: Kaggle node IP region is blocked by Google API policy. Please configure a healthy SOCKS5/HTTP proxy in GEMINI_PROXIES."
                logger.critical(block_alert)
                if event_callback:
                    await event_callback("info", block_alert)

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

        # 3. 429 Rate Limit Parsing
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

        # === CRITICAL FIX A: DAILY QUOTA REACHED (RPD CAP) ===
        if is_daily:
            logger.warning(f"🛑 Daily quota cap reached for model '{curr_model}' on Key #{prev_key_idx + 1}.")
            self.exhausted_keys[(prev_key_idx, curr_model)] = now + float(get_seconds_until_pacific_midnight())
            self.daily_request_counts[(prev_key_idx, curr_model)] = 99999

            daily_exhausted_keys = sum(
                1 for (k_i, m_n), exp_t in self.exhausted_keys.items()
                if m_n == curr_model and exp_t > now + 300.0
            )

            # If 2 or more keys hit daily limit, cascade to next 3.x model immediately!
            if daily_exhausted_keys >= 2:
                self.model_daily_exhaustion[curr_model] = now + float(get_seconds_until_pacific_midnight())
                old_m = curr_model
                new_m = self.rotate_to_next_model(reason=f"Model '{old_m}' daily quota exhausted across keys")
                cascade_alert = f"🔄 Daily quota exhausted for '{old_m}'. Cascading to fresh model '{new_m}'..."
                logger.warning(cascade_alert)
                if event_callback:
                    await event_callback("info", cascade_alert)
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
                return self._client

        # === CRITICAL FIX B: SHORT-TERM RPM / IP-BURST THROTTLE (NON-DAILY) ===
        if (now - self.last_429_timestamp) < 45.0:
            self.consecutive_429_count += 1
        else:
            self.consecutive_429_count = 1
        self.last_429_timestamp = now

        self.exhausted_keys[(prev_key_idx, curr_model)] = now + cooldown_sec

        acc_id = self.get_account_for_key(prev_key_idx)
        proj_id = self.get_project_for_key(prev_key_idx)

        logger.warning(
            f"Key #{prev_key_idx + 1}/{len(self.keys)} (Account #{acc_id+1}, Project #{proj_id+1}) "
            f"hit 429 [RPM/Burst ({cooldown_sec:.1f}s)] on '{curr_model}'. "
            f"Server retryDelay: {quota_meta.get('retry_delay_sec')}s. Consecutive: {self.consecutive_429_count}."
        )

        # Adaptive allostatic pacing: brief pause to respect Kaggle IP rate limits
        pacing_delay = min(15.0, max(1.0, 1.2 * (1.35 ** min(self.consecutive_429_count, 5))))
        if quota_meta.get("retry_delay_sec"):
            pacing_delay = max(pacing_delay, min(30.0, float(quota_meta["retry_delay_sec"])))
        if event_callback and self.consecutive_429_count > 1:
            await event_callback("info", f"⏳ Allostatic pacing pause ({pacing_delay:.1f}s) on Key #{prev_key_idx + 1} ({curr_model})...")
        await asyncio.sleep(pacing_delay)

        if self.consecutive_429_count >= 3:
            logger.warning(f"🚨 IP-level rate throttle detected ({self.consecutive_429_count} keys 429 consecutively).")

            if proxy_rotator.has_any_healthy_proxy():
                current_p = proxy_rotator.get_healthy_proxy()
                proxy_rotator.mark_proxy_429(current_p, cooldown_sec=30.0)
                next_p = proxy_rotator.get_healthy_proxy()
                if next_p:
                    switch_p_msg = f"🔄 IP rate-limit hit. Rotating proxy to healthy node '{next_p}'..."
                    logger.info(switch_p_msg)
                    if event_callback:
                        await event_callback("info", switch_p_msg)
                    self.consecutive_429_count = 0
                    self._init_client()
                    return self._client

            exact_delay = quota_meta.get("retry_delay_sec")
            if exact_delay is not None and 0.0 < float(exact_delay) <= 60.0:
                sleep_wait = float(exact_delay) + 1.5
            else:
                sleep_wait = min(30.0, 5.0 * (1.5 ** min(self.consecutive_429_count - 3, 4)))

            wait_msg = (
                f"⏳ IP Throttled by Google API. Holding key #{prev_key_idx + 1} "
                f"and sleeping {sleep_wait:.1f}s until IP bucket refills..."
            )
            logger.info(wait_msg)
            if event_callback:
                await event_callback("info", wait_msg)
            if agent and hasattr(agent, "update_heartbeat"):
                agent.update_heartbeat("RATE_PACING", f"Waiting IP bucket refill {sleep_wait:.1f}s")

            await asyncio.sleep(sleep_wait)
            self.exhausted_keys.pop((prev_key_idx, curr_model), None)
            self.consecutive_429_count = 0
            self._init_client()
            return self._client

        next_key_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=curr_model)
        if next_key_idx != -1:
            self.current_key_index = next_key_idx
            self._init_client()
            return self._client

        old_m = curr_model
        new_m = self.rotate_to_next_model(reason=f"All ready keys rate-limited on '{old_m}'")
        if new_m != old_m:
            cascade_alert = f"🔄 Switching model '{old_m}' -> '{new_m}' to acquire fresh rate limits..."
            logger.info(cascade_alert)
            if event_callback:
                await event_callback("info", cascade_alert)
            next_k_idx = self.select_best_key(estimated_tokens=estimated_tokens, model=new_m)
            if next_k_idx != -1:
                self.current_key_index = next_k_idx
            self._init_client()
            return self._client

        wait_needed = 15.0
        wait_msg = f"⏳ All keys in cooldown for '{self.get_model()}'. Pausing {wait_needed:.0f}s..."
        logger.warning(wait_msg)
        if event_callback:
            await event_callback("info", wait_msg)
        await asyncio.sleep(wait_needed)

        self.force_reset_cooldowns()
        self._init_client()
        return self._client


GeminiKeyManager = KeyManager
key_manager = KeyManager()
