# karyon_agent_runtime/main.py
"""
===============================================================================
KAGGLE CORE AGENT INTERACTIVE REPL & ADVANCED IPYWIDGETS RUNTIME PANEL (v31.0)
Features Continuous Autonomous Research Daemon Control (Unattended 24/7 Loop),
Bounded IPyWidgets Buffer, Mobile Sticky Scroll, Persistent SQLite Settings Sync,
Full Model Cascade (Gemini 3.8 -> 3.7 -> 3.5 -> 2.5 -> 2.0) and Swarm Telemetry.
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

# KARYON_PATCH_V31_APPLIED

import sys
import os
import re
import io
import time
import json
import base64
import traceback
from pathlib import Path

# 1. Force purge any cached agent modules from Jupyter RAM
for k in list(sys.modules.keys()):
    if 'karyon_agent_runtime' in k or k in [
        'agent_core', 'config', 'key_manager', 'db_manager',
        'context_compactor', 'prompt_builder', 'tools', 'process_engine'
    ]:
        del sys.modules[k]

# 2. Add project root and agent root to sys.path safely
_file_path = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / "karyon_agent_runtime" / "main.py"
_agent_dir = _file_path.parent
_project_root = _agent_dir.parent

for p in [str(_project_root), str(_agent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import asyncio
from IPython.display import display, Markdown, HTML

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from karyon_agent_runtime.agent_core import KaggleCoREAgent
    import karyon_agent_runtime.config as config
    from karyon_agent_runtime.tools.context_tools import get_context_token_status, compress_context_now
except Exception as imp_err:
    print("❌ Critical import error inside agent_core:")
    traceback.print_exc()
    raise imp_err

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

# =============================================================================
# SCROLL & DOM CONTROLLER (MOBILE RESPONSIVE & STICKY AUTO-SCROLL ENGINE)
# =============================================================================

SCROLL_GUARD_HTML = """
<style>
.karyon-chat-output-container {
    width: 100% !important;
    height: 520px !important;
    max-height: 520px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scroll-behavior: auto !important;
    overscroll-behavior-y: contain !important;
    overflow-anchor: none !important;
    -webkit-overflow-scrolling: touch !important;
    touch-action: pan-y !important;
    contain: paint layout !important;
    will-change: scroll-position !important;
    position: relative !important;
    background-color: var(--jp-layout-color1, #ffffff) !important;
    color: var(--jp-content-font-color1, #0f172a) !important;
    border: 1px solid var(--jp-border-color1, #cbd5e1) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    box-sizing: border-box !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.karyon-scroll-bottom-pill {
    position: absolute;
    bottom: 18px;
    right: 24px;
    background: #0284c7;
    color: #ffffff;
    font-size: 11.5px;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    cursor: pointer;
    z-index: 9999;
    opacity: 0;
    transform: translateY(12px);
    transition: opacity 0.25s ease, transform 0.25s ease;
    pointer-events: none;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 5px;
}
.karyon-scroll-bottom-pill.visible {
    opacity: 0.95;
    transform: translateY(0);
    pointer-events: auto;
}
.karyon-scroll-bottom-pill:active {
    transform: scale(0.95);
}

.karyon-chat-output-container p,
.karyon-chat-output-container li,
.karyon-chat-output-container span,
.karyon-chat-output-container div {
    color: inherit;
}

.karyon-chat-output-container h1,
.karyon-chat-output-container h2,
.karyon-chat-output-container h3,
.karyon-chat-output-container h4 {
    color: var(--jp-content-font-color0, #0f172a) !important;
    margin-top: 14px;
    margin-bottom: 8px;
    font-weight: 700;
}

.karyon-chat-output-container pre {
    background-color: var(--jp-layout-color2, #f8fafc) !important;
    color: var(--jp-content-font-color0, #0f172a) !important;
    border: 1px solid var(--jp-border-color2, #e2e8f0) !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    overflow-x: auto !important;
    font-family: "JetBrains Mono", Consolas, "Courier New", monospace !important;
    font-size: 0.88em !important;
    line-height: 1.5 !important;
}

.karyon-chat-output-container code {
    background-color: var(--jp-layout-color2, #f1f5f9) !important;
    color: #0369a1 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: "JetBrains Mono", Consolas, "Courier New", monospace !important;
    font-size: 0.88em !important;
}

.karyon-chat-output-container table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 14px 0 !important;
    font-size: 0.9em !important;
    border: 1px solid var(--jp-border-color1, #cbd5e1) !important;
    background-color: var(--jp-layout-color1, #ffffff) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

.karyon-chat-output-container th {
    background-color: var(--jp-layout-color2, #f1f5f9) !important;
    color: var(--jp-content-font-color0, #0f172a) !important;
    font-weight: 700 !important;
    padding: 9px 11px !important;
    border: 1px solid var(--jp-border-color1, #cbd5e1) !important;
    text-align: left !important;
}

.karyon-chat-output-container td {
    padding: 8px 11px !important;
    border: 1px solid var(--jp-border-color1, #cbd5e1) !important;
    color: var(--jp-content-font-color1, #1e293b) !important;
}

.karyon-chat-output-container tr:nth-child(even) {
    background-color: var(--jp-layout-color2, #f8fafc) !important;
}

.karyon-chat-output-container hr {
    border: 0 !important;
    height: 1px !important;
    background: var(--jp-border-color1, #e2e8f0) !important;
    margin: 16px 0 !important;
}

.karyon-btn-row,
.widget-hbox.karyon-btn-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin: 6px 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.karyon-btn-row button,
.karyon-btn-row .widget-button,
.widget-button {
    flex: 1 1 auto !important;
    min-width: 105px !important;
    max-width: 160px !important;
    white-space: nowrap !important;
    text-overflow: clip !important;
    overflow: visible !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    box-sizing: border-box !important;
    border-radius: 6px !important;
}

.karyon-chat-output-container::-webkit-scrollbar { width: 7px; }
.karyon-chat-output-container::-webkit-scrollbar-track { background: var(--jp-layout-color2, #f1f5f9); }
.karyon-chat-output-container::-webkit-scrollbar-thumb { background: var(--jp-border-color1, #94a3b8); border-radius: 4px; }
.karyon-chat-output-container::-webkit-scrollbar-thumb:hover { background: #64748b; }
</style>

<script>
(function() {
    function initContainer(el) {
        if (!el || el._karyon_engine_active) return;
        el._karyon_engine_active = true;
        el._karyon_pinned = true;
        el._karyon_touching = false;
        el._karyon_last_touch = 0;
        el._karyon_raf_id = null;
        el._karyon_user_scrolling = false;

        var pill = document.createElement('div');
        pill.className = 'karyon-scroll-bottom-pill';
        pill.innerHTML = '⬇️ Bottom';
        pill.title = 'Click to jump to latest message';

        var parent = el.parentElement;
        if (parent && getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }
        (parent || el).appendChild(pill);

        pill.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            el._karyon_pinned = true;
            pill.classList.remove('visible');
            el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        });

        function updatePill() {
            if (!el._karyon_pinned) {
                pill.classList.add('visible');
            } else {
                pill.classList.remove('visible');
            }
        }

        el.addEventListener('touchstart', function() {
            el._karyon_touching = true;
            el._karyon_user_scrolling = true;
            el._karyon_last_touch = Date.now();
        }, { passive: true });

        el.addEventListener('touchend', function() {
            el._karyon_touching = false;
            el._karyon_last_touch = Date.now();
        }, { passive: true });

        el.addEventListener('wheel', function() {
            el._karyon_user_scrolling = true;
            el._karyon_last_touch = Date.now();
        }, { passive: true });

        el.addEventListener('scroll', function() {
            var distToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
            if (distToBottom <= 40) {
                el._karyon_pinned = true;
                el._karyon_user_scrolling = false;
                updatePill();
            } else if (distToBottom > 80 && el._karyon_user_scrolling) {
                el._karyon_pinned = false;
                updatePill();
            }
        }, { passive: true });

        function scheduleScrollBottom(force) {
            if (!force && (!el._karyon_pinned || el._karyon_touching)) return;
            if (!force && (Date.now() - el._karyon_last_touch < 350)) return;

            if (el._karyon_raf_id) cancelAnimationFrame(el._karyon_raf_id);

            el._karyon_raf_id = requestAnimationFrame(function() {
                el._karyon_raf_id = null;
                var target = el.scrollHeight - el.clientHeight;
                if (target > 0) {
                    el.scrollTop = target;
                }
            });
        }

        var observer = new MutationObserver(function() { scheduleScrollBottom(); });
        observer.observe(el, { childList: true, subtree: true });

        if (window.ResizeObserver) {
            var resizeObs = new ResizeObserver(function() { scheduleScrollBottom(); });
            resizeObs.observe(el);
        }

        scheduleScrollBottom();
    }

    function initSendGuard() {
        var sendBtn = document.querySelector('.karyon-send-btn button, .karyon-send-btn');
        var textarea = document.querySelector('.karyon-text-input textarea, .karyon-text-input');
        if (!sendBtn || !textarea || sendBtn._karyon_send_guarded) return;
        sendBtn._karyon_send_guarded = true;

        textarea.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                sendBtn.click();
            }
        });

        sendBtn.addEventListener('click', function(e) {
            var isKernelBusy = false;
            if (document.body.classList.contains('jp-mod-busy') ||
                document.querySelector('.jp-mod-busy') ||
                document.querySelector('.jp-Toolbar-kernelStatus[title*="Busy"]') ||
                document.querySelector('[data-status="busy"]') ||
                (window.Jupyter && window.Jupyter.notebook && window.Jupyter.notebook.kernel && !window.Jupyter.notebook.kernel.is_idle())) {
                isKernelBusy = true;
            }

            textarea.dispatchEvent(new Event('input', { bubbles: true }));
            textarea.dispatchEvent(new Event('change', { bubbles: true }));

            if (isKernelBusy) {
                e.stopImmediatePropagation();
                e.preventDefault();

                var warnEl = document.getElementById('karyon-busy-warning');
                if (!warnEl) {
                    warnEl = document.createElement('div');
                    warnEl.id = 'karyon-busy-warning';
                    warnEl.style.cssText = 'position:fixed; bottom:24px; left:50%; transform:translateX(-50%); background:#dc2626; color:#ffffff; font-size:12.5px; font-weight:700; padding:10px 18px; border-radius:8px; z-index:999999; box-shadow:0 6px 20px rgba(0,0,0,0.5); text-align:center; transition:opacity 0.25s ease; pointer-events:none;';
                    document.body.appendChild(warnEl);
                }
                warnEl.innerText = '⚠️ Kaggle Kernel is currently busy executing another cell (e.g. %run). Message NOT sent to prevent desync. Please wait for that cell to finish!';
                warnEl.style.display = 'block';
                warnEl.style.opacity = '1';
                setTimeout(function() {
                    warnEl.style.opacity = '0';
                    setTimeout(function() { warnEl.style.display = 'none'; }, 300);
                }, 4500);

                return false;
            }

            var currentVal = textarea.value.trim();
            if (!currentVal) {
                e.stopImmediatePropagation();
                e.preventDefault();
                return false;
            }

            sendBtn.style.opacity = '0.6';
            sendBtn.style.pointerEvents = 'none';
            setTimeout(function() {
                sendBtn.style.opacity = '1.0';
                sendBtn.style.pointerEvents = 'auto';
            }, 1500);
        }, true);
    }

    function scanAll() {
        var elements = document.querySelectorAll('.karyon-chat-output-container');
        elements.forEach(function(el) { initContainer(el); });
        initSendGuard();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scanAll);
    } else {
        scanAll();
    }

    setInterval(scanAll, 1000);

    window.karyonForceScrollBottom = function(smooth) {
        var elements = document.querySelectorAll('.karyon-chat-output-container');
        elements.forEach(function(el) {
            el._karyon_pinned = true;
            el._karyon_user_scrolling = false;
            el._karyon_touching = false;
            var target = el.scrollHeight - el.clientHeight;
            if (target > 0) {
                if (smooth) el.scrollTo({ top: target, behavior: 'smooth' });
                else el.scrollTop = target;
            }
        });
    };
})();
</script>
"""


def optimize_image_for_display(path: Path, max_dim: int = 850, quality: int = 82) -> tuple:
    """Reads and resizes image to lightweight compressed Base64 to prevent mobile crashes."""
    if not HAS_PIL:
        ext = path.suffix.lower().replace(".", "")
        mime = f"image/{'svg+xml' if ext == 'svg' else ext or 'png'}"
        with open(path, "rb") as f:
            return mime, base64.b64encode(f.read()).decode("utf-8")
    try:
        with Image.open(path) as img:
            if getattr(img, "is_animated", False):
                with open(path, "rb") as f:
                    return "image/gif", base64.b64encode(f.read()).decode("utf-8")

            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                mime = "image/jpeg"

            return mime, base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        ext = path.suffix.lower().replace(".", "")
        mime = f"image/{'svg+xml' if ext == 'svg' else ext or 'png'}"
        with open(path, "rb") as f:
            return mime, base64.b64encode(f.read()).decode("utf-8")


def render_rich_media_markdown(text: str, root_dir: Path) -> str:
    """Scans markdown for local images and converts them into optimized Base64 URIs."""
    if not text:
        return ""

    def replace_local_md_img(match):
        alt = match.group(1)
        src = match.group(2).strip()
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        path = (root_dir / src).resolve()
        if path.exists() and path.is_file():
            ext = path.suffix.lower().replace(".", "")
            if ext in ["png", "jpg", "jpeg", "gif", "webp", "svg"]:
                try:
                    mime, b64_str = optimize_image_for_display(path)
                    return (
                        f'<div style="text-align: center; margin: 10px 0;">'
                        f'<img src="data:{mime};base64,{b64_str}" alt="{alt}" loading="lazy" decoding="async" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--jp-border-color1, #cbd5e1); box-shadow: 0 2px 8px rgba(0,0,0,0.1);" /><br/>'
                        f'<span style="font-size: 0.82em; color: var(--jp-content-font-color2, #64748b); font-weight: 500;">{alt}</span>'
                        f'</div>'
                    )
                except Exception:
                    return match.group(0)
        return match.group(0)

    rendered = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_local_md_img, text)

    def replace_local_html_img(match):
        prefix = match.group(1)
        src = match.group(2).strip()
        suffix = match.group(3)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (root_dir / src).resolve()
        if path.exists() and path.is_file():
            ext = path.suffix.lower().replace(".", "")
            if ext in ["png", "jpg", "jpeg", "gif", "webp", "svg"]:
                try:
                    mime, b64_str = optimize_image_for_display(path)
                    return f'{prefix}data:{mime};base64,{b64_str}" loading="lazy" decoding="async{suffix}'
                except Exception:
                    return match.group(0)
        return match.group(0)

    html_pattern = r'(<img[^>]+src=["\'])([^"\']+)(["\'][^>]*>)'
    return re.sub(html_pattern, replace_local_html_img, rendered)


def append_message(output_widget, display_obj, max_outputs: int = 40):
    """Appends display object cleanly while capping in-memory Comm tuple size."""
    if hasattr(output_widget, "outputs") and len(output_widget.outputs) >= max_outputs:
        output_widget.outputs = output_widget.outputs[-(max_outputs - 15):]
    output_widget.append_display_data(display_obj)


async def launch_widgets_ui(agent: KaggleCoREAgent):
    display(HTML(SCROLL_GUARD_HTML))

    header_html = widgets.HTML(layout=widgets.Layout(width='100%', margin='0 0 10px 0'))

    def refresh_header_stats():
        k_mgr = agent.key_manager
        stat = k_mgr.get_pool_status()
        sum_models = ", ".join(getattr(config, "SUMMARIZER_MODELS", []))
        models_pool_str = ", ".join(getattr(config, "GEMINI_MODELS", []))

        try:
            from karyon_hardware import get_hardware_engine
            hw_info = get_hardware_engine().get_telemetry()
            dev_type = hw_info.get('device_type', 'unknown').upper()
            dev_str = hw_info.get('device_str', 'unknown')
        except Exception:
            dev_type = "ACCEL"
            dev_str = "Compute Node"

        active_db_name = Path(config.DB_PATH).name

        header_html.value = f"""
        <div style="background: var(--jp-layout-color2, #f8fafc); border: 1px solid var(--jp-border-color1, #cbd5e1); border-radius: 8px; padding: 12px 16px; color: var(--jp-content-font-color0, #0f172a); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="font-weight: 700; font-size: 16px; margin-bottom: 4px;">🧠 Karyon-CoRE Autonomous Kaggle Research Agent (v31.0 Master)</div>
            <div style="font-size: 12.5px; color: var(--jp-content-font-color2, #64748b); margin-bottom: 8px;"><b>Biophysical Realism, Active Inference & Autonomous Continuous Swarm.</b></div>
            <div style="font-size: 12px; display: flex; flex-wrap: wrap; gap: 14px; font-family: 'JetBrains Mono', Consolas, monospace;">
                <span>🎯 <b>Active Model:</b> <code style="color:#0284c7;">{k_mgr.get_model()}</code></span>
                <span>💾 <b>Active DB:</b> <code style="color:#8b5cf6;">{active_db_name}</code></span>
                <span>🔑 <b>Key Pool:</b> <code style="color:#16a34a;">{stat['active_ready_keys']}/{stat['total_keys']} Ready</code> (Active: Key #{stat['current_key_index']})</span>
                <span>⚙️ <b>Tools:</b> <code>{len(agent.tools_map)} Registered</code></span>
                <span>⚡ <b>Hardware:</b> <code style="color:#d97706;">{dev_type} ({dev_str})</code></span>
                <span>🔄 <b>Max Turns:</b> <code>{config.MAX_AGENT_TURNS}</code></span>
                <span>👥 <b>Swarm Mode:</b> <code style="color:{'#16a34a' if agent.swarm_mode else '#64748b'};">{agent.swarm_mode}</code></span>
                <span>📦 <b>Slim Mode:</b> <code style="color:{'#16a34a' if config.SLIM_PROMPT_MODE else '#dc2626'};">{config.SLIM_PROMPT_MODE}</code></span>
                <span>⏳ <b>Turn Delay:</b> <code>{getattr(config, 'INTER_TURN_DELAY', 0.0)}s</code></span>
            </div>
            <div style="font-size: 11.5px; margin-top: 6px; color: #64748b; font-family: 'JetBrains Mono', Consolas, monospace; border-top: 1px dashed #cbd5e1; padding-top: 4px;">
                <span><b>Inference Models Cascade:</b> <code>{models_pool_str}</code></span><br/>
                <span><b>Summarizer Models Cascade:</b> <code>{sum_models}</code></span>
            </div>
        </div>
        """

    refresh_header_stats()
    display(header_html)

    await agent.db.save_turn(
        "system",
        f"[System Event: Karyon Research Agent UI Session Initialized | Model: {agent.key_manager.get_model()} | Keys: {len(agent.key_manager.keys)} | Tools: {len(agent.tools_map)} | Swarm: {agent.swarm_mode} | Summarizers: {', '.join(getattr(config, 'SUMMARIZER_MODELS', []))} | SlimMode: {config.SLIM_PROMPT_MODE} | MaxTurns: {config.MAX_AGENT_TURNS} | Safety: BLOCK_NONE]"
    )

    output_area = widgets.Output(layout=widgets.Layout(width='100%', height='520px', max_height='520px', overflow_y='auto', overflow_x='hidden'))
    output_area.add_class("karyon-chat-output-container")

    try:
        recent_turns = await agent.db.get_recent_turns(limit=40, include_tools=True)
        if recent_turns:
            history_blocks = ["*(Restored persistent session history from SQLite database:)*\n"]
            for turn in recent_turns:
                role = turn.get("role", "user").lower()
                text_val = turn.get("text", "")
                text_val_rendered = render_rich_media_markdown(text_val, config.PROJECT_ROOT)

                if role == "user":
                    role_label = "**You (Bazilevs):**"
                elif role == "tool_call":
                    role_label = "⚙️ **Tool Invocation:**"
                elif role == "tool_result":
                    role_label = "✅ **Tool Output:**"
                elif role == "agent_message":
                    role_label = "💬 **Karyon Agent (In-Loop):**"
                elif role == "system":
                    role_label = "📋 **System / Action Event:**"
                else:
                    role_label = "### 🤖 Karyon Agent:"

                history_blocks.append(f"\n{role_label}\n{text_val_rendered}\n\n---")

            output_area.append_display_data(Markdown("\n".join(history_blocks)))
    except Exception as e:
        logger.warning(f"Error restoring history: {str(e)}")

    text_input = widgets.Textarea(
        value='',
        placeholder='Type your command or research query here (Ctrl+Enter to send)...',
        description='',
        disabled=False,
        layout=widgets.Layout(width='100%', height='100px')
    )
    text_input.add_class("karyon-text-input")

    file_upload = widgets.FileUpload(
        accept='',  # allow all files
        multiple=True,
        description='Attach Files 📎',
        tooltip='Attach images, PDFs, code, or documents to send to Karyon Agent',
        layout=widgets.Layout(width='auto', height='36px', margin='2px')
    )
    file_upload.add_class("karyon-file-upload")

    btn_layout = widgets.Layout(min_width='105px', max_width='155px', flex='1 1 auto', height='36px', margin='2px')

    send_btn = widgets.Button(description='Send 🚀', button_style='primary', tooltip='Send message to Karyon Agent', layout=btn_layout)
    send_btn.add_class("karyon-send-btn")

    stop_btn = widgets.Button(description='Stop 🛑', button_style='danger', tooltip='Interrupt generation & kill subprocesses', disabled=True, layout=btn_layout)
    auto_loop_btn = widgets.Button(description='Auto Loop ♾️', button_style='success', tooltip='Start continuous autonomous research loop', layout=btn_layout)
    tokens_btn = widgets.Button(description='Tokens 🏷️', button_style='info', tooltip='Inspect exact live token utilization metrics', layout=btn_layout)
    compact_btn = widgets.Button(description='Compact 📦', button_style='', tooltip='Trigger lossless state distillation via summarizer', layout=btn_layout)
    reset_keys_btn = widgets.Button(description='Reset Keys ⚡', button_style='warning', tooltip='Clear all key cooldowns', layout=btn_layout)
    sync_db_btn = widgets.Button(description='Sync DB ☁️', button_style='success', tooltip='Push SQLite state to GitHub repository', layout=btn_layout)
    status_btn = widgets.Button(description='Status 📊', button_style='info', tooltip='Show diagnostics, tools & hardware', layout=btn_layout)
    rotate_btn = widgets.Button(description='Rotate Key 🔄', button_style='', tooltip='Force rotate to next API key', layout=btn_layout)
    rotate_model_btn = widgets.Button(description='Rotate Model 🎯', button_style='', tooltip='Force cascade to next model in pool', layout=btn_layout)
    clear_btn = widgets.Button(description='Clear 🗑', button_style='warning', tooltip='Clear dialogue history from DB', layout=btn_layout)
    swarm_btn = widgets.Button(description='Swarm 👥', button_style='info', tooltip='Inspect Cortical Sub-Agents & Synaptic Traffic', layout=btn_layout)

    status_label = widgets.HTML(value="<b>Status:</b> <span style='color: green;'>Ready</span>")

    # Complete options pool including all models
    model_options = [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
        "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
        "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini-2.5-flash-lite", "gemini-2.0-flash"
    ]
    for m in (config.GEMINI_MODELS + getattr(config, "SUMMARIZER_MODELS", [])):
        if m not in model_options:
            model_options.append(m)

    cfg_item_layout = widgets.Layout(width='100%', min_width='260px', max_width='440px')

    model_dropdown = widgets.Dropdown(
        options=model_options,
        value=agent.key_manager.get_model() if agent.key_manager.get_model() in model_options else model_options[0],
        description='Model:',
        layout=cfg_item_layout
    )

    summarizer_dropdown = widgets.Dropdown(
        options=model_options,
        value=getattr(config, "SUMMARIZER_MODEL", model_options[0]),
        description='Summarizer:',
        layout=cfg_item_layout
    )

    models_pool_input = widgets.Text(
        value=", ".join(config.GEMINI_MODELS),
        description='Model Cascade:',
        tooltip='Comma-separated fallback model list for agent query loop',
        layout=cfg_item_layout
    )

    summarizer_pool_input = widgets.Text(
        value=", ".join(getattr(config, "SUMMARIZER_MODELS", [])),
        description='Summ Pool:',
        tooltip='Comma-separated fallback model list for context distillation',
        layout=cfg_item_layout
    )

    swarm_mode_checkbox = widgets.Checkbox(
        value=bool(agent.swarm_mode),
        description='Swarm Mode (Laminar 4-Stage)',
        tooltip='When active, Autonomous Loop runs Researcher -> Coder -> Refactorer -> Critic',
        layout=cfg_item_layout
    )

    slim_mode_checkbox = widgets.Checkbox(
        value=getattr(config, "SLIM_PROMPT_MODE", True),
        description='Slim Mode (95% TPM Save)',
        tooltip='When active, prevents inlining 120k codebase tokens into every request to eliminate 429 errors',
        layout=cfg_item_layout
    )

    retries_slider = widgets.IntSlider(
        value=getattr(config, "API_MAX_RETRIES", 60),
        min=5,
        max=120,
        step=5,
        description='API Retries:',
        readout_format='d',
        layout=cfg_item_layout
    )

    inter_turn_delay_slider = widgets.FloatSlider(
        value=getattr(config, "INTER_TURN_DELAY", 0.5),
        min=0.0,
        max=10.0,
        step=0.5,
        description='Turn Delay (s):',
        readout_format='.1f',
        layout=cfg_item_layout
    )

    thinking_dropdown = widgets.Dropdown(
        options=["HIGH", "MEDIUM", "LOW", "OFF"],
        value=getattr(config, "THINKING_LEVEL", "HIGH"),
        description='Thinking:',
        layout=cfg_item_layout
    )

    thinking_budget_slider = widgets.IntSlider(
        value=getattr(config, "THINKING_BUDGET", 24576),
        min=0,
        max=32768,
        step=1024,
        description='Budget:',
        readout_format='d',
        layout=cfg_item_layout
    )

    compression_threshold_slider = widgets.IntSlider(
        value=getattr(config, "CONTEXT_COMPRESSION_THRESHOLD", 40000),
        min=15000,
        max=250000,
        step=5000,
        description='Compact At:',
        readout_format='d',
        layout=cfg_item_layout
    )

    max_turns_slider = widgets.IntSlider(
        value=getattr(config, "MAX_AGENT_TURNS", 10000),
        min=20,
        max=10000,
        step=20,
        description='Max Turns:',
        readout_format='d',
        layout=cfg_item_layout
    )

    temp_slider = widgets.FloatSlider(
        value=config.TEMPERATURE,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Temp:',
        readout_format='.2f',
        layout=cfg_item_layout
    )

    top_p_slider = widgets.FloatSlider(
        value=config.TOP_P,
        min=0.5,
        max=1.0,
        step=0.01,
        description='Top-P:',
        readout_format='.2f',
        layout=cfg_item_layout
    )

    apply_cfg_btn = widgets.Button(
        description='Apply & Persist Config ⚙️',
        button_style='primary',
        tooltip='Apply config updates immediately and write to SQLite runtime_settings',
        layout=widgets.Layout(width='200px', height='36px')
    )

    def set_ui_busy_state(is_busy: bool):
        send_btn.disabled = is_busy
        clear_btn.disabled = is_busy
        compact_btn.disabled = is_busy
        tokens_btn.disabled = is_busy
        text_input.disabled = is_busy
        file_upload.disabled = is_busy
        stop_btn.disabled = not is_busy

    async def on_apply_config_clicked_async(b):
        new_model = model_dropdown.value
        new_summarizer = summarizer_dropdown.value
        new_thinking = thinking_dropdown.value
        new_budget = thinking_budget_slider.value
        new_threshold = compression_threshold_slider.value
        new_max_turns = max_turns_slider.value
        new_temp = temp_slider.value
        new_top_p = top_p_slider.value
        new_slim = slim_mode_checkbox.value
        new_retries = retries_slider.value
        new_turn_delay = inter_turn_delay_slider.value
        new_swarm = swarm_mode_checkbox.value

        parsed_models_pool = [m.strip() for m in models_pool_input.value.split(",") if m.strip()]
        parsed_sum_pool = [m.strip() for m in summarizer_pool_input.value.split(",") if m.strip()]

        from karyon_agent_runtime.tools.config_tools import update_runtime_config
        res = await update_runtime_config(
            model=new_model,
            models_pool=",".join(parsed_models_pool),
            summarizer_model=new_summarizer,
            summarizer_models_pool=",".join(parsed_sum_pool),
            context_compression_threshold=new_threshold,
            max_turns=new_max_turns,
            temperature=new_temp,
            top_p=new_top_p,
            thinking_level=new_thinking,
            thinking_budget=new_budget,
            slim_prompt_mode=new_slim,
            api_max_retries=new_retries,
            inter_turn_delay=new_turn_delay,
            swarm_mode=new_swarm
        )

        refresh_header_stats()
        append_message(output_area, Markdown(f"⚙️ **Runtime Configuration Applied & Persisted:**\n```text\n{res}\n```"))

    apply_cfg_btn.on_click(lambda b: asyncio.create_task(on_apply_config_clicked_async(b)))

    config_box = widgets.VBox([
        widgets.HBox([model_dropdown, summarizer_dropdown], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([models_pool_input, summarizer_pool_input], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([swarm_mode_checkbox, slim_mode_checkbox, retries_slider, inter_turn_delay_slider], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([thinking_dropdown, thinking_budget_slider], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([compression_threshold_slider, max_turns_slider], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([temp_slider, top_p_slider], layout=widgets.Layout(flex_flow='row wrap')),
        widgets.HBox([apply_cfg_btn], layout=widgets.Layout(flex_flow='row wrap', margin='4px 0'))
    ])

    config_accordion = widgets.Accordion(children=[config_box])
    config_accordion.set_title(0, '⚙️ Runtime Configuration & Queue Controls (Live Mid-Loop Adjustment)')
    config_accordion.selected_index = None

    in_loop_rendered_final = False

    async def ui_stream_callback(event_type: str, payload: str):
        nonlocal in_loop_rendered_final
        if event_type == "tool_start":
            append_message(output_area, Markdown(f"⚙️ **Executing:** {payload}"))
        elif event_type == "tool_end":
            rendered_payload = render_rich_media_markdown(payload, config.PROJECT_ROOT)
            append_message(output_area, Markdown(f"{rendered_payload}"))
        elif event_type == "agent_message":
            rendered_msg = render_rich_media_markdown(payload, config.PROJECT_ROOT)
            append_message(output_area, Markdown(f"\n💬 **Karyon Agent (Progress Update):**\n{rendered_msg}\n\n---"))
        elif event_type == "final_answer":
            rendered_msg = render_rich_media_markdown(payload, config.PROJECT_ROOT)
            append_message(output_area, Markdown(f"\n### 🤖 Karyon Agent:\n{rendered_msg}\n\n---"))
            in_loop_rendered_final = True
        elif event_type == "media_album":
            append_message(output_area, HTML(payload))
        elif event_type == "synthesis":
            append_message(output_area, Markdown(f"🧠 *{payload}*"))
        elif event_type == "info":
            append_message(output_area, Markdown(f"ℹ️ *{payload}*"))

    last_send_timestamp = 0.0

    async def on_send_clicked(b):
        nonlocal in_loop_rendered_final, last_send_timestamp
        in_loop_rendered_final = False

        if agent.is_busy:
            append_message(output_area, Markdown("⚠️ **Notice:** An agent task is already active. Please wait for completion or click 'Stop 🛑'."))
            return

        now = time.time()
        if now - last_send_timestamp < 0.4:
            return
        last_send_timestamp = now

        user_text = text_input.value.strip()

        # Extract uploaded file attachments from FileUpload widget
        attachments = []
        uploaded_value = file_upload.value
        if uploaded_value:
            # FileUpload widget stores uploaded files in a tuple or dict structure depending on ipywidgets version
            items = uploaded_value if isinstance(uploaded_value, (list, tuple)) else uploaded_value.values() if isinstance(uploaded_value, dict) else []
            for item in items:
                fname = item.get("name", "attachment")
                raw_content = item.get("content", b"")
                # ipywidgets may store content as bytes or memoryview
                if isinstance(raw_content, memoryview):
                    raw_bytes = raw_content.tobytes()
                elif isinstance(raw_content, bytes):
                    raw_bytes = raw_content
                else:
                    raw_bytes = str(raw_content).encode("utf-8")

                b64_str = base64.b64encode(raw_bytes).decode("utf-8")

                # Detect MIME type
                ext = fname.split(".")[-1].lower() if "." in fname else ""
                mime_map = {
                    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml",
                    "pdf": "application/pdf", "txt": "text/plain", "py": "text/plain",
                    "json": "application/json", "csv": "text/csv", "md": "text/markdown"
                }
                mime = mime_map.get(ext, "application/octet-stream")
                attachments.append({"name": fname, "data": b64_str, "mime": mime})

        if not user_text and not attachments:
            return

        if not user_text and attachments:
            user_text = f"Analyze attached file(s): {', '.join([a['name'] for a in attachments])}"

        text_input.value = ''
        file_upload.value.clear() if hasattr(file_upload.value, 'clear') else None
        
        set_ui_busy_state(True)
        status_label.value = "<b>Status:</b> <span style='color: orange;'>⚙️ Thinking and executing tools...</span>"

        user_text_rendered = "\n".join([line + "  " for line in user_text.splitlines()])
        if attachments:
            att_labels = ", ".join([f"📎 `{a['name']}` ({a['mime']})" for a in attachments])
            user_text_rendered += f"\n\n*Attachments:* {att_labels}"

        append_message(output_area, Markdown(f"\n**You (Bazilevs):**\n{user_text_rendered}\n"))

        display(HTML("<script>if(window.karyonForceScrollBottom) window.karyonForceScrollBottom(false);</script>"))

        try:
            response_text = await agent.process_user_query(
                user_text,
                event_callback=ui_stream_callback,
                max_turns=config.MAX_AGENT_TURNS,
                attachments=attachments
            )
            if not in_loop_rendered_final:
                rendered_final = render_rich_media_markdown(response_text, config.PROJECT_ROOT)
                append_message(output_area, Markdown(f"\n### 🤖 Karyon Agent:\n{rendered_final}\n\n---"))
        except asyncio.CancelledError:
            append_message(output_area, Markdown("\n*🛑 Generation and active tasks interrupted by user.*\n\n---"))
        except Exception as e:
            err_details = traceback.format_exc()
            append_message(output_area, Markdown(f"\n**❌ Execution Error:**\n```text\n{err_details}\n```\n\n---"))
        finally:
            refresh_header_stats()
            set_ui_busy_state(False)
            status_label.value = "<b>Status:</b> <span style='color: green;'>Ready</span>"

    async def on_stop_clicked(b):
        status_label.value = "<b>Status:</b> <span style='color: red;'>Stopping...</span>"
        if agent.autonomous_loop_active:
            await agent.stop_autonomous_loop()
        else:
            await agent.cancel_active_generation()
        agent.reset_locks()
        auto_loop_btn.description = 'Auto Loop ♾️'
        auto_loop_btn.button_style = 'success'
        stop_btn.disabled = True
        set_ui_busy_state(False)
        status_label.value = "<b>Status:</b> <span style='color: red;'>Stopped by user</span>"
        append_message(output_area, Markdown("\n*🛑 Autonomous loop and all active generation tasks completely halted.*\n\n---"))

    async def on_auto_loop_clicked(b):
        if agent.autonomous_loop_active:
            if agent.autonomous_loop_paused:
                agent.resume_autonomous_loop()
                auto_loop_btn.description = 'Pause Loop ⏸️'
                auto_loop_btn.button_style = 'warning'
                stop_btn.disabled = False
                status_label.value = "<b>Status:</b> <span style='color: green;'>♾️ Autonomous Loop Resumed</span>"
                append_message(output_area, Markdown("▶️ **Autonomous Research Daemon Resumed.**"))
            else:
                agent.pause_autonomous_loop()
                auto_loop_btn.description = 'Resume Loop ▶️'
                auto_loop_btn.button_style = 'info'
                status_label.value = "<b>Status:</b> <span style='color: orange;'>⏸️ Autonomous Loop Paused</span>"
                append_message(output_area, Markdown("⏸️ **Autonomous Research Daemon Paused. In-flight tasks cleanly halted.**"))
        else:
            auto_loop_btn.description = 'Pause Loop ⏸️'
            auto_loop_btn.button_style = 'warning'
            stop_btn.disabled = False
            status_label.value = "<b>Status:</b> <span style='color: green;'>🚀 Autonomous Loop Active</span>"
            mode_tag = "Cortical Swarm (Laminar 4-Stage)" if agent.swarm_mode else "Monolithic Loop"
            append_message(output_area, Markdown(f"🚀 **Starting Continuous Autonomous Research Daemon ({mode_tag})...**"))
            await agent.start_autonomous_loop(
                research_agenda="Advance Karyon-CoRE biophysical realism and empirical throughput under KEP v9.0 Master.",
                max_cycles=1000,
                interval_seconds=15.0,
                status_callback=ui_stream_callback
            )

    async def on_tokens_clicked(b):
        append_message(output_area, Markdown("🏷️ *Auditing exact active token utilization across all context channels...*"))
        res = await get_context_token_status()
        append_message(output_area, Markdown(f"```text\n{res}\n```"))

    async def on_compact_clicked(b):
        append_message(output_area, Markdown("📦 *Executing lossless context state distillation via SUMMARIZER_MODELS pool...*"))
        res = await compress_context_now()
        append_message(output_area, Markdown(f"```text\n{res}\n```"))

    async def on_reset_keys_clicked(b):
        from karyon_agent_runtime.tools.config_tools import reset_key_cooldowns
        res = await reset_key_cooldowns()
        refresh_header_stats()
        append_message(output_area, Markdown(f"⚡ {res}"))

    async def on_status_clicked(b):
        from karyon_agent_runtime.tools.config_tools import get_runtime_config
        res = await get_runtime_config()
        append_message(output_area, Markdown(f"```text\n{res}\n```"))

    async def on_sync_db_clicked(b):
        append_message(output_area, Markdown("☁️ *Flushing SQLite checkpoint & syncing state to private repository...*"))
        from karyon_agent_runtime.tools.db_tools import sync_agent_database
        res = await sync_agent_database("chore(db): manual UI state snapshot sync")
        await agent.db.save_turn("system", "[System Event: User manually triggered SQLite database sync to GitHub]")
        append_message(output_area, Markdown(f"✅ **Database Sync Complete:**\n```text\n{res}\n```"))

    async def on_rotate_clicked(b):
        from karyon_agent_runtime.tools.config_tools import force_rotate_key
        res = await force_rotate_key()
        refresh_header_stats()
        append_message(output_area, Markdown(f"🔄 {res}"))

    async def on_rotate_model_clicked(b):
        from karyon_agent_runtime.tools.config_tools import force_rotate_model
        res = await force_rotate_model()
        model_dropdown.value = agent.key_manager.get_model()
        refresh_header_stats()
        append_message(output_area, Markdown(f"🎯 {res}"))

    async def on_clear_clicked(b):
        await agent.db.clear_turns()
        output_area.clear_output()
        await agent.db.save_turn("system", "[System Event: User cleared dialogue history]")
        append_message(output_area, Markdown("**Dialogue history cleared from database.**"))

    async def on_swarm_clicked(b):
        from karyon_agent_runtime.tools.multi_agent_tools import get_swarm_telemetry
        append_message(output_area, Markdown('👥 *Auditing Cortical Sub-Agents & Inter-Agent Synaptic Traffic...*'))
        res = await get_swarm_telemetry()
        append_message(output_area, Markdown(res))

    send_btn.on_click(lambda b: asyncio.create_task(on_send_clicked(b)))
    stop_btn.on_click(lambda b: asyncio.create_task(on_stop_clicked(b)))
    auto_loop_btn.on_click(lambda b: asyncio.create_task(on_auto_loop_clicked(b)))
    tokens_btn.on_click(lambda b: asyncio.create_task(on_tokens_clicked(b)))
    compact_btn.on_click(lambda b: asyncio.create_task(on_compact_clicked(b)))
    reset_keys_btn.on_click(lambda b: asyncio.create_task(on_reset_keys_clicked(b)))
    sync_db_btn.on_click(lambda b: asyncio.create_task(on_sync_db_clicked(b)))
    status_btn.on_click(lambda b: asyncio.create_task(on_status_clicked(b)))
    rotate_btn.on_click(lambda b: asyncio.create_task(on_rotate_clicked(b)))
    rotate_model_btn.on_click(lambda b: asyncio.create_task(on_rotate_model_clicked(b)))
    clear_btn.on_click(lambda b: asyncio.create_task(on_clear_clicked(b)))
    swarm_btn.on_click(lambda b: asyncio.create_task(on_swarm_clicked(b)))

    btn_row = widgets.HBox(
        [send_btn, stop_btn, auto_loop_btn, swarm_btn, tokens_btn, compact_btn, reset_keys_btn, sync_db_btn, status_btn, rotate_btn, rotate_model_btn, clear_btn],
        layout=widgets.Layout(flex_flow='row wrap', width='100%', margin='4px 0')
    )
    btn_row.add_class("karyon-btn-row")

    top_controls = widgets.VBox([btn_row, widgets.HBox([status_label])])
    ui_box = widgets.VBox([output_area, text_input, top_controls, config_accordion])
    display(ui_box)


async def run_console_session(agent: KaggleCoREAgent):
    display(Markdown(f"""
# 🧠 Karyon-CoRE Autonomous Kaggle Research Agent (v31.0 Console Mode)
**Biophysical Realism, Active Inference & Autonomous Tools Initialized.**
---
"""))
    await agent.db.save_turn("system", "[System Event: Karyon Console Session Started]")
    while True:
        try:
            user_input = input("\nYou (Bazilevs): ")
        except (KeyboardInterrupt, EOFError):
            break

        clean_input = user_input.strip()
        if not clean_input:
            continue
        if clean_input.lower() in ["exit", "quit"]:
            await agent.db.save_turn("system", "[System Event: User closed console session]")
            display(Markdown("**Session closed.**"))
            break
        if clean_input.lower() == "clear":
            await agent.db.clear_turns()
            display(Markdown("**Dialogue history cleared.**"))
            continue
        if clean_input.lower() == "loop":
            print("🚀 Launching Autonomous Continuous Research Daemon...")
            await agent.start_autonomous_loop()
            continue

        print("\nThinking and executing tools in environment...")

        async def console_stream_callback(event_type: str, payload: str):
            if event_type == "tool_start":
                print(f"⚙️  {payload}")
            elif event_type == "tool_end":
                print(f"  {payload[:300]}...")
            elif event_type in ["agent_message", "final_answer"]:
                print(f"\n💬 Karyon Agent: {payload}\n")
            elif event_type == "synthesis":
                print(f"🧠 {payload}")

        response_text = await agent.process_user_query(
            clean_input,
            event_callback=console_stream_callback,
            max_turns=config.MAX_AGENT_TURNS
        )
        display(Markdown(f"\n### 🤖 Karyon Agent:\n{response_text}\n\n---"))


async def main_entry():
    agent = KaggleCoREAgent()
    await agent.initialize()
    if HAS_WIDGETS:
        await launch_widgets_ui(agent)
    else:
        await run_console_session(agent)


def is_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def main():
    if is_in_colab():
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        try:
            nest_asyncio.apply()
            return loop.run_until_complete(main_entry())
        except Exception:
            return asyncio.create_task(main_entry())
    else:
        return loop.run_until_complete(main_entry())


if __name__ == "__main__":
    main()
