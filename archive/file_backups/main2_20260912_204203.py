# karyon_agent_runtime/main2.py
"""
===============================================================================
KARYON CORE AGENT ADVANCED MOBILE WEB RUNTIME (GOOGLE COLAB PRO SUITE v31.0 MASTER)
Synchronized with Extended Gemini Matrix, Multi-Model Summarizer Pool,
Unified Process Engine, Custom Tool Hot-Loading, and KaTeX LaTeX Math Engine:
- Full GFM Markdown via Marked.js & Real-Time LaTeX Formula Rendering via KaTeX
- Syntax Highlighting for 180+ Languages via Highlight.js with Copy-to-Clipboard
- PIL-Powered Image Downsampling & WebP/JPEG Lightweight Inlining (Anti-OOM)
- Anti-Jitter Touch Scroll with User Intent Tracking & Sticky Pin-to-Bottom State
- Client-Side Send-Guard with Busy Kernel Detection (%run Race Fix) & Ctrl+Enter
- Batch Single-Pass Historical Turn Restoration (DocumentFragment Performance)
- Bounded DOM Memory Management (Caps Active Messages to Prevent Mobile Crashes)
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

# KARYON_PATCH_V31_APPLIED

import sys
import os
import re
import io
import json
import time
import base64
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML

# 1. Resolve project and runtime paths
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for p in [str(_root_dir), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from karyon_agent_runtime.agent_core import KaggleCoREAgent
import karyon_agent_runtime.config as config
from karyon_agent_runtime.tools.db_tools import sync_agent_database
from karyon_agent_runtime.tools.context_tools import get_context_token_status, compress_context_now

try:
    from google.colab import output
    IS_COLAB = True
except ImportError:
    IS_COLAB = False


# 2. Rich Media Downsampler & Base64 Converter (Anti-OOM)
def optimize_image_for_display(path: Path, max_dim: int = 850, quality: int = 82) -> tuple:
    """Reads and resizes image to lightweight compressed Base64 to prevent mobile RAM crashes."""
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
    """Scans markdown and HTML for local images and converts them into optimized Base64 data URIs."""
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
                        f'<img src="data:{mime};base64,{b64_str}" alt="{alt}" loading="lazy" decoding="async" style="max-width: 100%; border-radius: 8px; border: 1px solid #45475a; box-shadow: 0 3px 10px rgba(0,0,0,0.3);" /><br/>'
                        f'<span style="font-size: 0.82em; color: #a6adc8; font-weight: 500;">{alt}</span>'
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

    return re.sub(r'(<img[^>]+src=["\'])([^"\']+)(["\'][^>]*>)', replace_local_html_img, rendered)


# 3. HTML Web UI Template with Complete CSS, JS & KaTeX Math Rendering
HTML_FULL_PANEL = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script>

    <style>
        #karyon-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #181825;
            color: #cdd6f4;
            border-radius: 12px;
            border: 1px solid #313244;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            max-width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            position: relative;
        }
        .header {
            background: #11111b;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #313244;
        }
        .title { font-weight: 700; font-size: 15px; color: #cdd6f4; display: flex; align-items: center; gap: 8px; }
        .badge { font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; transition: all 0.3s; }
        .badge-ready { background: #a6e3a1; color: #11111b; }
        .badge-busy { background: #fab387; color: #11111b; }
        .badge-stop { background: #f38ba8; color: #11111b; }
        
        #chat-window {
            height: 520px;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding: 16px;
            background: #1e1e2e;
            display: flex;
            flex-direction: column;
            gap: 14px;
            scroll-behavior: auto !important;
            overscroll-behavior-y: contain !important;
            overflow-anchor: none !important;
            -webkit-overflow-scrolling: touch !important;
            touch-action: pan-y !important;
            contain: paint layout !important;
            will-change: scroll-position !important;
            position: relative !important;
        }
        #chat-window::-webkit-scrollbar { width: 6px; }
        #chat-window::-webkit-scrollbar-thumb { background: #45475a; border-radius: 3px; }

        .karyon-scroll-bottom-pill {
            position: absolute;
            bottom: 155px;
            right: 24px;
            background: #89b4fa;
            color: #11111b;
            font-size: 11.5px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.4);
            cursor: pointer;
            z-index: 9999;
            opacity: 0;
            transform: translateY(10px);
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

        .msg {
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 13.5px;
            line-height: 1.6;
            max-width: 96%;
            word-break: break-word;
            box-sizing: border-box;
            position: relative;
        }
        .msg-user {
            background: #89b4fa;
            color: #11111b;
            align-self: flex-end;
            font-weight: 500;
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }
        .msg-agent { background: #181825; color: #cdd6f4; border: 1px solid #313244; align-self: flex-start; width: 96%; }
        .msg-inloop { background: #1e1e2e; border-left: 4px solid #f9e2af; color: #cdd6f4; padding: 10px 14px; align-self: flex-start; width: 96%; }
        .msg-tool-call { background: #11111b; border-left: 3px solid #89dceb; color: #89dceb; font-family: monospace; font-size: 12px; padding: 8px 12px; align-self: flex-start; width: 96%; }
        .msg-tool-res { background: #11111b; border-left: 3px solid #a6e3a1; color: #a6adc8; font-family: monospace; font-size: 12px; padding: 8px 12px; align-self: flex-start; width: 96%; }
        .msg-system { background: #11111b; border: 1px dashed #45475a; color: #6c7086; font-size: 12px; padding: 6px 12px; align-self: center; border-radius: 6px; text-align: center; }

        .msg table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            border: 1px solid #45475a;
            font-size: 13px;
        }
        .msg th, .msg td { border: 1px solid #45475a; padding: 8px 12px; text-align: left; }
        .msg th { background: #313244; color: #89b4fa; font-weight: 700; }
        .msg tr:nth-child(even) { background: #181825; }

        .msg pre {
            background: #11111b !important;
            padding: 12px 14px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #313244;
            position: relative;
        }
        .msg code { font-family: "JetBrains Mono", Consolas, monospace; font-size: 12px; }

        .karyon-media-album { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 12px 0; }
        .msg img { max-width: 100%; border-radius: 8px; margin: 8px 0; border: 1px solid #45475a; display: block; }

        .config-drawer {
            background: #11111b;
            border-bottom: 1px solid #313244;
            padding: 14px 16px;
            display: none;
            flex-direction: column;
            gap: 12px;
        }
        .cfg-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
        .cfg-item { display: flex; flex-direction: column; gap: 4px; font-size: 11px; color: #a6adc8; font-weight: 600; }
        .cfg-input { background: #1e1e2e; border: 1px solid #45475a; border-radius: 6px; color: #cdd6f4; padding: 6px 8px; font-size: 12px; outline: none; }

        .btn-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding: 8px 12px;
            background: #11111b;
            border-top: 1px solid #313244;
        }
        .btn {
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: transform 0.1s, opacity 0.2s;
        }
        .btn:active { transform: scale(0.96); }
        .btn-primary { background: #89b4fa; color: #11111b; }
        .btn-danger { background: #f38ba8; color: #11111b; }
        .btn-warning { background: #fab387; color: #11111b; }
        .btn-success { background: #a6e3a1; color: #11111b; }
        .btn-info { background: #89dceb; color: #11111b; }
        .btn-secondary { background: #313244; color: #cdd6f4; border: 1px solid #45475a; }

        .log-bar {
            background: #11111b;
            color: #89dceb;
            font-family: "JetBrains Mono", monospace;
            font-size: 11.5px;
            padding: 6px 14px;
            border-top: 1px solid #313244;
            display: none;
            white-space: pre-wrap;
            max-height: 70px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div id="karyon-container">
        <div class="header">
            <div style="display: flex; flex-direction: column; gap: 2px;">
                <span class="title">🧠 Karyon CoRE Agent (Colab Pro Suite v31.0 | Unified Process Runtime)</span>
                <span id="live-header-stats" style="font-size: 11.5px; color: #a6adc8; font-family: monospace;">Loading status...</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <button class="btn btn-secondary" onclick="toggleConfig()" style="padding: 4px 10px; font-size: 11px;">⚙ Config</button>
                <span id="badge" class="badge badge-ready">Ready</span>
            </div>
        </div>

        <div id="config-drawer" class="config-drawer">
            <div class="cfg-grid">
                <div class="cfg-item">
                    <label>Gemini Active Model:</label>
                    <select id="cfg-model" class="cfg-input">
                        <option value="gemini-3.8-flash" selected>gemini-3.8-flash</option>
                        <option value="gemini-3.7-flash">gemini-3.7-flash</option>
                        <option value="gemini-3.6-flash">gemini-3.6-flash</option>
                        <option value="gemini-3.5-flash">gemini-3.5-flash</option>
                        <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite</option>
                        <option value="gemini-3.1-pro-preview">gemini-3.1-pro-preview</option>
                        <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite</option>
                        <option value="gemini-3-flash-preview">gemini-3-flash-preview</option>
                    </select>
                </div>
                <div class="cfg-item">
                    <label>Primary Summarizer Model:</label>
                    <select id="cfg-summarizer" class="cfg-input">
                        <option value="gemini-3.8-flash" selected>gemini-3.8-flash</option>
                        <option value="gemini-3.7-flash">gemini-3.7-flash</option>
                        <option value="gemini-3.6-flash">gemini-3.6-flash</option>
                        <option value="gemini-3.5-flash">gemini-3.5-flash</option>
                        <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite</option>
                        <option value="gemini-3.1-pro-preview">gemini-3.1-pro-preview</option>
                        <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite</option>
                        <option value="gemini-3-flash-preview">gemini-3-flash-preview</option>
                    </select>
                </div>
                <div class="cfg-item">
                    <label>Inference Models Cascade:</label>
                    <input type="text" id="cfg-models-pool" class="cfg-input" value="gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash">
                </div>
                <div class="cfg-item">
                    <label>Summarizer Models Pool:</label>
                    <input type="text" id="cfg-sum-pool" class="cfg-input" value="gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash">
                </div>
                <div class="cfg-item">
                    <label>Swarm Mode (Laminar 4-Stage):</label>
                    <select id="cfg-swarm" class="cfg-input">
                        <option value="true" selected>Enabled (Laminar Swarm)</option>
                        <option value="false">Disabled (Monolithic)</option>
                    </select>
                </div>
                <div class="cfg-item">
                    <label>Slim Mode (95% TPM Save):</label>
                    <select id="cfg-slim" class="cfg-input">
                        <option value="true" selected>Enabled (Prevent 429)</option>
                        <option value="false">Disabled (Full Ingestion)</option>
                    </select>
                </div>
                <div class="cfg-item">
                    <label>API Max Retries:</label>
                    <input type="number" id="cfg-retries" class="cfg-input" value="60" min="5" max="120">
                </div>
                <div class="cfg-item">
                    <label>Turn Delay (Seconds):</label>
                    <input type="number" step="0.5" id="cfg-turndelay" class="cfg-input" value="0.5" min="0.0" max="10.0">
                </div>
                <div class="cfg-item">
                    <label>Thinking Level:</label>
                    <select id="cfg-thinking" class="cfg-input">
                        <option value="HIGH" selected>HIGH (Deep Reasoning)</option>
                        <option value="MEDIUM">MEDIUM (Balanced)</option>
                        <option value="LOW">LOW (Fast)</option>
                        <option value="OFF">OFF (Disabled)</option>
                    </select>
                </div>
                <div class="cfg-item">
                    <label>Compact Threshold (Tokens):</label>
                    <input type="number" id="cfg-threshold" class="cfg-input" value="40000" min="15000" max="250000" step="5000">
                </div>
                <div class="cfg-item">
                    <label>Thinking Budget:</label>
                    <input type="number" id="cfg-budget" class="cfg-input" value="24576" step="1024">
                </div>
                <div class="cfg-item">
                    <label>Max Turns Limit:</label>
                    <input type="number" id="cfg-turns" class="cfg-input" value="10000" step="20">
                </div>
                <div class="cfg-item">
                    <label>Temperature (0-1):</label>
                    <input type="number" step="0.05" id="cfg-temp" class="cfg-input" value="0.40">
                </div>
                <div class="cfg-item">
                    <label>Top-P (0.5-1):</label>
                    <input type="number" step="0.01" id="cfg-topp" class="cfg-input" value="0.95">
                </div>
            </div>
            <button class="btn btn-primary" onclick="applyConfig()" style="align-self: flex-start; margin-top: 4px;">Apply & Persist Config ⚙</button>
        </div>

        <div id="chat-window"></div>
        <div id="karyon-colab-pill" class="karyon-scroll-bottom-pill">⬇️ Bottom</div>

        <div id="log-bar" class="log-bar"></div>

        <div class="btn-toolbar">
            <button id="btn-send" class="btn btn-primary" onclick="sendMessage()">Send 🚀</button>
            <button id="btn-stop" class="btn btn-danger" onclick="stopTask()">Stop 🛑</button>
            <button id="btn-auto-loop" class="btn btn-success" onclick="toggleAutoLoop()">Auto Loop ♾️</button>
            <button id="btn-swarm" class="btn btn-info" onclick="getSwarmStatus()">Swarm 👥</button>
            <button class="btn btn-info" onclick="getTokens()">Tokens 🏷️</button>
            <button class="btn btn-secondary" onclick="compactContext()">Compact 📦</button>
            <button class="btn btn-warning" onclick="resetKeys()">Reset Keys ⚡</button>
            <button class="btn btn-success" onclick="syncDb()">Sync DB ☁</button>
            <button class="btn btn-info" onclick="getStatus()">Status 📊</button>
            <button class="btn btn-secondary" onclick="rotateKey()">Rotate Key 🔄</button>
            <button class="btn btn-secondary" onclick="rotateModel()">Rotate Model 🎯</button>
            <button class="btn btn-danger" onclick="clearHistory()" style="background: #313244; color: #f38ba8;">Clear 🗑</button>
        </div>

        <div style="padding: 10px 12px; background: #11111b; border-top: 1px solid #313244; display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; gap: 8px; align-items: center;">
                <label for="file-input" class="btn btn-secondary" style="cursor: pointer; padding: 6px 12px; font-size: 12px; margin: 0; display: inline-flex; align-items: center; gap: 4px;">
                    📎 Attach Files
                </label>
                <input type="file" id="file-input" multiple style="display: none;" onchange="handleFileSelect(event)">
                <div id="file-list" style="font-size: 11.5px; color: #a6adc8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;"></div>
            </div>
            <textarea id="msg-input" placeholder="Type your research query or command here (Ctrl+Enter to send)..." style="width: 100%; height: 80px; background: #1e1e2e; border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4; font-size: 13.5px; padding: 8px 10px; box-sizing: border-box; resize: vertical; outline: none;"></textarea>
        </div>
    </div>

    <script>
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            gfm: true,
            breaks: true
        });

        function renderMath(element) {
            if (window.renderMathInElement) {
                renderMathInElement(element, {
                    delimiters: [
                        {left: "$$", right: "$$", display: true},
                        {left: "$", right: "$", display: false},
                        {left: "\\\\(", right: "\\\\)", display: false},
                        {left: "\\\\[", right: "\\\\]", display: true}
                    ],
                    throwOnError: false
                });
            }
        }

        const chatWin = document.getElementById('chat-window');
        const pill = document.getElementById('karyon-colab-pill');

        chatWin._karyon_pinned = true;
        chatWin._karyon_user_scrolling = false;
        chatWin._karyon_touching = false;
        chatWin._karyon_last_touch = 0;
        chatWin._karyon_raf_id = null;

        pill.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            chatWin._karyon_pinned = true;
            chatWin._karyon_user_scrolling = false;
            pill.classList.remove('visible');
            chatWin.scrollTo({ top: chatWin.scrollHeight, behavior: 'smooth' });
        });

        function updatePill() {
            if (!chatWin._karyon_pinned) {
                pill.classList.add('visible');
            } else {
                pill.classList.remove('visible');
            }
        }

        chatWin.addEventListener('touchstart', function() {
            chatWin._karyon_touching = true;
            chatWin._karyon_user_scrolling = true;
            chatWin._karyon_last_touch = Date.now();
        }, { passive: true });

        chatWin.addEventListener('touchend', function() {
            chatWin._karyon_touching = false;
            chatWin._karyon_last_touch = Date.now();
        }, { passive: true });

        chatWin.addEventListener('wheel', function() {
            chatWin._karyon_user_scrolling = true;
            chatWin._karyon_last_touch = Date.now();
        }, { passive: true });

        chatWin.addEventListener('scroll', function() {
            var distToBottom = chatWin.scrollHeight - chatWin.scrollTop - chatWin.clientHeight;
            if (distToBottom <= 40) {
                chatWin._karyon_pinned = true;
                chatWin._karyon_user_scrolling = false;
                updatePill();
            } else if (distToBottom > 80 && chatWin._karyon_user_scrolling) {
                chatWin._karyon_pinned = false;
                updatePill();
            }
        }, { passive: true });

        function scheduleScrollBottom(force) {
            if (!force && (!chatWin._karyon_pinned || chatWin._karyon_touching)) return;
            if (!force && (Date.now() - chatWin._karyon_last_touch < 350)) return;

            if (chatWin._karyon_raf_id) cancelAnimationFrame(chatWin._karyon_raf_id);

            chatWin._karyon_raf_id = requestAnimationFrame(function() {
                chatWin._karyon_raf_id = null;
                var target = chatWin.scrollHeight - chatWin.clientHeight;
                if (target > 0) {
                    chatWin.scrollTop = target;
                }
            });
        }

        window.KaryonBridge = {
            setAutoLoopState: function(state, cycle) {
                const btn = document.getElementById('btn-auto-loop');
                if (!btn) return;
                if (state === 'running') {
                    btn.className = 'btn btn-warning';
                    btn.innerText = 'Pause Loop ⏸️ (#' + cycle + ')';
                } else if (state === 'paused') {
                    btn.className = 'btn btn-info';
                    btn.innerText = 'Resume Loop ▶️ (#' + cycle + ')';
                } else {
                    btn.className = 'btn btn-success';
                    btn.innerText = 'Auto Loop ♾️';
                }
            },
            updateHeaderStats: function(text) {
                const el = document.getElementById('live-header-stats');
                if (el) { el.innerText = text; }
            },
            appendMessage: function(type, htmlOrText) {
                const div = document.createElement('div');
                div.className = 'msg ' + type;
                if (type === 'msg-user' || type === 'msg-system') {
                    div.innerText = htmlOrText;
                } else {
                    div.innerHTML = marked.parse(htmlOrText);
                    renderMath(div);
                }
                chatWin.appendChild(div);

                while (chatWin.children.length > 50) {
                    chatWin.removeChild(chatWin.firstChild);
                }

                scheduleScrollBottom(type === 'msg-user');
            },
            restoreHistory: function(turnsList) {
                if (!Array.isArray(turnsList) || turnsList.length === 0) return;
                const frag = document.createDocumentFragment();
                for (var i = 0; i < turnsList.length; i++) {
                    var item = turnsList[i];
                    var div = document.createElement('div');
                    div.className = 'msg ' + item.type;
                    if (item.type === 'msg-user' || item.type === 'msg-system') {
                        div.innerText = item.text;
                    } else {
                        div.innerHTML = marked.parse(item.text);
                        renderMath(div);
                    }
                    frag.appendChild(div);
                }
                chatWin.appendChild(frag);

                while (chatWin.children.length > 50) {
                    chatWin.removeChild(chatWin.firstChild);
                }

                chatWin._karyon_pinned = true;
                requestAnimationFrame(function() {
                    chatWin.scrollTop = chatWin.scrollHeight;
                });
            },
            updateLog: function(text) {
                const bar = document.getElementById('log-bar');
                if (text) { bar.style.display = 'block'; bar.innerText = text; }
                else { bar.style.display = 'none'; }
                scheduleScrollBottom(false);
            },
            setBusy: function(busy) {
                const badge = document.getElementById('badge');
                const btnSend = document.getElementById('btn-send');
                const input = document.getElementById('msg-input');
                if (busy) {
                    badge.className = 'badge badge-busy'; badge.innerText = 'Busy';
                    btnSend.disabled = true; btnSend.style.opacity = '0.6'; input.disabled = true;
                } else {
                    badge.className = 'badge badge-ready'; badge.innerText = 'Ready';
                    btnSend.disabled = false; btnSend.style.opacity = '1.0'; input.disabled = false;
                    input.focus();
                }
            },
            clearChat: function() {
                chatWin.innerHTML = '';
            }
        };

        function toggleConfig() {
            const drawer = document.getElementById('config-drawer');
            drawer.style.display = (drawer.style.display === 'flex') ? 'none' : 'flex';
        }

        document.getElementById('msg-input').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });

        var currentAttachments = [];

        function handleFileSelect(e) {
            const files = e.target.files;
            if (!files || files.length === 0) return;
            const fileListEl = document.getElementById('file-list');
            for (let i = 0; i < files.length; i++) {
                const f = files[i];
                const reader = new FileReader();
                reader.onload = function(evt) {
                    const b64Data = evt.target.result.split(',')[1] || '';
                    currentAttachments.push({
                        name: f.name,
                        data: b64Data,
                        mime: f.type || 'application/octet-stream'
                    });
                    renderAttachmentPills();
                };
                reader.readAsDataURL(f);
            }
        }

        function renderAttachmentPills() {
            const fileListEl = document.getElementById('file-list');
            if (currentAttachments.length === 0) {
                fileListEl.innerHTML = '';
                return;
            }
            fileListEl.innerHTML = currentAttachments.map((a, idx) => 
                `<span style="display:inline-block; background:#313244; color:#cdd6f4; padding:2px 8px; border-radius:4px; margin-right:4px; font-size:11px;">📎 ${a.name} <b style="cursor:pointer; color:#f38ba8; margin-left:4px;" onclick="removeAttachment(${idx})">×</b></span>`
            ).join('');
        }

        function removeAttachment(idx) {
            currentAttachments.splice(idx, 1);
            renderAttachmentPills();
        }

        function sendMessage() {
            const input = document.getElementById('msg-input');
            let text = input.value.trim();
            if (!text && currentAttachments.length === 0) return;

            var isKernelBusy = false;
            try {
                if (window.google && google.colab && google.colab.kernel && typeof google.colab.kernel.isBusy === 'function') {
                    isKernelBusy = google.colab.kernel.isBusy();
                }
            } catch (e) {}

            if (document.querySelector('colab-run-button[running]') ||
                document.querySelector('.cell-execution-indicator[running]') ||
                document.body.classList.contains('colab-busy') ||
                document.querySelector('[status="busy"]')) {
                isKernelBusy = true;
            }

            if (isKernelBusy) {
                var warnEl = document.getElementById('karyon-colab-busy-warning');
                if (!warnEl) {
                    warnEl = document.createElement('div');
                    warnEl.id = 'karyon-colab-busy-warning';
                    warnEl.style.cssText = 'position:fixed; bottom:24px; left:50%; transform:translateX(-50%); background:#dc2626; color:#ffffff; font-size:12.5px; font-weight:700; padding:10px 18px; border-radius:8px; z-index:999999; box-shadow:0 6px 20px rgba(0,0,0,0.5); text-align:center; transition:opacity 0.25s ease; pointer-events:none;';
                    document.body.appendChild(warnEl);
                }
                warnEl.innerText = '⚠️ Colab Kernel is currently busy executing another cell (e.g. %run). Message NOT sent to prevent desync. Please wait for that cell to finish!';
                warnEl.style.display = 'block';
                warnEl.style.opacity = '1';
                setTimeout(function() {
                    warnEl.style.opacity = '0';
                    setTimeout(function() { warnEl.style.display = 'none'; }, 300);
                }, 4500);
                return;
            }

            chatWin._karyon_pinned = true;
            chatWin._karyon_user_scrolling = false;

            var userDisplay = text;
            if (currentAttachments.length > 0) {
                var attTags = currentAttachments.map(a => `📎 \`${a.name}\``).join(', ');
                userDisplay += (userDisplay ? '\n\n' : '') + `*Attachments:* ${attTags}`;
            }

            window.KaryonBridge.appendMessage('msg-user', userDisplay);
            input.value = '';
            
            var payloadAttachments = currentAttachments.slice();
            currentAttachments = [];
            renderAttachmentPills();
            document.getElementById('file-input').value = '';

            window.KaryonBridge.setBusy(true);
            window.KaryonBridge.updateLog('🚀 Query dispatched to agent...');

            scheduleScrollBottom(true);
            google.colab.kernel.invokeFunction('karyon.send_message', [text, payloadAttachments], {});
        }

        function getSwarmStatus() {
            window.KaryonBridge.updateLog('👥 Querying Cortical Swarm status...');
            google.colab.kernel.invokeFunction('karyon.get_swarm_status', [], {});
        }

        function toggleAutoLoop() {
            window.KaryonBridge.updateLog('♾️ Toggling Continuous Autonomous Loop...');
            google.colab.kernel.invokeFunction('karyon.toggle_auto_loop', [], {});
        }

        function stopTask() {
            window.KaryonBridge.updateLog('🛑 Stopping active tasks...');
            google.colab.kernel.invokeFunction('karyon.stop_task', [], {});
        }

        function getTokens() {
            window.KaryonBridge.updateLog('🏷️ Auditing token utilization...');
            google.colab.kernel.invokeFunction('karyon.get_tokens', [], {});
        }

        function resetKeys() {
            window.KaryonBridge.updateLog('⚡ Resetting API key cooldowns...');
            google.colab.kernel.invokeFunction('karyon.reset_keys', [], {});
        }

        function syncDb() {
            window.KaryonBridge.updateLog('☁ Syncing SQLite state to GitHub...');
            google.colab.kernel.invokeFunction('karyon.sync_db', [], {});
        }

        function getStatus() { google.colab.kernel.invokeFunction('karyon.get_status', [], {}); }
        function rotateKey() { google.colab.kernel.invokeFunction('karyon.rotate_key', [], {}); }
        function rotateModel() {
            window.KaryonBridge.updateLog('🎯 Cascading to next model in pool...');
            google.colab.kernel.invokeFunction('karyon.rotate_model', [], {});
        }
        function compactContext() {
            window.KaryonBridge.updateLog('📦 Distilling context state...');
            google.colab.kernel.invokeFunction('karyon.compact_context', [], {});
        }
        function clearHistory() {
            window.KaryonBridge.clearChat();
            google.colab.kernel.invokeFunction('karyon.clear_history', [], {});
        }

        function applyConfig() {
            const cfg = {
                model: document.getElementById('cfg-model').value,
                summarizer: document.getElementById('cfg-summarizer').value,
                models_pool: document.getElementById('cfg-models-pool').value,
                sum_pool: document.getElementById('cfg-sum-pool').value,
                swarm_mode: document.getElementById('cfg-swarm').value === 'true',
                slim: document.getElementById('cfg-slim').value === 'true',
                retries: parseInt(document.getElementById('cfg-retries').value) || 60,
                turndelay: parseFloat(document.getElementById('cfg-turndelay').value) || 0.5,
                thinking: document.getElementById('cfg-thinking').value,
                threshold: parseInt(document.getElementById('cfg-threshold').value) || 40000,
                budget: parseInt(document.getElementById('cfg-budget').value) || 24576,
                max_turns: parseInt(document.getElementById('cfg-turns').value) || 10000,
                temp: parseFloat(document.getElementById('cfg-temp').value) || 0.4,
                top_p: parseFloat(document.getElementById('cfg-topp').value) || 0.95
            };
            google.colab.kernel.invokeFunction('karyon.apply_config', [JSON.stringify(cfg)], {});
        }
    </script>
</body>
</html>
"""


# 4. Safe Python -> JS Bridge
def py_js_call(fn_name: str, *args):
    args_json = [json.dumps(a, ensure_ascii=False) for a in args]
    display(HTML(f"<script>if(window.KaryonBridge && window.KaryonBridge.{fn_name}) {{ window.KaryonBridge.{fn_name}({', '.join(args_json)}); }}</script>"))


# 5. Async Colab Execution Entry
async def async_main():
    agent = KaggleCoREAgent()
    await agent.initialize()

    await agent.db.save_turn(
        "system",
        f"[System Event: Karyon Colab Web UI Initialized | ActiveDB: {Path(config.DB_PATH).name} | Model: {agent.key_manager.get_model()} | Keys: {len(agent.key_manager.keys)} | Tools: {len(agent.tools_map)} | Swarm: {agent.swarm_mode} | Summarizers: {', '.join(getattr(config, 'SUMMARIZER_MODELS', []))} | SlimMode: {config.SLIM_PROMPT_MODE} | MaxTurns: {config.MAX_AGENT_TURNS} | Safety: BLOCK_NONE]"
    )

    in_loop_final_rendered = False
    last_send_timestamp = 0.0

    async def on_send_message_py(user_text: str, attachments: Optional[List[Dict[str, Any]]] = None):
        nonlocal in_loop_final_rendered, last_send_timestamp
        in_loop_final_rendered = False

        if agent.is_busy:
            py_js_call("appendMessage", "msg-system", "⚠️ **Notice:** An agent task is already active. Please wait for completion or click 'Stop 🛑'.")
            py_js_call("setBusy", False)
            return

        now = time.time()
        if now - last_send_timestamp < 0.4:
            return
        last_send_timestamp = now

        if not user_text and attachments:
            user_text = f"Analyze attached file(s): {', '.join([a.get('name', 'file') for a in attachments])}"

        async def stream_event_cb(event_type: str, payload: str):
            nonlocal in_loop_final_rendered
            if event_type == "tool_start":
                py_js_call("updateLog", f"⚙ Executing: {payload}")
                py_js_call("appendMessage", "msg-tool-call", f"⚙ **Tool Call:** `{payload}`")
            elif event_type == "tool_end":
                py_js_call("updateLog", "✅ Tool completed")
                rendered_payload = render_rich_media_markdown(payload, config.PROJECT_ROOT)
                py_js_call("appendMessage", "msg-tool-res", rendered_payload)
            elif event_type == "agent_message":
                rendered_msg = render_rich_media_markdown(payload, config.PROJECT_ROOT)
                py_js_call("appendMessage", "msg-inloop", f"💬 **Karyon Agent (In-Loop):**\n\n{rendered_msg}")
            elif event_type == "final_answer":
                rendered_msg = render_rich_media_markdown(payload, config.PROJECT_ROOT)
                py_js_call("appendMessage", "msg-agent", f"### 🤖 Karyon Agent:\n\n{rendered_msg}")
                in_loop_final_rendered = True
            elif event_type == "media_album":
                py_js_call("appendMessage", "msg-agent", payload)
            elif event_type == "synthesis":
                py_js_call("updateLog", "🧠 Synthesizing analytical report...")
            elif event_type == "info":
                py_js_call("appendMessage", "msg-system", payload)

        try:
            response = await agent.process_user_query(
                user_text,
                event_callback=stream_event_cb,
                max_turns=config.MAX_AGENT_TURNS,
                attachments=attachments
            )
            py_js_call("updateLog", "")
            if not in_loop_final_rendered:
                rendered_final = render_rich_media_markdown(response, config.PROJECT_ROOT)
                py_js_call("appendMessage", "msg-agent", f"### 🤖 Karyon Agent:\n\n{rendered_final}")
            push_header_update()
            py_js_call("setBusy", False)
        except asyncio.CancelledError:
            py_js_call("updateLog", "")
            py_js_call("appendMessage", "msg-system", "🛑 Generation interrupted by user.")
            push_header_update()
            py_js_call("setBusy", False)
        except Exception as e:
            py_js_call("updateLog", "")
            py_js_call("appendMessage", "msg-agent", f"❌ **Execution Error:**\n```text\n{traceback.format_exc()}\n```")
            push_header_update()
            py_js_call("setBusy", False)

    async def on_toggle_auto_loop_py():
        if agent.autonomous_loop_active:
            if agent.autonomous_loop_paused:
                agent.resume_autonomous_loop()
                py_js_call("setAutoLoopState", "running", agent.autonomous_cycle_count)
                py_js_call("appendMessage", "msg-system", "▶️ **Autonomous Research Daemon Resumed.**")
            else:
                agent.pause_autonomous_loop()
                py_js_call("setAutoLoopState", "paused", agent.autonomous_cycle_count)
                py_js_call("appendMessage", "msg-system", "⏸️ **Autonomous Research Daemon Paused. In-flight tasks halted.**")
        else:
            py_js_call("setAutoLoopState", "running", agent.autonomous_cycle_count + 1)
            mode_tag = "Cortical Swarm (4-Stage)" if agent.swarm_mode else "Monolithic Loop"
            py_js_call("appendMessage", "msg-system", f"🚀 **Starting Continuous Autonomous Research Daemon ({mode_tag})...**")
            await agent.start_autonomous_loop(
                research_agenda="Advance Karyon-CoRE biophysical realism and empirical throughput under KEP v9.0 Master.",
                max_cycles=1000,
                interval_seconds=15.0,
                status_callback=stream_event_cb
            )

    async def on_stop_task_py():
        if agent.autonomous_loop_active:
            await agent.stop_autonomous_loop()
            py_js_call("setAutoLoopState", "stopped", 0)
        else:
            await agent.cancel_active_generation()
        agent.reset_locks()
        await agent.db.save_turn("system", "[System Event: User stopped active generation via UI]")
        py_js_call("updateLog", "")
        py_js_call("appendMessage", "msg-system", "🛑 Generation and active subprocesses stopped by user.")
        push_header_update()
        py_js_call("setBusy", False)

    async def on_get_tokens_py():
        py_js_call("updateLog", "🏷️ Auditing token utilization...")
        res = await get_context_token_status()
        py_js_call("updateLog", "")
        py_js_call("appendMessage", "msg-agent", f"```text\n{res}\n```")

    async def on_reset_keys_py():
        from karyon_agent_runtime.tools.config_tools import reset_key_cooldowns
        res = await reset_key_cooldowns()
        py_js_call("updateLog", "")
        push_header_update()
        py_js_call("appendMessage", "msg-system", f"⚡ {res}")

    async def on_sync_db_py():
        py_js_call("updateLog", "☁ Syncing SQLite state to GitHub...")
        res = await sync_agent_database("chore(db): manual Colab UI state snapshot")
        await agent.db.save_turn("system", "[System Event: User triggered manual SQLite DB sync to GitHub]")
        py_js_call("updateLog", "")
        py_js_call("appendMessage", "msg-agent", f"☁ **SQLite Database Synchronized:**\n```text\n{res}\n```")

    async def on_get_swarm_status_py():
        from karyon_agent_runtime.tools.multi_agent_tools import get_swarm_telemetry
        py_js_call('updateLog', '👥 Auditing Cortical Sub-Agents & Synaptic Traffic...')
        res = await get_swarm_telemetry()
        py_js_call('updateLog', '')
        py_js_call('appendMessage', 'msg-agent', res)

    async def on_get_status_py():
        from karyon_agent_runtime.tools.config_tools import get_runtime_config
        res = await get_runtime_config()
        py_js_call("appendMessage", "msg-agent", f"```text\n{res}\n```")

    async def on_rotate_key_py():
        from karyon_agent_runtime.tools.config_tools import force_rotate_key
        res = await force_rotate_key()
        push_header_update()
        py_js_call("appendMessage", "msg-system", f"🔄 {res}")

    async def on_rotate_model_py():
        from karyon_agent_runtime.tools.config_tools import force_rotate_model
        res = await force_rotate_model()
        push_header_update()
        py_js_call("appendMessage", "msg-system", f"🎯 {res}")

    async def on_compact_context_py():
        py_js_call("updateLog", "📦 Distilling context state...")
        res = await compress_context_now()
        py_js_call("updateLog", "")
        py_js_call("appendMessage", "msg-agent", f"```text\n{res}\n```")

    async def on_clear_history_py():
        await agent.db.clear_turns()
        await agent.db.save_turn("system", "[System Event: User cleared dialogue history]")
        py_js_call("appendMessage", "msg-system", "🗑 **Dialogue history cleared from database.**")

    def push_header_update():
        stat = agent.key_manager.get_pool_status()
        try:
            from karyon_hardware import get_hardware_engine
            hw_info = get_hardware_engine().get_telemetry()
            dev_type = hw_info.get('device_type', 'unknown').upper()
            dev_str = hw_info.get('device_str', 'unknown')
        except Exception:
            dev_type = "ACCEL"
            dev_str = "Compute Node"
        active_db_name = Path(config.DB_PATH).name
        loop_stat = agent.get_autonomous_loop_status()
        loop_tag = f"Loop: {'Active' if loop_stat['active'] else 'Idle'} (#{loop_stat['cycle']})"
        header_text = (
            f"Model: {agent.key_manager.get_model()} | "
            f"Keys: {stat['active_ready_keys']}/{stat['total_keys']} (Key #{stat['current_key_index']}) | "
            f"DB: {active_db_name} | "
            f"{loop_tag} | "
            f"Swarm: {agent.swarm_mode} | "
            f"Hardware: {dev_type} ({dev_str}) | "
            f"MaxTurns: {config.MAX_AGENT_TURNS} | "
            f"Slim: {config.SLIM_PROMPT_MODE} | "
            f"Tools: {len(agent.tools_map)}"
        )
        py_js_call("updateHeaderStats", header_text)

    async def on_apply_config_py(cfg_json_str: str):
        cfg = json.loads(cfg_json_str)
        from karyon_agent_runtime.tools.config_tools import update_runtime_config
        res = await update_runtime_config(
            model=cfg.get("model"),
            models_pool=cfg.get("models_pool"),
            summarizer_model=cfg.get("summarizer"),
            summarizer_models_pool=cfg.get("sum_pool"),
            context_compression_threshold=cfg.get("threshold"),
            max_turns=cfg.get("max_turns"),
            temperature=cfg.get("temp"),
            top_p=cfg.get("top_p"),
            thinking_level=cfg.get("thinking"),
            thinking_budget=cfg.get("budget"),
            slim_prompt_mode=cfg.get("slim", True),
            api_max_retries=cfg.get("retries", 60),
            inter_turn_delay=cfg.get("turndelay", 0.5),
            swarm_mode=cfg.get("swarm_mode", True)
        )
        push_header_update()
        py_js_call("appendMessage", "msg-system", f"⚙ **Config Live Applied & Persisted:**\n```text\n{res}\n```")

    # Register Colab Callbacks
    if IS_COLAB:
        output.register_callback('karyon.send_message', lambda text: asyncio.create_task(on_send_message_py(text)))
        output.register_callback('karyon.stop_task', lambda: asyncio.create_task(on_stop_task_py()))
        output.register_callback('karyon.toggle_auto_loop', lambda: asyncio.create_task(on_toggle_auto_loop_py()))
        output.register_callback('karyon.get_tokens', lambda: asyncio.create_task(on_get_tokens_py()))
        output.register_callback('karyon.reset_keys', lambda: asyncio.create_task(on_reset_keys_py()))
        output.register_callback('karyon.sync_db', lambda: asyncio.create_task(on_sync_db_py()))
        output.register_callback('karyon.get_status', lambda: asyncio.create_task(on_get_status_py()))
        output.register_callback('karyon.get_swarm_status', lambda: asyncio.create_task(on_get_swarm_status_py()))
        output.register_callback('karyon.rotate_key', lambda: asyncio.create_task(on_rotate_key_py()))
        output.register_callback('karyon.rotate_model', lambda: asyncio.create_task(on_rotate_model_py()))
        output.register_callback('karyon.compact_context', lambda: asyncio.create_task(on_compact_context_py()))
        output.register_callback('karyon.clear_history', lambda: asyncio.create_task(on_clear_history_py()))
        output.register_callback('karyon.apply_config', lambda cfg: asyncio.create_task(on_apply_config_py(cfg)))

    display(HTML(HTML_FULL_PANEL))
    push_header_update()

    # Restore full history on startup in a single batch
    try:
        recent_turns = await agent.db.get_recent_turns(limit=40, include_tools=True)
        if recent_turns:
            history_batch = []
            for t in recent_turns:
                role = t.get("role", "user")
                text_val = t.get("text", "")
                rendered_text = render_rich_media_markdown(text_val, config.PROJECT_ROOT)

                if role == "user":
                    history_batch.append({"type": "msg-user", "text": text_val})
                elif role == "tool_call":
                    history_batch.append({"type": "msg-tool-call", "text": f"⚙ **Tool Call:** {rendered_text}"})
                elif role == "tool_result":
                    history_batch.append({"type": "msg-tool-res", "text": rendered_text})
                elif role == "agent_message":
                    history_batch.append({"type": "msg-inloop", "text": f"💬 **Karyon Agent:**\n\n{rendered_text}"})
                elif role == "system":
                    history_batch.append({"type": "msg-system", "text": rendered_text})
                else:
                    history_batch.append({"type": "msg-agent", "text": f"### 🤖 Karyon Agent:\n\n{rendered_text}"})

            py_js_call("restoreHistory", history_batch)
    except Exception as err:
        py_js_call("appendMessage", "msg-system", f"⚠ History restoration notice: {str(err)}")

    print("🟢 Karyon CoRE Web Panel (Colab Suite v31.0 | Unified Process Runtime) fully active!")
    while True:
        await asyncio.sleep(1)


def main():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        try:
            nest_asyncio.apply()
            return loop.run_until_complete(async_main())
        except Exception:
            return asyncio.create_task(async_main())
    else:
        return loop.run_until_complete(async_main())


if __name__ == "__main__":
    main()
