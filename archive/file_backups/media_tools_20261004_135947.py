# karyon_agent_runtime/tools/media_tools.py
"""
===============================================================================
MULTIMEDIA & SCIENTIFIC PLOT EMBEDDING TOOLKIT
Renders Single Plots, Multi-Image Series, and Interactive HTML Media Albums.
===============================================================================
"""

import base64
import logging
from pathlib import Path
from typing import List, Optional
import karyon_agent_runtime.config as config

logger = logging.getLogger("ProxyAgent.Media")

MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml"
}


def file_to_base64_data_uri(filepath: Path, max_dim: int = 850, quality: int = 82) -> Optional[str]:
    """Reads a local image, resizes if large, and encodes into a lightweight base64 data URI."""
    if not filepath.exists() or not filepath.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            if getattr(img, "is_animated", False):
                with open(filepath, "rb") as f:
                    return f"data:image/gif;base64,{base64.b64encode(f.read()).decode('utf-8')}"

            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

            import io
            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                mime = "image/jpeg"

            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:{mime};base64,{b64_data}"
    except Exception:
        ext = filepath.suffix.lower().replace(".", "")
        mime = MIME_MAP.get(ext, "image/png")
        try:
            with open(filepath, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64_data}"
        except Exception:
            return None


async def embed_media(file_paths: List[str], captions: List[str] = None, layout: str = "grid") -> str:
    """
    Embeds one or more local images, plots, diagrams, or charts into the dialogue response as a styled media album.

    Args:
        file_paths: List of relative or absolute paths to local image files (e.g. ['experiments/loss_plot.png', 'experiments/vram.png']).
        captions: Optional list of descriptive captions for each image.
        layout: Visual layout mode: 'grid' (2-column responsive gallery), 'single' (full-width banner), 'carousel' (horizontal scrolling strip), or 'column' (vertical stack).
    """
    if not file_paths:
        return "Error: No file_paths provided to embed_media."

    captions = captions or []
    encoded_items = []
    root = config.PROJECT_ROOT

    for idx, fpath_str in enumerate(file_paths):
        path = (root / fpath_str).resolve()
        caption = captions[idx] if idx < len(captions) else f"Figure #{idx + 1}: {Path(fpath_str).name}"

        if path.exists() and path.is_file():
            data_uri = file_to_base64_data_uri(path)
            if data_uri:
                encoded_items.append({"src": data_uri, "caption": caption, "filename": path.name, "path": str(fpath_str)})
            else:
                encoded_items.append({"src": None, "caption": f"[Error reading: {fpath_str}]", "filename": path.name, "path": str(fpath_str)})
        elif fpath_str.startswith(("http://", "https://", "data:")):
            encoded_items.append({"src": fpath_str, "caption": caption, "filename": "Remote Image", "path": str(fpath_str)})
        else:
            encoded_items.append({"src": None, "caption": f"[File Not Found: {fpath_str}]", "filename": Path(fpath_str).name, "path": str(fpath_str)})

    valid_count = sum(1 for it in encoded_items if it["src"] is not None)
    if valid_count == 0:
        return f"Error: None of the specified media files could be found or read: {file_paths}"

    # Build Dark-Themed Responsive HTML Album Component
    cards = []
    for item in encoded_items:
        if item["src"]:
            card_html = (
                '<div style="flex: 1 1 300px; max-width: 520px; min-width: 260px; background: #1e1e2e; border: 1px solid #313244; border-radius: 10px; padding: 10px; box-sizing: border-box; box-shadow: 0 4px 12px rgba(0,0,0,0.45); text-align: center;">\n'
                f'  <img src="{item["src"]}" alt="{item["caption"]}" style="width: 100%; height: auto; border-radius: 6px; display: block; object-fit: contain; max-height: 380px; background: #11111b;" />\n'
                f'  <div style="margin-top: 8px; font-size: 0.85em; color: #cdd6f4; font-weight: 600; padding: 2px 4px;">{item["caption"]}</div>\n'
                '</div>'
            )
            cards.append(card_html)
        else:
            cards.append(f'<div style="flex: 1 1 260px; max-width: 400px; background: #2a1b1b; border: 1px solid #f38ba8; border-radius: 8px; padding: 12px; text-align: center; color: #f38ba8;">⚠️ <b>Media Error:</b> {item["caption"]}</div>')

    flex_wrap = "wrap" if layout in ["grid", "gallery", "column"] else "nowrap"
    overflow_x = "auto" if layout in ["carousel", "horizontal"] else "visible"
    flex_dir = "column" if layout == "column" else "row"

    cards_joined = "\n".join(cards)
    album_html = (
        f'<div class="karyon-media-album" style="display: flex; flex-direction: {flex_dir}; flex-wrap: {flex_wrap}; overflow-x: {overflow_x}; gap: 14px; margin: 14px 0; justify-content: center; padding: 4px;">\n'
        f'{cards_joined}\n'
        '</div>'
    )

    # Emit real-time media event to active UI if running
    from karyon_agent_runtime.agent_core import get_active_agent
    agent = get_active_agent()
    if agent and agent.active_event_callback:
        import asyncio
        asyncio.create_task(agent.active_event_callback("media_album", album_html))

    return f"Successfully generated media album ({valid_count} figures):\n{album_html}"
