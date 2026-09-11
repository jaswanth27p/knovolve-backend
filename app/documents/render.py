"""Markdown-to-PDF rendering for exports and generated course documents."""
import base64
import logging
import os
import sys
import urllib.request
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

if sys.platform == "darwin" and "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ:
    for _prefix in ("/opt/homebrew/lib", "/usr/local/lib"):
        if os.path.isdir(_prefix):
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = _prefix
            break

from weasyprint import HTML

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)


def markdown_to_html(body_markdown: str) -> str:
    # A fresh Markdown instance per call — no shared mutable converter state
    # across concurrent Celery task execution.
    return markdown.markdown(body_markdown or "", extensions=["fenced_code", "tables"])


def diagram_data_uri(url: str | None) -> str | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        encoded = base64.b64encode(raw).decode("ascii")
    except Exception:
        logger.warning("failed to fetch/encode diagram at %r; omitting", url, exc_info=True)
        return None
    return f"data:image/svg+xml;base64,{encoded}"


def _render_template(template_name: str, payload: dict) -> str:
    return _ENV.get_template(template_name).render(**payload)


def _write_pdf(payload: dict, document_kind: str) -> bytes:
    source = _render_template("export.html.j2", {**payload, "document_kind": document_kind})
    pdf = HTML(string=source).write_pdf()
    assert pdf is not None
    return pdf


def render_course_pdf(payload: dict) -> bytes:
    return _write_pdf(payload, "course")


def render_assignments_pdf(payload: dict) -> bytes:
    return _write_pdf(payload, "assignments")


def render_custom_pdf(payload: dict) -> bytes:
    return _write_pdf(payload, "custom")
