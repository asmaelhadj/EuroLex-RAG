"""HTML bits for the UI. Model output is escaped before it goes into the page."""

import html
import re

_MARKER = re.compile(r"\[(\d+)\]")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def answer_html(text: str) -> str:
    """Escape the answer, then turn [n] markers into badges.

    The result is rendered as markdown with HTML allowed, so escaping first stops the model
    (or a passage it copied) from adding tags. Markdown images are dropped so an answer
    cannot make the browser load an outside URL.
    """
    safe = html.escape(_IMAGE.sub(r"\1", text), quote=False)
    return _MARKER.sub(r'<span class="cite">\1</span>', safe)


def seconds(ms: int | float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 100 else f"{ms:.0f} ms"
