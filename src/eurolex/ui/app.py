"""Streamlit UI. Run with `make ui` (the API must be running: `make api`).

The chat history lives only in this browser session; nothing is saved or logged.
"""

import html
from datetime import datetime
from typing import Any

import streamlit as st

from eurolex.config import get_settings
from eurolex.ui.client import Reply, ask, is_up
from eurolex.ui.render import answer_html, seconds

EXAMPLES = [
    "How long does a company have to answer my access request?",
    "Must an organisation report every personal data breach?",
    "When does an organisation have to appoint a data protection officer?",
    "Can a company make a significant decision about me using only automated processing?",
]

CSS = """
<style>
:root {
  --accent: #5b4bdb; --accent-soft: #eeebfc; --ink: #1d1b2c; --muted: #6b6880;
  --line: #e7e5f0; --card: #ffffff; --page: #fafafd; --good: #e8f6ec; --good-ink: #23733f;
}
.stApp { background: var(--page); }
.block-container { padding-top: 2rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .6rem; font-size: 1.35rem;
  font-weight: 700; color: var(--ink); margin-bottom: 1.2rem; }
.brand .logo { width: 36px; height: 36px; border-radius: 10px; background: var(--accent);
  color: #fff; display: grid; place-items: center; font-size: 1.1rem; }
.side-label { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); margin: 1.2rem 0 .4rem; }
.corpus { border: 1px solid var(--line); border-radius: 12px; padding: .7rem .8rem;
  background: var(--accent-soft); color: var(--ink); font-size: .9rem; }
.corpus small { color: var(--muted); }
.header h1 { font-size: 1.6rem; margin: 0; color: var(--ink); }
.header p { color: var(--muted); margin: .2rem 0 0; }
.pill { display: inline-flex; align-items: center; gap: .45rem; border: 1px solid var(--line);
  border-radius: 999px; padding: .35rem .85rem; background: #fff; font-size: .85rem;
  color: var(--ink); float: right; margin-top: .4rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.on { background: #22a052; } .dot.off { background: #d64545; }
.banner { border: 1px solid #f0dca0; background: #fff8e3; color: #6b5208; border-radius: 12px;
  padding: .6rem .9rem; font-size: .88rem; margin: 1rem 0 1.2rem; }
.q { background: var(--accent-soft); border: 1px solid #dcd6fa; border-radius: 14px;
  padding: .9rem 1.1rem; color: var(--ink); margin: 0 0 .8rem 2.5rem; }
.q time, .a time { float: right; color: var(--muted); font-size: .75rem; }
.a { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1rem 1.2rem; margin-bottom: 1.4rem; box-shadow: 0 1px 2px rgba(20,16,60,.04); }
.a .who { font-weight: 600; color: var(--ink); margin-bottom: .5rem; }
.a .who .logo { display: inline-grid; place-items: center; width: 22px; height: 22px;
  border-radius: 6px; background: var(--accent); color: #fff; font-size: .7rem;
  margin-right: .4rem; }
.a.refused { border-color: #f0dca0; background: #fffdf6; }
.a .body { color: var(--ink); line-height: 1.6; }
.cite { display: inline-grid; place-items: center; min-width: 1.2rem; height: 1.2rem;
  padding: 0 .25rem; border-radius: 6px; background: var(--accent-soft); color: var(--accent);
  font-size: .72rem; font-weight: 700; vertical-align: super; margin: 0 1px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--line);
  border-radius: 12px; margin-top: .9rem; }
.stats div { padding: .55rem .8rem; border-right: 1px solid var(--line); }
.stats div:last-child { border-right: 0; }
.stats small { display: block; color: var(--muted); font-size: .72rem; }
.stats b { font-size: 1.05rem; color: var(--ink); }
.panel-title { font-weight: 700; color: var(--ink); margin: 0 0 .6rem; }
.src { display: flex; gap: .6rem; align-items: flex-start; }
.src .n { color: var(--muted); font-weight: 600; min-width: 1.1rem; }
.src .t { flex: 1; color: var(--ink); font-size: .92rem; }
.src .t small { display: block; color: var(--muted); }
.score { background: var(--good); color: var(--good-ink); border-radius: 8px;
  padding: .1rem .45rem; font-size: .78rem; font-weight: 600; }
.empty { color: var(--muted); font-size: .9rem; }
@media (max-width: 640px) { .stats { grid-template-columns: repeat(2, 1fr); }
  .q { margin-left: 0; } }
</style>
"""

settings = get_settings()
st.set_page_config(page_title="EuroLex-RAG", page_icon="⚖️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

history: list[dict[str, Any]] = st.session_state.setdefault("history", [])

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="logo">§</span>EuroLex-RAG</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="side-label">Corpus</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="corpus"><b>GDPR</b><br><small>Regulation (EU) 2016/679, '
        "articles and recitals, English</small></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">Try a question</div>', unsafe_allow_html=True)
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example
    st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        history.clear()
    st.caption("Questions are not stored. The chat disappears when you close the tab.")

up = is_up(settings.api_url)
dot = "on" if up else "off"
status = f'<span class="dot {dot}"></span>{"Connected" if up else "API offline"}'
st.markdown(
    f'<span class="pill">{status}</span><div class="header"><h1>GDPR Assistant</h1>'
    "<p>Answers from the official EU text, with a citation for every claim</p></div>"
    '<div class="banner">General information about the GDPR, not legal advice. Check the '
    "cited articles, and ask a lawyer about your own situation.</div>",
    unsafe_allow_html=True,
)

question = st.chat_input(
    "Ask a question about the GDPR", max_chars=settings.max_question_chars
) or st.session_state.pop("pending", None)

chat, sources = st.columns([3, 2], gap="large")


def answer_card(reply: Reply, when: str) -> str:
    if not reply.ok:
        return (
            f'<div class="a refused"><div class="who">Error<time>{when}</time></div>'
            f'<div class="body">{html.escape(reply.error)}</div></div>'
        )
    lat = reply.latency_ms
    stats = (
        '<div class="stats">'
        f"<div><small>Sources cited</small><b>{len(reply.citations)}</b></div>"
        f"<div><small>Passages read</small><b>{reply.passages or '-'}</b></div>"
        f"<div><small>Search</small><b>{seconds(lat.get('retrieval', 0))}</b></div>"
        f"<div><small>Answer</small><b>{seconds(lat.get('generation', 0))}</b></div>"
        "</div>"
    )
    kind = "a refused" if reply.refused else "a"
    label = "No answer" if reply.refused else "EuroLex-RAG answer"
    return (
        f'<div class="{kind}"><div class="who"><span class="logo">§</span>{label}'
        f'<time>{when}</time></div><div class="body">\n\n{answer_html(reply.answer)}\n\n'
        f"</div>{'' if reply.refused else stats}</div>"
    )


with chat:
    if not history:
        st.markdown(
            '<div class="a"><div class="who"><span class="logo">§</span>Welcome</div>'
            '<div class="body">Ask about rights, obligations or definitions in the GDPR, '
            "for example how fast a company must answer an access request. Each answer "
            "cites the articles it is based on.</div></div>",
            unsafe_allow_html=True,
        )
    for turn in history:
        when = turn["time"].strftime("%H:%M")
        st.markdown(
            f'<div class="q">{html.escape(turn["question"])}<time>{when}</time></div>',
            unsafe_allow_html=True,
        )
        st.markdown(answer_card(turn["reply"], when), unsafe_allow_html=True)
    if question and question.strip():
        if settings.api_key is None:
            st.error("API_KEY is not set. Copy .env.example to .env and fill it in.")
            st.stop()
        # show the question straight away; the answer can take a minute on a laptop GPU
        st.markdown(f'<div class="q">{html.escape(question)}</div>', unsafe_allow_html=True)
        with st.spinner("Searching the GDPR and writing an answer (this can take a minute)..."):
            reply = ask(
                question, api_url=settings.api_url, api_key=settings.api_key.get_secret_value()
            )
        history.append({"question": question, "reply": reply, "time": datetime.now()})
        st.rerun()  # redraw so the sources panel shows this answer

with sources:
    latest: Reply | None = history[-1]["reply"] if history else None
    with st.container(border=True):
        st.markdown('<div class="panel-title">Supporting sources</div>', unsafe_allow_html=True)
        if latest is None or not latest.citations:
            message = "Sources for the latest answer appear here."
            if latest is not None and latest.ok and latest.refused:
                message = "No sources: the question was not answered."
            st.markdown(f'<div class="empty">{message}</div>', unsafe_allow_html=True)
        else:
            for c in latest.citations:
                title = html.escape(c["article_title"] or c["article_id"])
                score = c.get("score")
                badge = (
                    '<span class="score" title="Retrieval similarity, not a probability">'
                    f"{score:.2f}</span>"
                    if isinstance(score, int | float)
                    else ""
                )
                st.markdown(
                    f'<div class="src"><span class="n">{c["n"]}</span><span class="t">'
                    f"{title}<small>{html.escape(c['article_id'])}</small></span>{badge}</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Show text"):
                    st.text(c["text"])
                    st.markdown(f"[Open on EUR-Lex]({c['url']})")
