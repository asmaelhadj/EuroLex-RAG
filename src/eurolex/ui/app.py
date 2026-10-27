"""Streamlit UI. Run with `make ui` (the API must be running: `make api`)."""

import streamlit as st

from eurolex.config import get_settings
from eurolex.ui.client import ask

settings = get_settings()

st.set_page_config(page_title="EuroLex-RAG", page_icon="⚖️")
st.title("EuroLex-RAG")
st.warning(
    "This tool gives general information about the GDPR, not legal advice. "
    "Check the cited articles, and ask a lawyer for advice on your situation."
)

with st.form("ask"):
    question = st.text_area(
        "Your question about the GDPR",
        max_chars=settings.max_question_chars,
        placeholder="How long does a company have to answer my access request?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    if settings.api_key is None:
        st.error("API_KEY is not set. Copy .env.example to .env and fill it in.")
        st.stop()
    with st.spinner("Searching the GDPR and writing an answer (this can take a minute)..."):
        reply = ask(question, api_url=settings.api_url, api_key=settings.api_key.get_secret_value())
    if not reply.ok:
        st.error(reply.error)
    elif reply.refused:
        st.info(reply.answer)
    else:
        st.markdown(reply.answer)
        if reply.citations:
            st.subheader("Sources")
            for c in reply.citations:
                title = f"[{c['n']}] {c['article_id']}"
                if c["article_title"]:
                    title += f": {c['article_title']}"
                with st.expander(title):
                    st.text(c["text"])
                    st.markdown(f"[Open on EUR-Lex]({c['url']})")
