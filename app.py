"""Zen local RAG studio — Streamlit UI."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag import OllamaClient, RAGEngine
from rag.ollama_client import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "data" / "chroma_db_app"

st.set_page_config(
    page_title="zen · local rag",
    page_icon="◇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ZEN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Outfit:wght@300;400;500&display=swap');

:root {
  --zen-ink: #1f241f;
  --zen-muted: #5f695f;
  --zen-line: #c9cfc8;
  --zen-paper: #f7f8f6;
  --zen-btn: #2a322a;
  --zen-btn-text: #f4f5f3;
}

html, body, .stApp, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  font-size: 18px;
  color: var(--zen-ink) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, #e8ece8 0%, transparent 55%),
    linear-gradient(180deg, #f4f5f3 0%, #eceeea 100%);
}

.block-container {
  padding-top: 3.5rem;
  padding-bottom: 4rem;
  max-width: 760px;
}

p, li, label, .stMarkdown, .stCaption,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] {
  color: var(--zen-ink) !important;
  font-size: 1.05rem !important;
  line-height: 1.65 !important;
}

.zen-brand {
  font-family: 'Cormorant Garamond', serif;
  font-size: 3.8rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-align: center;
  margin: 0 0 0.35rem 0;
  color: var(--zen-ink) !important;
}

.zen-sub {
  text-align: center;
  font-weight: 300;
  font-size: 1.15rem !important;
  color: var(--zen-muted) !important;
  margin-bottom: 2.75rem;
  letter-spacing: 0.04em;
}

.zen-rule {
  border: none;
  border-top: 1px solid var(--zen-line);
  margin: 1.75rem 0 1.5rem 0;
}

.zen-label {
  font-size: 0.9rem !important;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--zen-muted) !important;
  margin-bottom: 0.65rem;
}

.zen-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.9rem 0;
  border-bottom: 1px solid #d5dad4;
  font-size: 1.05rem !important;
  color: var(--zen-ink) !important;
}

.zen-status .muted { color: var(--zen-muted) !important; font-weight: 300; }

.zen-pill {
  display: inline-block;
  padding: 0.25rem 0.65rem;
  border: 1px solid #b7c0b6;
  border-radius: 2px;
  font-size: 0.9rem !important;
  letter-spacing: 0.06em;
  color: #4a554a !important;
  background: transparent;
}
.zen-pill.ok { border-color: #7d947d; color: #3d5a3d !important; }
.zen-pill.warn { border-color: #b8a88a; color: #6e5f3d !important; }

/* --- Inputs --- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: #ffffff !important;
  border: 1px solid var(--zen-line) !important;
  border-radius: 2px !important;
  color: var(--zen-ink) !important;
  caret-color: var(--zen-ink) !important;
  font-size: 1.1rem !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder {
  color: #7a847a !important;
  opacity: 1 !important;
}

/* --- Buttons (incl. form submit) --- */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
button[kind="primaryFormSubmit"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--zen-btn) !important;
  color: var(--zen-btn-text) !important;
  border: 1px solid var(--zen-btn) !important;
  border-radius: 2px !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-size: 0.95rem !important;
  padding: 0.7rem 1.35rem !important;
}
.stButton > button *,
div[data-testid="stFormSubmitButton"] > button *,
button[kind="primary"] *,
button[kind="primaryFormSubmit"] *,
button[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primaryFormSubmit"] * {
  color: var(--zen-btn-text) !important;
  fill: var(--zen-btn-text) !important;
}
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  opacity: 0.9 !important;
  border-color: var(--zen-btn) !important;
  color: var(--zen-btn-text) !important;
}
.stButton > button:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled {
  background: #9aa39a !important;
  color: #f4f5f3 !important;
  border-color: #9aa39a !important;
  opacity: 1 !important;
}

/* --- File uploader dropzone --- */
div[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed #9aa69a !important;
  background: var(--zen-paper) !important;
  border-radius: 2px !important;
}
div[data-testid="stFileUploaderDropzone"] *,
section[data-testid="stFileUploaderDropzone"] * {
  color: var(--zen-ink) !important;
  fill: var(--zen-ink) !important;
}
div[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button {
  background: #eef0ec !important;
  color: var(--zen-ink) !important;
  border: 1px solid #a8b0a7 !important;
}

/* Uploaded file chips: light cards, dark readable names */
div[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileData"],
li[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] ul li,
div[data-testid="stFileUploader"] [class*="uploadedFile"] {
  background: #ffffff !important;
  border: 1px solid var(--zen-line) !important;
  border-radius: 4px !important;
  color: var(--zen-ink) !important;
}
div[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileData"] *,
li[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileName"] *,
div[data-testid="stFileUploader"] ul li *,
div[data-testid="stFileUploader"] [class*="uploadedFile"] *,
small[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderDeleteBtn"] svg {
  color: var(--zen-ink) !important;
  fill: var(--zen-ink) !important;
  background: transparent !important;
}

/* Fallback if Streamlit keeps a dark chip: force light text */
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] ~ * span,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] ~ * small {
  color: var(--zen-ink) !important;
}

.zen-answer {
  margin-top: 1.25rem;
  padding: 1.1rem 0 0.25rem 0;
  border-top: 1px solid var(--zen-line);
  color: var(--zen-ink) !important;
}
.zen-answer .q {
  font-size: 0.85rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--zen-muted) !important;
  margin-bottom: 0.35rem;
}
.zen-answer .q-text,
.zen-answer .a-text {
  color: var(--zen-ink) !important;
  font-size: 1.12rem !important;
  line-height: 1.7 !important;
  margin: 0 0 1rem 0;
}
.zen-answer .meta {
  font-size: 0.95rem !important;
  color: var(--zen-muted) !important;
  font-weight: 300;
}

code, .stCode, pre { font-size: 0.98rem !important; color: var(--zen-ink) !important; }

/* --- Sliders --- */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p {
  color: var(--zen-ink) !important;
  font-size: 0.98rem !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--zen-btn) !important;
}

/* --- Checkboxes --- */
div[data-testid="stCheckbox"] label p {
  color: var(--zen-ink) !important;
  font-size: 1.02rem !important;
}

.zen-chunk {
  border-top: 1px solid var(--zen-line);
  padding: 0.85rem 0 0.4rem 0;
  margin-top: 0.5rem;
}
.zen-chunk .meta {
  font-size: 0.85rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--zen-muted) !important;
  margin-bottom: 0.35rem;
}
.zen-chunk .body {
  color: var(--zen-ink) !important;
  font-size: 1rem !important;
  line-height: 1.55 !important;
  margin: 0 0 0.45rem 0;
  white-space: pre-wrap;
}
.zen-chunk .vec {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.82rem !important;
  color: var(--zen-muted) !important;
  word-break: break-all;
  margin: 0;
}

#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
div[data-testid="stStatusWidget"] { visibility: hidden; }
"""


def _default_ollama_url() -> str:
    try:
        secret = st.secrets.get("OLLAMA_BASE_URL", "")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    env = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if env:
        return env
    return f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


def _init_state() -> None:
    defaults = {
        "ollama_url": _default_ollama_url(),
        "ollama_host": DEFAULT_HOST,
        "ollama_port": DEFAULT_PORT,
        "ollama_status": None,
        "llm_role": "You are a calm, precise research assistant. Answer only from the document context.",
        "indexed_files": [],
        "index_preview": [],
        "index_modes_used": [],
        "last_turn": None,
        "ready": False,
        "chunk_size": 600,
        "chunk_overlap": 200,
        "index_mode": "vector",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Drop any leftover multi-turn history from earlier versions
    st.session_state.pop("messages", None)
    st.session_state.pop("last_upload_key", None)
    st.session_state.pop("show_chunks_dialog", None)
    st.session_state.pop("opt_vector", None)
    st.session_state.pop("opt_bm25", None)
    st.session_state.pop("opt_hybrid", None)


def _get_engine(client: OllamaClient) -> RAGEngine:
    existing = st.session_state.get("rag_engine")
    # Recreate after code reload — old instances lack new ingest kwargs
    if not isinstance(existing, RAGEngine):
        st.session_state.rag_engine = RAGEngine(
            chroma_dir=CHROMA_DIR,
            ollama=client,
        )
    else:
        existing.ollama = client
    return st.session_state.rag_engine


def _choose_index_mode(mode: str) -> None:
    key = f"chk_{mode}"
    if st.session_state.get(key):
        st.session_state.index_mode = mode
    else:
        # Keep exactly one option selected
        st.session_state[key] = True


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _format_vector(values: list[float] | None, *, preview: int = 12) -> str:
    if not values:
        return "No dense vector (BM-25 / keyword index only)"
    head = ", ".join(f"{v:.4f}" for v in values[:preview])
    more = len(values) - preview
    suffix = f", … (+{more} more)" if more > 0 else ""
    return f"[{head}{suffix}]  ·  dim={len(values)}"


@st.dialog("Chunks & vectors", width="large")
def _chunks_dialog() -> None:
    preview = st.session_state.get("index_preview") or []
    modes = st.session_state.get("index_modes_used") or []
    mode_label = ", ".join(modes) if modes else "—"
    st.caption(f"{len(preview)} chunk(s) · indexing: {mode_label}")

    if not preview:
        st.info("No chunks yet. Index documents first.")
        return

    for i, item in enumerate(preview, start=1):
        meta = item.get("metadata") or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        header = (
            f"Chunk {i} · {_escape(str(source))} · page {_escape(str(page))} · "
            f"id {_escape(str(item.get('id', '')))}"
        )
        body = _escape(item.get("text") or "")
        vec = _escape(_format_vector(item.get("embedding")))
        st.markdown(
            f"""
            <div class="zen-chunk">
              <div class="meta">{header}</div>
              <p class="body">{body}</p>
              <p class="vec">{vec}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    _init_state()
    st.markdown(f"<style>{ZEN_CSS}</style>", unsafe_allow_html=True)

    st.markdown('<p class="zen-brand">zen</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="zen-sub">local document quiet · llama3.2:1b</p>',
        unsafe_allow_html=True,
    )

    # --- 1. Ollama connection ---
    st.markdown('<div class="zen-label">Model</div>', unsafe_allow_html=True)
    st.caption("Local: http://127.0.0.1:11434 · Cloud: paste a public Ollama URL (e.g. ngrok).")
    c1, c2 = st.columns([4, 1])
    with c1:
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            label_visibility="collapsed",
            placeholder="http://127.0.0.1:11434",
        )
    with c2:
        connect = st.button("Connect", use_container_width=True)

    if connect:
        st.session_state.ollama_url = ollama_url.strip()
        client = OllamaClient.from_base_url(
            st.session_state.ollama_url,
            model=DEFAULT_MODEL,
        )
        st.session_state.ollama_host = client.host
        st.session_state.ollama_port = client.port
        status = client.check()
        st.session_state.ollama_status = status
        if status.connected and status.model_ready:
            st.session_state.ready = True
            st.session_state.pop("rag_engine", None)
            _get_engine(client)
        else:
            st.session_state.ready = False

    status = st.session_state.ollama_status
    if status is None:
        st.markdown(
            '<div class="zen-status"><span class="muted">Run llama3.2:1b locally, then connect</span>'
            '<span class="zen-pill">idle</span></div>',
            unsafe_allow_html=True,
        )
        st.code(
            "ollama serve\n"
            "ollama pull llama3.2:1b\n\n"
            "# Expose Ollama for Streamlit Cloud (required host-header):\n"
            'ngrok http 11434 --host-header="localhost:11434"',
            language="bash",
        )
    else:
        pill_class = "ok" if status.model_ready else "warn"
        pill_text = "ready" if status.model_ready else "waiting"
        st.markdown(
            f'<div class="zen-status">'
            f"<span>{status.message}</span>"
            f'<span class="zen-pill {pill_class}">{pill_text}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="zen-status">'
            f'<span class="muted">endpoint</span>'
            f"<span><code>{status.host}:{status.port}</code></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="zen-rule" />', unsafe_allow_html=True)

    # --- 2. LLM role ---
    st.markdown('<div class="zen-label">LLM role</div>', unsafe_allow_html=True)
    st.caption("Used as the system prompt after the query is rewritten for retrieval.")
    role = st.text_area(
        "Role",
        value=st.session_state.llm_role,
        height=100,
        label_visibility="collapsed",
        placeholder="Who should the model be while answering?",
    )
    st.session_state.llm_role = role

    st.markdown('<hr class="zen-rule" />', unsafe_allow_html=True)

    # --- 3. PDF upload & indexing ---
    st.markdown('<div class="zen-label">Documents</div>', unsafe_allow_html=True)

    chunk_size = st.slider(
        "Select Chunk Size",
        min_value=100,
        max_value=2000,
        value=int(st.session_state.chunk_size),
        step=50,
        disabled=not st.session_state.ready,
    )
    chunk_overlap = st.slider(
        "Select Chunk Overlap Size",
        min_value=0,
        max_value=min(1000, max(chunk_size - 1, 0)),
        value=min(int(st.session_state.chunk_overlap), max(chunk_size - 1, 0)),
        step=25,
        disabled=not st.session_state.ready,
    )
    st.session_state.chunk_size = int(chunk_size)
    st.session_state.chunk_overlap = int(chunk_overlap)

    st.caption("Choose how documents should be indexed (one only).")
    mode = st.session_state.index_mode
    st.session_state.chk_vector = mode == "vector"
    st.session_state.chk_bm25 = mode == "bm25"
    st.session_state.chk_hybrid = mode == "hybrid"

    c_vec, c_bm25, c_hyb = st.columns(3)
    with c_vec:
        st.checkbox(
            "Vector Indexing",
            key="chk_vector",
            disabled=not st.session_state.ready,
            on_change=_choose_index_mode,
            args=("vector",),
        )
    with c_bm25:
        st.checkbox(
            "Non Vector Indexing",
            key="chk_bm25",
            disabled=not st.session_state.ready,
            on_change=_choose_index_mode,
            args=("bm25",),
        )
    with c_hyb:
        st.checkbox(
            "Hybrid Indexing",
            key="chk_hybrid",
            disabled=not st.session_state.ready,
            on_change=_choose_index_mode,
            args=("hybrid",),
        )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not st.session_state.ready,
        help="Drop one or more PDF files, then click Index Documents",
    )

    if not st.session_state.ready:
        st.caption("Connect to Ollama before uploading.")
    elif not uploaded_files:
        st.caption("Upload one or more PDFs, then index them.")
    else:
        st.caption(f"{len(uploaded_files)} file(s) ready · not indexed until you click below.")

    modes = {st.session_state.index_mode}

    can_index = st.session_state.ready and bool(uploaded_files)
    index_clicked = st.button(
        "Index Documents",
        use_container_width=True,
        disabled=not can_index,
    )

    if index_clicked and can_index and uploaded_files:
        client = OllamaClient.from_base_url(
            st.session_state.ollama_url,
            model=DEFAULT_MODEL,
        )
        # Drop stale engine from earlier app versions
        st.session_state.pop("rag_engine", None)
        engine = _get_engine(client)
        indexed = []
        with st.spinner(f"Indexing {len(uploaded_files)} PDF(s)…"):
            for i, pdf in enumerate(uploaded_files):
                info = engine.ingest_pdf_bytes(
                    pdf.name,
                    pdf.getvalue(),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    modes=modes,
                    clear_first=(i == 0),
                )
                indexed.append(info)
        st.session_state.indexed_files = indexed
        st.session_state.index_preview = engine.get_preview()
        st.session_state.index_modes_used = sorted(modes)
        st.session_state.last_turn = None
        total_pages = sum(i["pages"] for i in indexed)
        total_chunks = sum(i["chunks"] for i in indexed)
        names = ", ".join(i["file_name"] for i in indexed)
        mode_label = ", ".join(sorted(modes))
        st.success(
            f"{len(indexed)} file(s) · {total_pages} pages · {total_chunks} chunks · "
            f"{mode_label} — {names}"
        )
    elif st.session_state.indexed_files:
        total_pages = sum(i["pages"] for i in st.session_state.indexed_files)
        total_chunks = sum(i["chunks"] for i in st.session_state.indexed_files)
        names = ", ".join(i["file_name"] for i in st.session_state.indexed_files)
        mode_label = ", ".join(st.session_state.index_modes_used) or "—"
        st.caption(
            f"Indexed · {len(st.session_state.indexed_files)} file(s) · "
            f"{total_pages} pages · {total_chunks} chunks · {mode_label} — {names}"
        )

    if st.session_state.indexed_files and st.session_state.index_preview:
        if st.button("Show Chunks & Vectors Generated", use_container_width=True):
            _chunks_dialog()

    st.markdown('<hr class="zen-rule" />', unsafe_allow_html=True)

    # --- 4. Ask (single-turn) ---
    st.markdown('<div class="zen-label">Ask</div>', unsafe_allow_html=True)

    can_ask = st.session_state.ready and bool(st.session_state.indexed_files)

    if not st.session_state.ready:
        st.caption("Connect to Ollama first.")
    elif not st.session_state.indexed_files:
        st.caption("Upload PDFs and click Index Documents, then ask below.")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Question",
            height=110,
            label_visibility="collapsed",
            placeholder="Ask something from the documents…",
            disabled=not can_ask,
        )
        asked = st.form_submit_button(
            "Ask",
            use_container_width=True,
            disabled=not can_ask,
        )

    if asked and question and question.strip() and can_ask:
        question = question.strip()
        client = OllamaClient.from_base_url(
            st.session_state.ollama_url,
            model=DEFAULT_MODEL,
        )
        engine = _get_engine(client)

        with st.spinner("Rewriting · retrieving · answering…"):
            result = engine.ask(question, role=st.session_state.llm_role)

        source_bits = []
        for hit in result["sources"][:3]:
            meta = hit["metadata"]
            source_bits.append(
                f"{meta.get('source', 'doc')} p.{meta.get('page', '?')}"
            )
        mode = getattr(engine, "retrieval_mode", None) or "vector"
        meta_line = f"rewritten · {result['rewritten_query']}  ·  {mode}"
        if source_bits:
            meta_line += "  ·  " + " · ".join(source_bits)

        # Replace previous turn — no history kept
        st.session_state.last_turn = {
            "question": question,
            "answer": result["answer"],
            "meta": meta_line,
        }

    turn = st.session_state.last_turn
    if turn:
        st.markdown(
            f"""
            <div class="zen-answer">
              <div class="q">Question</div>
              <p class="q-text">{_escape(turn["question"])}</p>
              <div class="q">Answer</div>
              <p class="a-text">{_escape(turn["answer"])}</p>
              <p class="meta">{_escape(turn["meta"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
