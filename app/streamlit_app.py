from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import streamlit as st
import torch

# ─── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CHROMA_DIR, DEFAULT_LOCAL_LLM, chunk_json_dirs
from pipelines.chatbot import ask_university_chatbot, initialize_local_llm, initialize_retrieval_system
from app.prompts import (
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_QWEN_ONLY,
    SYSTEM_PROMPT_QUERY_REWRITE,
    NO_CONTEXT_RESPONSE,
    build_rag_user_message,
    build_query_rewrite_message,
)

# ─── Constants ────────────────────────────────────────────────────────────────
LOGO_PATH = PROJECT_ROOT / "app" / "logo" / "UNETI_Logo.png"
MODE_RETRIEVAL = "Qwen + Retrieval"
MODE_QWEN_ONLY = "Qwen only"
CHAT_MODES = [MODE_RETRIEVAL, MODE_QWEN_ONLY]


# ─── Types ────────────────────────────────────────────────────────────────────
class DeviceInfo(NamedTuple):
    type: str
    name: str


class SourceRow(NamedTuple):
    chunk_id: str
    source: str
    preview: str


# ─── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def detect_device() -> DeviceInfo:
    if torch.cuda.is_available():
        return DeviceInfo("CUDA", torch.cuda.get_device_name(0))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("MPS", "Apple Silicon (MPS)")
    return DeviceInfo("CPU", "CPU")


@st.cache_resource(show_spinner=False)
def setup_ngrok(token: str) -> str | None:
    if not token:
        return None
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        return str(ngrok.connect(8501, "http"))
    except Exception as exc:
        return f"ERROR:{exc}"


@st.cache_resource(show_spinner=False)
def get_retrieval_system(db_dir: str):
    return initialize_retrieval_system(
        db_directory=db_dir,
        json_folder_paths=chunk_json_dirs(),
    )


@st.cache_resource(show_spinner=False)
def get_local_llm(model_id: str):
    return initialize_local_llm(model_id=model_id)


# ─── LLM helpers ──────────────────────────────────────────────────────────────
def _call_pipeline(pipeline, tokenizer, system: str, user: str) -> str:
    """Gọi pipeline với system + user message, trả về text đã strip."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(prompt)
    return outputs[0]["generated_text"].strip(" -:\n\t")


def rewrite_query(raw_query: str, pipeline, tokenizer) -> str:
    """Viết lại câu hỏi để cải thiện chất lượng retrieval."""
    return _call_pipeline(
        pipeline, tokenizer,
        system=SYSTEM_PROMPT_QUERY_REWRITE,
        user=build_query_rewrite_message(raw_query),
    )


def ask_qwen_only(query: str, pipeline, tokenizer) -> str:
    return _call_pipeline(
        pipeline, tokenizer,
        system=SYSTEM_PROMPT_QWEN_ONLY,
        user=query,
    )


def ask_with_context(
    query: str,
    retrieved_chunks: list[dict],
    pipeline,
    tokenizer,
) -> str:
    """Sinh câu trả lời từ context đã truy xuất."""
    if not retrieved_chunks:
        return NO_CONTEXT_RESPONSE
    user_message = build_rag_user_message(query, retrieved_chunks)
    return _call_pipeline(
        pipeline, tokenizer,
        system=SYSTEM_PROMPT_RAG,
        user=user_message,
    )


# ─── UI helpers ───────────────────────────────────────────────────────────────
def chunks_to_source_rows(chunks: list[dict]) -> list[SourceRow]:
    return [
        SourceRow(
            chunk_id=c.get("chunk_id", ""),
            source=Path(str(c.get("metadata", {}).get("source", ""))).name,
            preview=c.get("content", "")[:300].replace("\n", " "),
        )
        for c in chunks
    ]


def render_sources(rows: list[SourceRow]) -> None:
    if not rows:
        st.caption("Không có tài liệu nào được truy xuất.")
        return
    for row in rows:
        st.markdown(f"- `{row.chunk_id}` | `{row.source}`")
        st.caption(row.preview)


# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UNETI RAG Chatbot", layout="wide")
device = detect_device()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_exists = LOGO_PATH.exists()
    if logo_exists:
        st.image(str(LOGO_PATH), use_container_width=True)

    st.header("Cấu hình")
    st.success(f"Thiết bị: {device.name}")

    chat_mode = st.radio("Chế độ hỏi đáp", options=CHAT_MODES, index=0)
    db_dir = st.text_input("Chroma DB", value=str(CHROMA_DIR))
    local_model_id = st.text_input("Qwen model", value=DEFAULT_LOCAL_LLM)
    top_k = st.slider("Top K retrieval", min_value=1, max_value=5, value=3)
    use_query_rewrite = st.toggle(
        "Viết lại câu hỏi trước khi tìm kiếm",
        value=True,
        help="Cải thiện chất lượng retrieval với câu hỏi khẩu ngữ hoặc mơ hồ.",
    )

    if st.button("Xoá lịch sử chat"):
        st.session_state.pop("messages", None)
        st.rerun()

    ngrok_result = setup_ngrok(os.environ.get("NGROK_AUTH_TOKEN", ""))
    if ngrok_result:
        if ngrok_result.startswith("ERROR:"):
            st.warning(f"Ngrok lỗi: {ngrok_result[6:]}")
        else:
            st.success(f"🌐 Public URL: {ngrok_result}")

# ─── Header ───────────────────────────────────────────────────────────────────
left, right = st.columns([1, 6])
with left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
with right:
    st.title("UNETI RAG Chatbot")
    st.caption("Tư vấn học vụ · Hai chế độ: Qwen only và Qwen + Hybrid Retrieval + Reranker")

if device.type == "CPU":
    st.warning("⚠️ Đang chạy trên CPU — phản hồi có thể chậm hơn bình thường.")

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Render lịch sử ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Nguồn tham chiếu"):
                render_sources(msg["sources"])

# ─── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Nhập câu hỏi về quy chế, học vụ, chuẩn đầu ra..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved_chunks: list[dict] = []
    answer = ""

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Đang nạp Qwen trên {device.type}..."):
                tokenizer, pipeline = get_local_llm(local_model_id)

            if chat_mode == MODE_RETRIEVAL:
                # (Tuỳ chọn) Viết lại câu hỏi để tăng chất lượng retrieval
                search_query = prompt
                if use_query_rewrite:
                    with st.spinner("Đang phân tích câu hỏi..."):
                        search_query = rewrite_query(prompt, pipeline, tokenizer)

                with st.spinner("Đang tìm kiếm tài liệu liên quan..."):
                    retrieval_system = get_retrieval_system(db_dir)
                    _, _, retrieved_chunks = ask_university_chatbot(
                        query=search_query,
                        retrieval_system=retrieval_system,
                        top_k=top_k,
                        backend="local",
                        local_pipeline=pipeline,
                        local_tokenizer=tokenizer,
                    )

                with st.spinner("Đang tổng hợp câu trả lời..."):
                    # Sinh câu trả lời với prompt RAG chuẩn, dùng câu hỏi GỐC
                    answer = ask_with_context(prompt, retrieved_chunks, pipeline, tokenizer)

            else:
                with st.spinner("Đang sinh câu trả lời..."):
                    answer = ask_qwen_only(prompt, pipeline, tokenizer)

        except RuntimeError as exc:
            answer = f"⚠️ Lỗi model: {exc}"
        except Exception as exc:
            answer = f"⚠️ Lỗi hệ thống: {exc}"

        st.markdown(answer)

        source_rows = chunks_to_source_rows(retrieved_chunks)
        if chat_mode == MODE_RETRIEVAL:
            with st.expander("Nguồn tham chiếu"):
                render_sources(source_rows)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_rows}
    )