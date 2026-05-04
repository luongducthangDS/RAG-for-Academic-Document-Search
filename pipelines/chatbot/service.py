from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from core.config import (
    CHROMA_DIR,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_LLM,
    DEFAULT_RERANKER_MODEL,
    chunk_json_dirs,
)
from core.gemini import resolve_gemini_keys
from pipelines.vectorstore.service import ensure_cuda_available, load_existing_vector_db


@dataclass
class RetrievalSystem:
    vector_store: Any # Type hint tuỳ thuộc vào VectorStore bạn dùng (VD: langchain_chroma.Chroma)
    bm25_index: BM25Okapi
    bm25_chunk_ids: list[str]
    chunk_contents: dict[str, dict]
    reranker: CrossEncoder


def get_optimal_device() -> str:
    """Tự động phát hiện thiết bị tốt nhất hiện có (CUDA, MPS, hoặc CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def simple_tokenize(text: str) -> list[str]:
    # Gợi ý: Nếu có thể, hãy dùng thư viện 'pyvi' hoặc 'underthesea' 
    # để tokenize tiếng Việt chính xác hơn thay vì chỉ split bằng khoảng trắng.
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def normalize_id(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def load_all_chunks_for_bm25(json_folder_paths: list[Path] | None = None) -> tuple[list[list[str]], list[str], dict[str, dict]]:
    corpus: list[list[str]] = []
    chunk_ids: list[str] = []
    chunk_contents: dict[str, dict] = {}

    for folder in json_folder_paths or chunk_json_dirs():
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for json_file in sorted(folder_path.glob("*_chunks.json")):
            try:
                chunks = json.loads(json_file.read_text(encoding="utf-8"))
                for chunk in chunks:
                    norm_id = normalize_id(chunk.get("chunk_id", ""))
                    if not norm_id: continue # Bỏ qua chunk không có ID hợp lệ
                    
                    normalized_chunk = dict(chunk)
                    normalized_chunk["chunk_id"] = norm_id
                    # Fix lỗi có thể xảy ra nếu metadata là None
                    normalized_chunk["metadata"] = dict(chunk.get("metadata") or {})

                    corpus.append(simple_tokenize(chunk.get("content", "")))
                    chunk_ids.append(norm_id)
                    chunk_contents[norm_id] = normalized_chunk
            except Exception as e:
                print(f"Lỗi khi đọc file {json_file.name}: {e}")

    return corpus, chunk_ids, chunk_contents


def initialize_retrieval_system(
    db_directory: str | Path = CHROMA_DIR,
    json_folder_paths: list[Path] | None = None,
) -> RetrievalSystem:
    # Vẫn giữ hàm của bạn, nhưng hệ thống sẽ linh hoạt hơn nếu gpu không có
    try:
        ensure_cuda_available()
    except Exception as e:
        print(f"Cảnh báo GPU: {e}. Sẽ tự động chuyển sang cấu hình có sẵn.")
        
    device = get_optimal_device()

    vector_store = load_existing_vector_db(
        persist_directory=db_directory,
        device=device,
        require_gpu=(device == "cuda"),
    )

    corpus, bm25_chunk_ids, chunk_contents = load_all_chunks_for_bm25(json_folder_paths)
    if not corpus:
        raise RuntimeError("Không tìm thấy chunk JSON để khởi tạo BM25.")

    bm25_index = BM25Okapi(corpus)
    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL, max_length=512, device=device)

    return RetrievalSystem(
        vector_store=vector_store,
        bm25_index=bm25_index,
        bm25_chunk_ids=bm25_chunk_ids,
        chunk_contents=chunk_contents,
        reranker=reranker,
    )


def initialize_local_llm(model_id: str = DEFAULT_LOCAL_LLM):
    try:
        ensure_cuda_available()
    except Exception:
        pass # Fallback an toàn

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    return tokenizer, llm_pipeline


def retrieve_for_chat(
    query: str,
    retrieval_system: RetrievalSystem,
    top_k: int = 3,
    bm25_k: int = 15,
    dense_k: int = 15,
    fusion_k: int = 10,
) -> list[dict]:
    tokenized_query = simple_tokenize(query)
    
    # 1. Sparse Retrieval (BM25)
    doc_scores = retrieval_system.bm25_index.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(doc_scores)[::-1][:bm25_k]
    sparse_results = [
        {"chunk_id": retrieval_system.bm25_chunk_ids[idx], "rank": rank + 1}
        for rank, idx in enumerate(top_bm25_indices)
    ]

    # 2. Dense Retrieval (Vector Store)
    # Lấy nội dung page_content để đề phòng thiếu ID (tương thích LangChain)
    docs_with_scores = retrieval_system.vector_store.similarity_search_with_score(query, k=dense_k)
    dense_results = []
    
    for rank, (doc, _) in enumerate(docs_with_scores):
        chunk_id = normalize_id(doc.metadata.get("chunk_id", ""))
        
        # FIX LOGIC RRF: Nếu doc từ vector DB thiếu chunk_id, tạo ID tạm thời dựa trên hash nội dung
        if not chunk_id:
            content_str = getattr(doc, 'page_content', str(doc))
            chunk_id = "hash_" + hashlib.md5(content_str.encode()).hexdigest()
            # Bổ sung tạm vào chunk_contents để CrossEncoder có thể lấy data
            if chunk_id not in retrieval_system.chunk_contents:
                retrieval_system.chunk_contents[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": content_str,
                    "metadata": doc.metadata
                }
                
        dense_results.append({"chunk_id": chunk_id, "rank": rank + 1})

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    rrf_scores: dict[str, float] = {}
    for result in sparse_results + dense_results:
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + result["rank"]))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:fusion_k]

    # 4. CrossEncoder Re-ranking
    cross_input: list[list[str]] = []
    hybrid_docs: list[dict] = []
    for chunk_id, _ in sorted_rrf:
        chunk_data = retrieval_system.chunk_contents.get(chunk_id)
        if not chunk_data:
            continue
        cross_input.append([query, chunk_data.get("content", "")])
        hybrid_docs.append(chunk_data)

    if not cross_input:
        return []

    rerank_scores = retrieval_system.reranker.predict(cross_input)
    top_final_indices = np.argsort(rerank_scores)[::-1][:top_k]
    return [hybrid_docs[index] for index in top_final_indices]


def _system_instruction() -> str:
    # Đã sửa lại thành tiếng Việt có dấu. LLM sẽ trả lời tự nhiên và format chuẩn hơn rất nhiều.
    return (
        "Bạn là chuyên viên tư vấn học vụ ảo của Trường Đại học. "
        "Nhiệm vụ của bạn là giải đáp thắc mắc cho sinh viên dựa trên các quy chế chính thức của trường.\n\n"
        "NGUYÊN TẮC QUAN TRỌNG:\n"
        "1. CHỈ sử dụng các thông tin được cung cấp trong phần TÀI LIỆU THAM KHẢO để trả lời.\n"
        "2. Nếu tài liệu tham khảo không chứa câu trả lời, hãy nói thẳng: "
        "'Xin lỗi, tôi không tìm thấy quy định nào về vấn đề này trong hệ thống dữ liệu hiện tại.' "
        "TUYỆT ĐỐI KHÔNG tự bịa ra thông tin.\n"
        "3. Trả lời thân thiện, mạch lạc, dễ hiểu, trình bày dạng danh sách nếu có nhiều ý.\n"
        "4. Không mở đầu bằng các cụm như 'Dựa vào tài liệu tham khảo', 'Theo tài liệu được cung cấp', "
        "'Dựa trên ngữ cảnh', hoặc các câu dẫn nhập tương tự.\n"
        "5. Trả lời trực tiếp vào nội dung chính của câu hỏi."
    )


def _cleanup_answer_style(answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return ""

    leadin_patterns = [
        r"^\s*dựa vào tài liệu tham khảo(?: mà bạn đã cung cấp| được cung cấp)?[:,]?\s*",
        r"^\s*theo tài liệu tham khảo(?: mà bạn đã cung cấp| được cung cấp)?[:,]?\s*",
        r"^\s*dựa trên (?:ngữ cảnh|tài liệu tham khảo|tài liệu được cung cấp)[:,]?\s*",
        r"^\s*theo (?:ngữ cảnh|tài liệu được cung cấp)[:,]?\s*",
        r"^\s*trong tài liệu tham khảo[:,]?\s*",
        r"^\s*theo quy định trong tài liệu tham khảo[:,]?\s*",
        r"^\s*dua vao tai lieu tham khao[:,]?\s*",
        r"^\s*theo tai lieu tham khao[:,]?\s*",
        r"^\s*dua tren ngu canh[:,]?\s*",
        r"^\s*theo ngu canh[:,]?\s*",
    ]

    for pattern in leadin_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)

    return answer.strip(" -:\n\t")


def _format_context(best_docs: list[dict]) -> tuple[str, list[str]]:
    context = []
    sources: list[str] = []

    for index, doc in enumerate(best_docs, start=1):
        context.append(f"--- TÀI LIỆU SỐ {index} ---\n{doc.get('content', '')}\n")
        source_name = Path(str(doc.get("metadata", {}).get("source", ""))).name
        if source_name and source_name not in sources:
            sources.append(source_name)

    return "\n".join(context), sources


def _resolve_gemini_key(explicit_key: str | list[str] | tuple[str, ...] | None = None) -> str | None:
    keys = resolve_gemini_keys(explicit_key)
    return keys[0] if keys else None


def _generate_with_gemini(query: str, context: str, api_key: str, model_name: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    full_prompt = f"{_system_instruction()}\n\nTÀI LIỆU THAM KHẢO:\n{context}\n\nCÂU HỎI: {query}\n\nCÂU TRẢ LỜI:"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return _cleanup_answer_style(response.text or "")
    except Exception as e:
         return f"Lỗi kết nối tới mô hình Gemini: {e}"


def _generate_with_local_llm(query: str, context: str, local_pipeline, local_tokenizer) -> str:
    messages = [
        {"role": "system", "content": _system_instruction()},
        {"role": "user", "content": f"TÀI LIỆU THAM KHẢO:\n{context}\n\nCÂU HỎI CỦA TÔI:\n{query}"},
    ]
    prompt_str = local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = local_pipeline(prompt_str)
    return _cleanup_answer_style(outputs[0]["generated_text"].strip())


def ask_university_chatbot(
    query: str,
    retrieval_system: RetrievalSystem,
    top_k: int = 3,
    backend: str = "gemini",
    gemini_api_key: str | list[str] | tuple[str, ...] | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    local_pipeline=None,
    local_tokenizer=None,
    raise_on_error: bool = False,
) -> tuple[str, list[str], list[dict]]:
    best_docs = retrieve_for_chat(query=query, retrieval_system=retrieval_system, top_k=top_k)

    if not best_docs:
        return (
            "Xin lỗi, không có dữ liệu nào trong quy chế khớp với câu hỏi của bạn.",
            [],
            [],
        )

    context, sources = _format_context(best_docs)

    try:
        if backend == "local":
            if local_pipeline is None or local_tokenizer is None:
                raise ValueError("Local backend requires initialized tokenizer and pipeline.")
            answer = _generate_with_local_llm(query, context, local_pipeline, local_tokenizer)
        else:
            api_key = _resolve_gemini_key(gemini_api_key)
            if not api_key:
                raise ValueError("Gemini backend requires GOOGLE_API_KEY or GEMINI_API_KEY.")
            answer = _generate_with_gemini(query, context, api_key=api_key, model_name=gemini_model)
    except Exception as exc:
        if raise_on_error:
            raise
        answer = f"Lỗi trong quá trình sinh câu trả lời: {exc}"

    # Bỏ lời gọi hàm _cleanup_answer_style ở đây vì đã gọi trong các hàm generate rồi
    return answer, sources, best_docs
