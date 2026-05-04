from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import CHROMA_DIR, DEFAULT_CHROMA_COLLECTION, DEFAULT_EMBEDDING_MODEL, chunk_json_dirs


def ensure_cuda_available() -> str:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to check CUDA availability.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA is required. CPU mode is not supported for this deployment.")
    return "cuda"


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def create_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    require_gpu: bool = False,
) -> HuggingFaceEmbeddings:
    resolved_device = ensure_cuda_available() if require_gpu else resolve_device(device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_all_json_chunks(json_folder_paths: list[Path] | None = None) -> list[dict]:
    all_chunks_data: list[dict] = []
    for folder in json_folder_paths or chunk_json_dirs():
        if not Path(folder).exists():
            continue
        for json_file in sorted(Path(folder).glob("*_chunks.json")):
            all_chunks_data.extend(json.loads(json_file.read_text(encoding="utf-8")))
    return all_chunks_data


def prepare_documents_for_langchain(chunks_data: list[dict]) -> list[Document]:
    documents: list[Document] = []
    for chunk in chunks_data:
        metadata = dict(chunk.get("metadata", {}))
        metadata["chunk_id"] = chunk.get("chunk_id", "")
        documents.append(Document(page_content=chunk.get("content", ""), metadata=metadata))
    return documents


def rebuild_vector_db(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    batch_size: int = 100,
    clear_existing: bool = True,
    require_gpu: bool = False,
) -> Chroma:
    persist_directory = Path(persist_directory)
    persist_directory.parent.mkdir(parents=True, exist_ok=True)

    if clear_existing and persist_directory.exists():
        shutil.rmtree(persist_directory)

    embeddings = create_embeddings(model_name=embedding_model, device=device, require_gpu=require_gpu)
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    for start in range(0, len(documents), batch_size):
        vector_db.add_documents(documents[start : start + batch_size])

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def load_existing_vector_db(
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    require_gpu: bool = False,
) -> Chroma:
    embeddings = create_embeddings(model_name=embedding_model, device=device, require_gpu=require_gpu)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(Path(persist_directory)),
    )


def build_vector_db_from_chunk_dirs(
    json_folder_paths: list[Path] | None = None,
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    batch_size: int = 100,
    clear_existing: bool = True,
    require_gpu: bool = False,
) -> Chroma:
    chunks = load_all_json_chunks(json_folder_paths)
    documents = prepare_documents_for_langchain(chunks)
    return rebuild_vector_db(
        documents=documents,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
        device=device,
        batch_size=batch_size,
        clear_existing=clear_existing,
        require_gpu=require_gpu,
    )
