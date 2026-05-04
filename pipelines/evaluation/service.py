from __future__ import annotations

import copy
import json
import os
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*google\\.generativeai package has ended.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"instructor\.providers\.gemini\.client",
)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult

from core.config import (
    CHROMA_DIR,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_EVALUATOR_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_QUESTIONS_PATH,
    RAGAS_DETAILS_PATH,
    RAGAS_SAMPLES_PATH,
    RAGAS_SUMMARY_PATH,
)
from core.gemini import GeminiKeyRotator, is_gemini_quota_error, resolve_gemini_keys
from pipelines.chatbot import (
    ask_university_chatbot,
    initialize_local_llm,
    initialize_retrieval_system,
)
from pipelines.retrieval import load_questions_dataset


def normalize_id(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).strip()


def _resolve_gemini_key(explicit_key: str | Sequence[str] | None = None) -> str | None:
    keys = resolve_gemini_keys(explicit_key)
    return keys[0] if keys else None


def load_default_ragas_metrics() -> list[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    return [
        copy.deepcopy(context_precision),
        copy.deepcopy(context_recall),
        copy.deepcopy(faithfulness),
        copy.deepcopy(answer_relevancy),
        copy.deepcopy(answer_correctness),
    ]


def _batched_records(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _build_ragas_clients(api_key: str, evaluator_model: str, embedding_model: str) -> tuple[Any, Any]:
    ragas_llm = ChatGoogleGenerativeAI(
        model=evaluator_model,
        api_key=api_key,
        temperature=0.0,
    )
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
    )
    return ragas_llm, ragas_embeddings


def _rotate_key_or_raise(rotator: GeminiKeyRotator | None, exc: Exception, stage: str) -> None:
    if rotator is None or not is_gemini_quota_error(exc) or not rotator.can_rotate():
        raise exc
    previous = rotator.masked_current_key()
    rotator.rotate()
    print(f"[{stage}] Quota hit on {previous}. Switching to {rotator.masked_current_key()}.")


def _reference_contexts_from_record(record: dict, chunk_contents: dict[str, dict]) -> tuple[list[str], list[str]]:
    raw_ids: list[str] = []
    if record.get("chunk_id"):
        raw_ids.append(record["chunk_id"])
    raw_ids.extend(record.get("chunk_ids", []) or [])

    seen: set[str] = set()
    reference_ids: list[str] = []
    reference_contexts: list[str] = []

    for chunk_id in raw_ids:
        normalized_id = normalize_id(chunk_id)
        if not normalized_id or normalized_id in seen:
            continue
        seen.add(normalized_id)
        reference_ids.append(normalized_id)
        chunk = chunk_contents.get(normalized_id)
        if chunk and chunk.get("content"):
            reference_contexts.append(chunk["content"])

    return reference_ids, reference_contexts


def build_ragas_dataset(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    db_directory: str | Path = CHROMA_DIR,
    top_k: int = 3,
    answer_backend: str = "gemini",
    gemini_api_key: str | Sequence[str] | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    limit: int | None = None,
) -> tuple[EvaluationDataset, list[dict[str, Any]]]:
    if answer_backend not in {"gemini", "local"}:
        raise ValueError("answer_backend must be 'gemini' or 'local'.")

    api_keys = resolve_gemini_keys(gemini_api_key)
    key_rotator = GeminiKeyRotator(api_keys) if api_keys else None
    if answer_backend == "gemini" and not api_keys:
        raise ValueError("RAGAS evaluation requires GOOGLE_API_KEY or GEMINI_API_KEY.")

    retrieval_system = initialize_retrieval_system(db_directory=db_directory)
    local_tokenizer = None
    local_pipeline = None
    if answer_backend == "local":
        local_tokenizer, local_pipeline = initialize_local_llm()

    questions = load_questions_dataset(questions_path)
    if limit is not None:
        questions = questions[:limit]

    samples: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []

    for record in questions:
        query = str(record.get("cau_hoi", "")).strip()
        reference = str(record.get("answer", "")).strip()
        if not query or not reference:
            continue

        while True:
            try:
                answer, sources, best_docs = ask_university_chatbot(
                    query=query,
                    retrieval_system=retrieval_system,
                    top_k=top_k,
                    backend=answer_backend,
                    gemini_api_key=key_rotator.current_key if key_rotator else None,
                    gemini_model=gemini_model,
                    local_pipeline=local_pipeline,
                    local_tokenizer=local_tokenizer,
                    raise_on_error=True,
                )
                break
            except Exception as exc:
                _rotate_key_or_raise(key_rotator, exc, stage="answer-generation")

        retrieved_contexts = [str(doc.get("content", "")) for doc in best_docs if doc.get("content")]
        retrieved_context_ids = [normalize_id(doc.get("chunk_id", "")) for doc in best_docs if doc.get("chunk_id")]
        reference_context_ids, reference_contexts = _reference_contexts_from_record(
            record=record,
            chunk_contents=retrieval_system.chunk_contents,
        )

        sample = {
            "user_input": query,
            "response": answer,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
            "reference_contexts": reference_contexts,
        }
        samples.append(sample)
        raw_records.append(
            {
                "stt": record.get("stt"),
                "chu_de": record.get("chu_de", ""),
                "user_input": query,
                "response": answer,
                "reference": reference,
                "sources": sources,
                "retrieved_context_ids": retrieved_context_ids,
                "reference_context_ids": reference_context_ids,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
            }
        )

    if not samples:
        raise ValueError("No valid evaluation samples were built from the questions dataset.")

    return EvaluationDataset.from_list(samples), raw_records


def evaluate_with_ragas(
    dataset: EvaluationDataset,
    gemini_api_key: str | Sequence[str] | None = None,
    evaluator_model: str = DEFAULT_GEMINI_EVALUATOR_MODEL,
    embedding_model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
    experiment_name: str | None = None,
    batch_size: int | None = None,
    samples_per_batch: int = 5,
) -> tuple[EvaluationResult, pd.DataFrame]:
    api_keys = resolve_gemini_keys(gemini_api_key)
    if not api_keys:
        raise ValueError("RAGAS evaluation requires GOOGLE_API_KEY or GEMINI_API_KEY.")

    records = dataset.to_list()
    if not records:
        raise ValueError("RAGAS evaluation dataset is empty.")

    effective_samples_per_batch = max(int(samples_per_batch or len(records)), 1)
    record_batches = _batched_records(records, effective_samples_per_batch)
    key_rotator = GeminiKeyRotator(api_keys)
    details_frames: list[pd.DataFrame] = []
    last_result: EvaluationResult | None = None

    for batch_index, batch_records in enumerate(record_batches, start=1):
        batch_dataset = EvaluationDataset.from_list(batch_records)

        while True:
            try:
                ragas_llm, ragas_embeddings = _build_ragas_clients(
                    api_key=key_rotator.current_key,
                    evaluator_model=evaluator_model,
                    embedding_model=embedding_model,
                )
                result = evaluate(
                    dataset=batch_dataset,
                    metrics=load_default_ragas_metrics(),
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    experiment_name=experiment_name,
                    show_progress=True,
                    batch_size=batch_size,
                    raise_exceptions=True,
                )
                last_result = result
                details_frames.append(result.to_pandas())
                print(
                    f"[ragas] Completed batch {batch_index}/{len(record_batches)} "
                    f"with key {key_rotator.masked_current_key()}."
                )
                break
            except Exception as exc:
                _rotate_key_or_raise(key_rotator, exc, stage=f"ragas-batch-{batch_index}")

    if last_result is None:
        raise ValueError("RAGAS evaluation did not produce any batch result.")
    details_df = pd.concat(details_frames, ignore_index=True) if details_frames else pd.DataFrame()
    return last_result, details_df


def _summary_from_details(details_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in details_df.columns
        if column
        in {
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        }
    ]

    summary_rows: list[dict[str, Any]] = []
    for column in metric_columns:
        series = pd.to_numeric(details_df[column], errors="coerce")
        summary_rows.append(
            {
                "metric": column,
                "mean": float(series.mean()) if not series.dropna().empty else float("nan"),
                "min": float(series.min()) if not series.dropna().empty else float("nan"),
                "max": float(series.max()) if not series.dropna().empty else float("nan"),
                "count": int(series.count()),
            }
        )

    return pd.DataFrame(summary_rows)


def save_ragas_reports(
    raw_records: list[dict[str, Any]],
    details_df: pd.DataFrame,
    samples_path: str | Path = RAGAS_SAMPLES_PATH,
    details_path: str | Path = RAGAS_DETAILS_PATH,
    summary_path: str | Path = RAGAS_SUMMARY_PATH,
) -> pd.DataFrame:
    samples_path = Path(samples_path)
    details_path = Path(details_path)
    summary_path = Path(summary_path)

    for path in (samples_path, details_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with samples_path.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df = _summary_from_details(details_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df
