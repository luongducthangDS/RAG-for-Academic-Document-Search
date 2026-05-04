from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.config import (
    CHROMA_DIR,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RERANKER_MODEL,
    RETRIEVAL_CHART_PATH,
    chunk_json_dirs,
)
from pipelines.vectorstore.service import load_existing_vector_db, resolve_device


@dataclass
class ChunkCorpus:
    tokenized_corpus: list[list[str]]
    chunk_ids: list[str]
    chunk_contents: dict[str, str]
    chunk_metadata: dict[str, dict]


@dataclass
class GroundTruthStats:
    query_count: int
    avg_relevant_docs: float
    median_relevant_docs: float
    min_relevant_docs: int
    max_relevant_docs: int
    single_label_queries: int
    multi_label_queries: int
    explicit_graded_queries: int
    label_distribution: dict[int, int]


def simple_tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return normalized.split()


def normalize_id(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def load_chunk_corpus(json_folder_paths: list[Path] | None = None) -> ChunkCorpus:
    tokenized_corpus: list[list[str]] = []
    chunk_ids: list[str] = []
    chunk_contents: dict[str, str] = {}
    chunk_metadata: dict[str, dict] = {}

    for folder in json_folder_paths or chunk_json_dirs():
        if not Path(folder).exists():
            continue
        for json_file in sorted(Path(folder).glob("*_chunks.json")):
            chunks = json.loads(json_file.read_text(encoding="utf-8"))
            for chunk in chunks:
                chunk_id = normalize_id(chunk["chunk_id"])
                tokenized_corpus.append(simple_tokenize(chunk["content"]))
                chunk_ids.append(chunk_id)
                chunk_contents[chunk_id] = chunk["content"]
                chunk_metadata[chunk_id] = chunk.get("metadata", {})

    return ChunkCorpus(
        tokenized_corpus=tokenized_corpus,
        chunk_ids=chunk_ids,
        chunk_contents=chunk_contents,
        chunk_metadata=chunk_metadata,
    )


def build_bm25_index(corpus: ChunkCorpus) -> BM25Okapi:
    return BM25Okapi(corpus.tokenized_corpus)


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
) -> CrossEncoder:
    return CrossEncoder(model_name, max_length=512, device=resolve_device(device))


def sparse_retrieval_bm25(query: str, bm25_index: BM25Okapi, chunk_ids: list[str], top_k: int = 5) -> list[dict]:
    scores = bm25_index.get_scores(simple_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"chunk_id": chunk_ids[index], "score": float(scores[index]), "rank": rank + 1}
        for rank, index in enumerate(top_indices)
    ]


def dense_retrieval_chroma(query: str, vector_store: Chroma, top_k: int = 5) -> list[dict]:
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    results = []
    for rank, (doc, score) in enumerate(docs_with_scores, start=1):
        results.append(
            {
                "chunk_id": normalize_id(doc.metadata.get("chunk_id", "")),
                "score": float(score),
                "rank": rank,
            }
        )
    return results


def hybrid_retrieval(
    query: str,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    vector_store: Chroma,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    sparse_results = sparse_retrieval_bm25(query, bm25_index, chunk_ids, top_k=candidate_k)
    dense_results = dense_retrieval_chroma(query, vector_store, top_k=candidate_k)

    rrf_k = 60
    rrf_scores: dict[str, float] = {}

    for result in sparse_results + dense_results:
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + result["rank"]))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"chunk_id": chunk_id, "score": float(score), "rank": rank + 1}
        for rank, (chunk_id, score) in enumerate(sorted_rrf[:top_k])
    ]


def hybrid_reranking_retrieval(
    query: str,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    vector_store: Chroma,
    reranker: CrossEncoder,
    chunk_contents: dict[str, str],
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    hybrid_results = hybrid_retrieval(
        query=query,
        bm25_index=bm25_index,
        chunk_ids=chunk_ids,
        vector_store=vector_store,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )

    if not hybrid_results:
        return []

    cross_input = []
    ordered_ids = []
    for result in hybrid_results:
        chunk_id = result["chunk_id"]
        cross_input.append([query, chunk_contents.get(chunk_id, "")])
        ordered_ids.append(chunk_id)

    rerank_scores = reranker.predict(cross_input)
    reranked_results = []

    for idx in np.argsort(rerank_scores)[::-1]:
        reranked_results.append(
            {
                "chunk_id": ordered_ids[idx],
                "score": float(rerank_scores[idx]),
                "rank": len(reranked_results) + 1,
            }
        )
        if len(reranked_results) == top_k:
            break

    return reranked_results


def load_questions_dataset(questions_path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[dict]:
    path = Path(questions_path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _has_explicit_graded_relevance(record: dict) -> bool:
    return any(record.get(field) for field in ("graded_relevance", "relevances", "relevance"))


def extract_graded_relevance(record: dict) -> dict[str, int]:
    graded_relevance: dict[str, int] = {}

    for field in ("graded_relevance", "relevances", "relevance"):
        payload = record.get(field)
        if not payload:
            continue

        if isinstance(payload, dict):
            if "chunk_id" in payload:
                chunk_id = normalize_id(payload.get("chunk_id", ""))
                score = int(payload.get("score", payload.get("relevance", payload.get("grade", 0))))
                if chunk_id:
                    graded_relevance[chunk_id] = max(0, score)
            else:
                for chunk_id, score in payload.items():
                    normalized_id = normalize_id(chunk_id)
                    if normalized_id:
                        graded_relevance[normalized_id] = max(0, int(score))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    chunk_id = normalize_id(item.get("chunk_id", item.get("id", "")))
                    score = int(item.get("score", item.get("relevance", item.get("grade", 0))))
                    if chunk_id:
                        graded_relevance[chunk_id] = max(0, score)
                elif isinstance(item, str):
                    chunk_id = normalize_id(item)
                    if chunk_id:
                        graded_relevance[chunk_id] = max(graded_relevance.get(chunk_id, 0), 1)

    if graded_relevance:
        return {chunk_id: score for chunk_id, score in graded_relevance.items() if score > 0}

    primary_chunk_id = normalize_id(record.get("chunk_id", ""))
    if primary_chunk_id:
        graded_relevance[primary_chunk_id] = 2

    chunk_ids_list = record.get("chunk_ids", []) or []
    n = len(chunk_ids_list)
    for position, chunk_id in enumerate(chunk_ids_list):
        normalized_id = normalize_id(chunk_id)
        if normalized_id:
            score = max(1, n - position)  # 5,4,3,2,1 với n=5
            graded_relevance[normalized_id] = score

    return graded_relevance


def _expected_ids(record: dict) -> set[str]:
    return set(extract_graded_relevance(record).keys())


def summarize_ground_truth(questions: list[dict]) -> GroundTruthStats:
    relevant_counts = [len(extract_graded_relevance(record)) for record in questions]
    query_count = len(relevant_counts)
    single_label_queries = sum(count == 1 for count in relevant_counts)
    multi_label_queries = sum(count > 1 for count in relevant_counts)
    explicit_graded_queries = sum(_has_explicit_graded_relevance(record) for record in questions)

    label_distribution: dict[int, int] = {}
    for count in relevant_counts:
        label_distribution[count] = label_distribution.get(count, 0) + 1

    if not relevant_counts:
        return GroundTruthStats(
            query_count=0,
            avg_relevant_docs=0.0,
            median_relevant_docs=0.0,
            min_relevant_docs=0,
            max_relevant_docs=0,
            single_label_queries=0,
            multi_label_queries=0,
            explicit_graded_queries=0,
            label_distribution={},
        )

    return GroundTruthStats(
        query_count=query_count,
        avg_relevant_docs=mean(relevant_counts),
        median_relevant_docs=float(median(relevant_counts)),
        min_relevant_docs=min(relevant_counts),
        max_relevant_docs=max(relevant_counts),
        single_label_queries=single_label_queries,
        multi_label_queries=multi_label_queries,
        explicit_graded_queries=explicit_graded_queries,
        label_distribution=label_distribution,
    )


def calculate_metrics(
    results: list[dict],
    expected_chunk_ids: set[str],
    recall_k: int = 10,
) -> tuple[float, float, float]:
    normalized_results = [normalize_id(result["chunk_id"]) for result in results]
    relevant_count = max(len(expected_chunk_ids), 1)

    hit_at_1 = 1.0 if normalized_results and normalized_results[0] in expected_chunk_ids else 0.0

    recall_hits = sum(chunk_id in expected_chunk_ids for chunk_id in normalized_results[:recall_k])
    recall_at_k = recall_hits / relevant_count

    mrr = 0.0
    for rank, chunk_id in enumerate(normalized_results, start=1):
        if chunk_id in expected_chunk_ids:
            mrr = 1.0 / rank
            break

    return hit_at_1, mrr, recall_at_k


def calculate_ndcg(results: list[dict], graded_relevance: dict[str, int], k: int = 3) -> float:
    if not graded_relevance:
        return 0.0

    def discounted_gain(relevance_score: int, rank: int) -> float:
        return (2**relevance_score - 1) / np.log2(rank + 1)

    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        relevance_score = graded_relevance.get(normalize_id(result["chunk_id"]), 0)
        dcg += discounted_gain(relevance_score, rank)

    ideal_scores = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = sum(discounted_gain(score, rank) for rank, score in enumerate(ideal_scores, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def plot_evaluation_results(df_metrics: pd.DataFrame, save_path: str | Path) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    melted = df_metrics.melt(id_vars="Method", var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Metric", y="Score", hue="Method", palette="viridis", ax=ax)

    plt.title("Retrieval Performance Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.ylim(0, 1.18)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")

    n_methods = df_metrics["Method"].nunique()

    for i, patch in enumerate(ax.patches):
        h = patch.get_height()
        if h <= 0:
            continue
        group_pos = i % n_methods
        vertical_offset = 6 + (group_pos % 2) * 8

        ax.annotate(
            f"{h:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, vertical_offset),
            textcoords="offset points",
        )

    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def evaluate_retrieval(
    json_folder_paths: list[Path] | None = None,
    db_directory: str | Path = CHROMA_DIR,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    chart_save_path: str | Path = RETRIEVAL_CHART_PATH,
    top_k: int = 10,
    device: str | None = None,
) -> pd.DataFrame:
    recall_k = 10
    ndcg_k = 3
    retrieval_depth = max(top_k, recall_k)

    corpus = load_chunk_corpus(json_folder_paths)
    bm25_index = build_bm25_index(corpus)
    vector_store = load_existing_vector_db(
        persist_directory=db_directory,
        collection_name=DEFAULT_CHROMA_COLLECTION,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        device=device,
    )
    reranker = load_reranker(device=device)
    questions = load_questions_dataset(questions_path)

    methods = [
        "Baseline 1: BM25",
        "Baseline 2: Dense",
        "Proposed 1: Hybrid",
        "Proposed 2: Hybrid + Reranker",
    ]
    metrics_sum = {method: {"hit": 0.0, "mrr": 0.0, "recall": 0.0, "ndcg": 0.0} for method in methods}

    for record in questions:
        query = record.get("cau_hoi", "")
        graded_relevance = extract_graded_relevance(record)
        expected_ids = _expected_ids(record)

        bm25_results = sparse_retrieval_bm25(query, bm25_index, corpus.chunk_ids, top_k=retrieval_depth)
        dense_results = dense_retrieval_chroma(query, vector_store, top_k=retrieval_depth)
        hybrid_results = hybrid_retrieval(
            query,
            bm25_index,
            corpus.chunk_ids,
            vector_store,
            top_k=retrieval_depth,
            candidate_k=retrieval_depth,
        )
        rerank_results = hybrid_reranking_retrieval(
            query,
            bm25_index,
            corpus.chunk_ids,
            vector_store,
            reranker,
            corpus.chunk_contents,
            top_k=retrieval_depth,
            candidate_k=retrieval_depth,
        )

        for method, results in [
            (methods[0], bm25_results),
            (methods[1], dense_results),
            (methods[2], hybrid_results),
            (methods[3], rerank_results),
        ]:
            hit, mrr, recall = calculate_metrics(
                results,
                expected_ids,
                recall_k=recall_k,
            )
            ndcg = calculate_ndcg(results, graded_relevance, k=ndcg_k)
            metrics_sum[method]["hit"] += hit
            metrics_sum[method]["mrr"] += mrr
            metrics_sum[method]["recall"] += recall
            metrics_sum[method]["ndcg"] += ndcg

    query_count = max(len(questions), 1)
    rows = []
    for method in methods:
        rows.append(
            {
                "Method": method,
                "Hit@1":    metrics_sum[method]["hit"]    / query_count,
                "MRR@10":   metrics_sum[method]["mrr"]    / query_count,
                "Recall@10": metrics_sum[method]["recall"] / query_count,
                "NDCG@3":   metrics_sum[method]["ndcg"]   / query_count,
            }
        )

    df_metrics = pd.DataFrame(rows)
    plot_evaluation_results(df_metrics, chart_save_path)
    return df_metrics