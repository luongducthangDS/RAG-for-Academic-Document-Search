from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "source"

PDF_DIR = DATA_DIR / "pdf"
PDF_MD_DIR = PDF_DIR / "md"
SCAN_DIR = DATA_DIR / "scan"
SCAN_MD_DIR = SCAN_DIR / "md"

CHROMA_DIR = DATA_DIR / "chroma_db"
APP_DIR = PROJECT_ROOT / "app"
RETRIEVAL_CHART_PATH = DATA_DIR / "retrieval_evaluation_chart.png"
CHUNK_ANALYSIS_PATH = DATA_DIR / "chunk_analysis.json"
RAGAS_SAMPLES_PATH = DATA_DIR / "ragas_evaluation_samples.jsonl"
RAGAS_DETAILS_PATH = DATA_DIR / "ragas_evaluation_details.csv"
RAGAS_SUMMARY_PATH = DATA_DIR / "ragas_evaluation_summary.csv"

DEFAULT_CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "quy_che_dai_hoc")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_LOCAL_LLM = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_GEMINI_EVALUATOR_MODEL = os.getenv("GEMINI_EVALUATOR_MODEL", DEFAULT_GEMINI_MODEL)
DEFAULT_GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions1.json"
if not DEFAULT_QUESTIONS_PATH.exists():
    DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.json"


def markdown_dirs() -> list[Path]:
    return [PDF_MD_DIR, SCAN_MD_DIR]


def chunk_json_dirs() -> list[Path]:
    return markdown_dirs()


def ensure_runtime_dirs() -> None:
    for path in [PDF_MD_DIR, SCAN_MD_DIR, CHROMA_DIR, APP_DIR]:
        path.mkdir(parents=True, exist_ok=True)
