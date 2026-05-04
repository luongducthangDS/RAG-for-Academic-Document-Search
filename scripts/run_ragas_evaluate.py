from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from pipelines.evaluation import build_ragas_dataset, evaluate_with_ragas, save_ragas_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation with RAGAS.")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--backend", choices=["gemini", "local"], default="gemini")
    parser.add_argument(
        "--api-key",
        action="append",
        dest="api_keys",
        default=None,
        help="Gemini API key. Repeat this flag to provide multiple keys.",
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_GEMINI_EVALUATOR_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_GEMINI_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=5)
    parser.add_argument("--experiment-name", default="uneti-rag-ragas-eval")
    parser.add_argument("--samples-out", type=Path, default=RAGAS_SAMPLES_PATH)
    parser.add_argument("--details-out", type=Path, default=RAGAS_DETAILS_PATH)
    parser.add_argument("--summary-out", type=Path, default=RAGAS_SUMMARY_PATH)
    args = parser.parse_args()

    try:
        dataset, raw_records = build_ragas_dataset(
            questions_path=args.questions,
            db_directory=args.db_dir,
            top_k=args.top_k,
            answer_backend=args.backend,
            gemini_api_key=args.api_keys,
            gemini_model=args.gemini_model,
            limit=args.limit,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to prepare RAGAS dataset: {exc}") from exc
    print(f"Prepared {len(raw_records)} evaluation samples.")

    try:
        result, details_df = evaluate_with_ragas(
            dataset=dataset,
            gemini_api_key=args.api_keys,
            evaluator_model=args.evaluator_model,
            embedding_model=args.embedding_model,
            experiment_name=args.experiment_name,
            batch_size=args.batch_size,
            samples_per_batch=args.samples_per_batch,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to run RAGAS evaluation: {exc}") from exc
    summary_df = save_ragas_reports(
        raw_records=raw_records,
        details_df=details_df,
        samples_path=args.samples_out,
        details_path=args.details_out,
        summary_path=args.summary_out,
    )

    print("\nRAGAS Summary")
    if summary_df.empty:
        print("  No metric output was produced.")
    else:
        print(summary_df.to_string(index=False))

    print("\nSaved files")
    print(f"  Samples: {args.samples_out}")
    print(f"  Details: {args.details_out}")
    print(f"  Summary: {args.summary_out}")

    try:
        total_tokens = result.total_tokens()
        if total_tokens is not None:
            print(f"\nTotal evaluator tokens: {total_tokens}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
