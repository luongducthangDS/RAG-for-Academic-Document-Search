from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CHROMA_DIR, DEFAULT_QUESTIONS_PATH, RETRIEVAL_CHART_PATH, chunk_json_dirs
from pipelines.retrieval import evaluate_retrieval, load_questions_dataset, summarize_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against the question set.")
    parser.add_argument("--chunk-dir", action="append", type=Path, dest="chunk_dirs")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--chart", type=Path, default=RETRIEVAL_CHART_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    questions = load_questions_dataset(args.questions)
    stats = summarize_ground_truth(questions)
    print("Ground Truth Summary")
    print(f"  Queries              : {stats.query_count}")
    print(f"  Avg relevant/query   : {stats.avg_relevant_docs:.2f}")
    print(f"  Median relevant/query: {stats.median_relevant_docs:.2f}")
    print(f"  Min/Max relevant     : {stats.min_relevant_docs}/{stats.max_relevant_docs}")
    print(f"  Single-label queries : {stats.single_label_queries}")
    print(f"  Multi-label queries  : {stats.multi_label_queries}")
    print(f"  Explicit graded      : {stats.explicit_graded_queries}")
    print(f"  Label distribution   : {stats.label_distribution}")
    print()

    df = evaluate_retrieval(
        json_folder_paths=args.chunk_dirs or chunk_json_dirs(),
        db_directory=args.db_dir,
        questions_path=args.questions,
        chart_save_path=args.chart,
        top_k=args.top_k,
        device=args.device,
    )
    print("Evaluation Metrics")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()