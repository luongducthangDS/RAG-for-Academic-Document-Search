from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CHROMA_DIR
from pipelines.chatbot import ask_university_chatbot, initialize_local_llm, initialize_retrieval_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG chatbot in a terminal session.")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--backend", choices=["gemini", "local"], default="gemini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retrieval_system = initialize_retrieval_system(db_directory=args.db_dir)
    local_tokenizer = None
    local_pipeline = None

    if args.backend == "local":
        local_tokenizer, local_pipeline = initialize_local_llm()

    print("Type 'exit' to quit.")
    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer, sources, _ = ask_university_chatbot(
            query=query,
            retrieval_system=retrieval_system,
            top_k=args.top_k,
            backend=args.backend,
            gemini_api_key=args.api_key,
            local_pipeline=local_pipeline,
            local_tokenizer=local_tokenizer,
        )
        print("\nAnswer:")
        print(answer)
        print("\nSources:")
        for source in sources:
            print(f"  - {source}")
        print()


if __name__ == "__main__":
    main()

