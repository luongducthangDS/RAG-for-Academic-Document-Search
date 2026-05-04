from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CHUNK_ANALYSIS_PATH, markdown_dirs
from pipelines.chunking import analyze_chunk_outputs, chunk_directories, save_chunk_analysis


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Chunk Markdown files into RAG JSON chunks.")
    parser.add_argument("--md-dir", action="append", type=Path, dest="md_dirs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--analysis-out", type=Path, default=CHUNK_ANALYSIS_PATH)
    args = parser.parse_args()

    outputs = chunk_directories(
        md_dirs=args.md_dirs or markdown_dirs(),
        skip_existing=args.skip_existing,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    for item in outputs:
        print(item)

    analysis = analyze_chunk_outputs(
        chunk_json_paths=outputs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    analysis_path = save_chunk_analysis(analysis, output_path=args.analysis_out)
    print(analysis_path)
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
