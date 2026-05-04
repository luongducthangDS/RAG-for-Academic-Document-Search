from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CHROMA_DIR, chunk_json_dirs
from pipelines.vectorstore import build_vector_db_from_chunk_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the Chroma vector database.")
    parser.add_argument("--chunk-dir", action="append", type=Path, dest="chunk_dirs")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    build_vector_db_from_chunk_dirs(
        json_folder_paths=args.chunk_dirs or chunk_json_dirs(),
        persist_directory=args.db_dir,
        device=args.device,
        clear_existing=not args.append,
        require_gpu=True,
    )
    print(f"Vector DB ready at {args.db_dir}")


if __name__ == "__main__":
    main()
