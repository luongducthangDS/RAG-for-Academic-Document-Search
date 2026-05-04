from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PDF_DIR, SCAN_DIR
from pipelines.extract import run_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF files to Markdown.")
    parser.add_argument("--text-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--scan-dir", type=Path, default=SCAN_DIR)
    parser.add_argument("--api-key", action="append", dest="api_keys")
    parser.add_argument("--clean-text-with-ai", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = run_extraction(
        text_pdf_dir=args.text_dir,
        scan_pdf_dir=args.scan_dir,
        api_keys=args.api_keys,
        clean_text_with_ai=args.clean_text_with_ai,
        skip_existing=not args.overwrite,
    )

    print("Text outputs:")
    for item in results["text"]:
        print(f"  - {item}")
    print("Scan outputs:")
    for item in results["scan"]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()

