from __future__ import annotations

import io
import os
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

from core.config import DEFAULT_GEMINI_MODEL, PDF_DIR, SCAN_DIR


def _resolve_api_keys(api_keys: str | Sequence[str] | None) -> list[str]:
    if isinstance(api_keys, str):
        return [api_keys]
    if api_keys:
        return [key for key in api_keys if key]

    env_value = (
        os.getenv("GEMINI_API_KEYS")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not env_value:
        return []
    return [part.strip() for part in env_value.split(",") if part.strip()]


def clean_markdown_with_gemini(
    markdown_text: str,
    api_keys: str | Sequence[str],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    from google import genai
    from google.genai import types

    keys = _resolve_api_keys(api_keys)
    if not keys:
        return markdown_text

    prompt = (
        "You are a Vietnamese document cleanup assistant.\n"
        "Rules:\n"
        "1. Fix broken Vietnamese words caused by bad spacing.\n"
        "2. Merge broken Markdown headers into one meaningful line.\n"
        "3. Preserve tables and lists.\n"
        "4. Do not add, remove, or summarize content.\n"
        "5. Return only cleaned Markdown."
    )

    try:
        client = genai.Client(api_key=keys[0])
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, markdown_text],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text or markdown_text
    except Exception:
        return markdown_text


def extract_text_pdf_with_docling(
    pdf_path: str | Path,
    output_md_path: str | Path | None = None,
    api_keys: str | Sequence[str] | None = None,
    clean_with_ai: bool = False,
) -> Path:
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    md_path = Path(output_md_path) if output_md_path else pdf_path.parent / "md" / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown_text = result.document.export_to_markdown()

    if clean_with_ai:
        markdown_text = clean_markdown_with_gemini(markdown_text, api_keys or [])

    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


def extract_scan_pdf_with_vlm(
    pdf_path: str | Path,
    api_keys: str | Sequence[str],
    output_md_path: str | Path | None = None,
    dpi: int = 300,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Path:
    from google import genai
    from google.genai import types
    from pdf2image import convert_from_path

    pdf_path = Path(pdf_path)
    md_path = Path(output_md_path) if output_md_path else pdf_path.parent / "md" / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    keys = _resolve_api_keys(api_keys)
    if not keys:
        raise ValueError("Gemini API keys are required for scan PDF extraction.")

    images = convert_from_path(str(pdf_path), dpi=dpi)
    current_key = 0
    client = genai.Client(api_key=keys[current_key])

    prompt = (
        "Convert this Vietnamese document page to Markdown.\n"
        "Rules:\n"
        "1. Preserve tables using Markdown tables.\n"
        "2. Keep all numbers and legal references.\n"
        "3. Use Markdown headings for sections.\n"
        "4. Merge broken headings into one line.\n"
        "5. Ignore decorative page numbers and repeated headers or footers.\n"
        "6. Fix spacing issues in Vietnamese words."
    )

    pages: list[str] = []
    max_attempts = max(len(keys) * 2, 1)

    for page_index, image in enumerate(images, start=1):
        attempts = 0
        success = False

        while not success and attempts < max_attempts:
            try:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")

                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
                    ],
                    config=types.GenerateContentConfig(temperature=0.0),
                )

                page_text = response.text or "[No text returned]"
                pages.append(
                    f"<!-- Bắt đầu Trang {page_index} -->\n\n"
                    f"{page_text}\n\n"
                    f"<!-- Kết thúc Trang {page_index} -->\n\n---\n\n"
                )
                success = True
                time.sleep(2)
            except Exception as exc:
                attempts += 1
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    current_key = (current_key + 1) % len(keys)
                    client = genai.Client(api_key=keys[current_key])
                    if current_key == 0:
                        time.sleep(30)
                else:
                    pages.append(
                        f"<!-- Bắt đầu Trang {page_index} -->\n\n"
                        f"[Extraction error: {exc}]\n\n"
                        f"<!-- Kết thúc Trang {page_index} -->\n\n---\n\n"
                    )
                    success = True
                    traceback.print_exc()

    md_path.write_text("".join(pages), encoding="utf-8")
    return md_path


def _iter_pdf_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.pdf") if path.is_file())


def run_extraction(
    text_pdf_dir: str | Path = PDF_DIR,
    scan_pdf_dir: str | Path = SCAN_DIR,
    api_keys: str | Sequence[str] | None = None,
    clean_text_with_ai: bool = False,
    skip_existing: bool = True,
) -> dict[str, list[str]]:
    text_pdf_dir = Path(text_pdf_dir)
    scan_pdf_dir = Path(scan_pdf_dir)
    keys = _resolve_api_keys(api_keys)

    results = {"text": [], "scan": []}

    for pdf_path in _iter_pdf_files(text_pdf_dir):
        output_path = pdf_path.parent / "md" / f"{pdf_path.stem}.md"
        if skip_existing and output_path.exists():
            results["text"].append(str(output_path))
            continue
        results["text"].append(
            str(
                extract_text_pdf_with_docling(
                    pdf_path,
                    output_md_path=output_path,
                    api_keys=keys,
                    clean_with_ai=clean_text_with_ai,
                )
            )
        )

    for pdf_path in _iter_pdf_files(scan_pdf_dir):
        output_path = pdf_path.parent / "md" / f"{pdf_path.stem}.md"
        if skip_existing and output_path.exists():
            results["scan"].append(str(output_path))
            continue
        if not keys:
            raise ValueError("Scan pipeline requires GEMINI_API_KEY or GEMINI_API_KEYS.")
        results["scan"].append(
            str(extract_scan_pdf_with_vlm(pdf_path, api_keys=keys, output_md_path=output_path))
        )

    return results
