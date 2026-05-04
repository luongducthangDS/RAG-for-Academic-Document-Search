from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from core.config import CHUNK_ANALYSIS_PATH, PROJECT_ROOT, markdown_dirs


def flatten_markdown_table(table_text: str) -> str:
    lines = [line.strip() for line in table_text.strip().splitlines()]
    if len(lines) < 3:
        return table_text

    headers = [header.strip() for header in lines[0].split("|") if header.strip()]
    rows = ["", "[DỮ LIỆU BẢNG QUY ĐỔI]:"]

    for line in lines[2:]:
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        semantics = []
        for header, cell in zip(headers, cells):
            if cell and cell != "-":
                semantics.append(f"{header}: {cell}")
        if semantics:
            rows.append(f"- {', '.join(semantics)}.")

    return "\n".join(rows) + "\n"


def process_tables_in_text(markdown_text: str) -> str:
    table_pattern = re.compile(r"(\|.+?\|\n\|[-:\s|]+\|\n(?:\|.+?\|\n)+)")
    tables = table_pattern.findall(markdown_text)
    for table in tables:
        markdown_text = markdown_text.replace(table, flatten_markdown_table(table))
    return markdown_text


def _relative_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def simple_token_count(text: str) -> int:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return 0
    return len(normalized.split(" "))


def chunk_markdown_file(
    md_file_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> Path:
    md_file_path = Path(md_file_path)
    markdown_text = md_file_path.read_text(encoding="utf-8")
    markdown_text = process_tables_in_text(markdown_text)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )
    final_splits = text_splitter.split_documents(header_splits)

    file_base_name = md_file_path.stem
    processed_chunks = []

    for index, split in enumerate(final_splits, start=1):
        headers_context = " - ".join(value for value in split.metadata.values())
        if headers_context:
            context_prefix = f"[Tài liệu: {file_base_name} | Mục: {headers_context}]\n"
        else:
            context_prefix = f"[Tài liệu: {file_base_name}]\n"

        processed_chunks.append(
            {
                "chunk_id": f"{file_base_name}_chunk_{index}",
                "metadata": {
                    "source": _relative_source_path(md_file_path),
                    **split.metadata,
                },
                "content": context_prefix + split.page_content,
            }
        )

    output_json_path = md_file_path.with_name(f"{md_file_path.stem}_chunks.json")
    output_json_path.write_text(
        json.dumps(processed_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_json_path


def chunk_directories(
    md_dirs: list[Path] | None = None,
    skip_existing: bool = False,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Path]:
    md_dirs = md_dirs or markdown_dirs()
    outputs: list[Path] = []

    for directory in md_dirs:
        if not directory.exists():
            continue
        for md_file in sorted(directory.glob("*.md")):
            output_json_path = md_file.with_name(f"{md_file.stem}_chunks.json")
            if skip_existing and output_json_path.exists():
                outputs.append(output_json_path)
                continue
            outputs.append(
                chunk_markdown_file(
                    md_file,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

    return outputs


def analyze_chunk_outputs(
    chunk_json_paths: list[Path],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> dict:
    original_documents = 0
    total_chunks = 0
    token_lengths: list[int] = []
    source_files: set[str] = set()

    for chunk_json_path in chunk_json_paths:
        if not chunk_json_path.exists():
            continue

        original_documents += 1
        chunks = json.loads(chunk_json_path.read_text(encoding="utf-8"))
        total_chunks += len(chunks)

        for chunk in chunks:
            source = str((chunk.get("metadata") or {}).get("source", "")).strip()
            if source:
                source_files.add(source)
            token_lengths.append(simple_token_count(chunk.get("content", "")))

    avg_length = round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0.0
    min_length = min(token_lengths) if token_lengths else 0
    max_length = max(token_lengths) if token_lengths else 0

    rows = [
        {"Thuộc tính": "Số tài liệu gốc", "Giá trị": original_documents},
        {"Thuộc tính": "Tổng số chunk", "Giá trị": total_chunks},
        {"Thuộc tính": "Chunk size (token)", "Giá trị": chunk_size},
        {"Thuộc tính": "Overlap (token)", "Giá trị": chunk_overlap},
        {"Thuộc tính": "Độ dài chunk trung bình", "Giá trị": avg_length},
        {"Thuộc tính": "Độ dài chunk min/max", "Giá trị": {"min": min_length, "max": max_length}},
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unit_note": (
            "Chunk size/overlap là giá trị cấu hình truyền vào pipeline. "
            "Độ dài chunk được ước lượng theo số từ tách bằng khoảng trắng trên nội dung chunk đã lưu."
        ),
        "summary": rows,
        "details": {
            "original_documents": original_documents,
            "unique_sources_from_metadata": len(source_files),
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "average_chunk_length": avg_length,
            "min_chunk_length": min_length,
            "max_chunk_length": max_length,
            "all_chunk_json_files": [str(path) for path in chunk_json_paths if path.exists()],
        },
    }


def save_chunk_analysis(
    analysis: dict,
    output_path: str | Path = CHUNK_ANALYSIS_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return output_path
