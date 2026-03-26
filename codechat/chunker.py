"""Code chunking - split code into semantic chunks for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Minimum lines per chunk — avoid 2-3 line fragments
MIN_CHUNK_LINES = 10
MIN_CHUNK_CHARS = 200


@dataclass
class Chunk:
    """A piece of code with metadata."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_index: int


def _split_by_lines(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split text into overlapping line-based chunks. Returns (content, start_line, end_line)."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(lines):
        end = start
        char_count = 0
        while end < len(lines) and char_count < chunk_size:
            char_count += len(lines[end]) + 1
            end += 1
        if end <= start:
            end = start + 1

        content = "\n".join(lines[start:end])
        if content.strip():
            chunks.append((content, start + 1, end))

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


# Strict patterns: only match actual top-level definitions
_FN_PATTERNS = [
    # Python
    re.compile(r"^def \w+\s*\("),
    re.compile(r"^async def \w+\s*\("),
    re.compile(r"^class \w+"),
    # JavaScript / TypeScript
    re.compile(r"^export\s+(default\s+)?(function|class|const|let|var)\s+\w+"),
    re.compile(r"^function\s+\w+\s*\("),
    re.compile(r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(.*\)\s*=>"),
    re.compile(r"^(const|let|var)\s+\w+\s*=\s*function\s*\("),
    # Go
    re.compile(r"^func\s+(\(\w+\s+\*\w+\)\s+)?\w+\s*\("),
    # Rust
    re.compile(r"^(pub\s+)?(async\s+)?fn\s+\w+"),
    re.compile(r"^impl\b"),
    re.compile(r"^pub\s+(struct|enum|trait|impl)\b"),
    # Java / Kotlin / C#
    re.compile(r"^(public|private|protected|internal)\s+(static\s+)?(async\s+)?[\w<>\[\]]+\s+\w+\s*\("),
    re.compile(r"^fun\s+\w+\s*\("),
    # C / C++
    re.compile(r"^[\w\s\*]+\w+\s*\([^)]*\)\s*\{"),
    re.compile(r"^class\s+\w+"),
    # Ruby
    re.compile(r"^def\s+\w+"),
    re.compile(r"^class\s+\w+"),
    re.compile(r"^module\s+\w+"),
    # PHP
    re.compile(r"^(public|private|protected)?\s*function\s+\w+\s*\("),
    # Shell
    re.compile(r"^\w+\s*\(\)\s*\{"),
]


def _is_fn_def(line: str) -> bool:
    """Check if a line is a top-level function/class definition."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _FN_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _merge_small_chunks(
    raw_chunks: list[tuple[str, int, int]],
    min_lines: int = MIN_CHUNK_LINES,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[tuple[str, int, int]]:
    """Merge tiny chunks with their neighbors."""
    if not raw_chunks:
        return []

    merged: list[tuple[str, int, int]] = []
    buf_content = ""
    buf_start = 0
    buf_end = 0

    for content, start, end in raw_chunks:
        if not buf_content:
            buf_content = content
            buf_start = start
            buf_end = end
            continue

        lines_in_buf = buf_end - buf_start + 1
        if lines_in_buf < min_lines or len(buf_content) < min_chars:
            # Too small, merge with next
            buf_content += "\n" + content
            buf_end = end
        else:
            merged.append((buf_content, buf_start, buf_end))
            buf_content = content
            buf_start = start
            buf_end = end

    if buf_content:
        # Last chunk: merge backward if too small
        lines_in_buf = buf_end - buf_start + 1
        if (lines_in_buf < min_lines or len(buf_content) < min_chars) and merged:
            prev_content, prev_start, prev_end = merged.pop()
            merged.append((prev_content + "\n" + buf_content, prev_start, buf_end))
        else:
            merged.append((buf_content, buf_start, buf_end))

    return merged


def _split_by_functions(text: str) -> list[tuple[str, int, int]]:
    """
    Split code by function/class definitions.
    Only splits at real definitions, merges tiny fragments.
    """
    lines = text.splitlines()
    if len(lines) < MIN_CHUNK_LINES:
        return [(text, 1, len(lines))]

    boundaries: list[int] = [0]
    for i, line in enumerate(lines):
        if _is_fn_def(line):
            # Don't split at line 0-1 (would create empty prefix)
            if i > 2:
                boundaries.append(i)

    if len(boundaries) <= 1:
        return [(text, 1, len(lines))]

    raw_chunks: list[tuple[str, int, int]] = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start:end])
        if content.strip():
            raw_chunks.append((content, start + 1, end))

    # Merge tiny fragments
    return _merge_small_chunks(raw_chunks)


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a file's content into chunks with metadata.

    Strategy: AST (Tree-sitter) → regex function splitter → line-based
    """
    lines = content.splitlines()
    total_lines = len(lines)

    # For small files, keep as single chunk
    if len(content) <= chunk_size:
        if content.strip():
            return [Chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                chunk_index=0,
            )]
        return []

    # Try AST-aware splitting first
    try:
        from .ast_chunker import ast_split_definitions
        ast_chunks = ast_split_definitions(file_path, content, chunk_size)
        if ast_chunks and len(ast_chunks) > 1:
            ast_final: list[Chunk] = []
            for sub_content, start_line, end_line in ast_chunks:
                if sub_content.strip():
                    ast_final.append(Chunk(
                        content=sub_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_index=len(ast_final),
                    ))
            if ast_final:
                return ast_final
    except Exception:
        pass

    # Fallback: regex function splitting
    fn_chunks = _split_by_functions(content)

    # For medium files, return function chunks directly
    if len(content) <= chunk_size * 3:
        fn_final: list[Chunk] = []
        for sub_content, start_line, end_line in fn_chunks:
            if sub_content.strip():
                fn_final.append(Chunk(
                    content=sub_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_index=len(fn_final),
                ))
        return fn_final

    # For large files: function splitting, then line-based for oversized chunks
    final_chunks: list[Chunk] = []
    for sub_content, start_line, end_line in fn_chunks:
        if len(sub_content) <= chunk_size * 1.5:
            if sub_content.strip():
                final_chunks.append(Chunk(
                    content=sub_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_index=len(final_chunks),
                ))
        else:
            for sub_s, sub_ss, sub_es in _split_by_lines(sub_content, chunk_size, overlap):
                if sub_s.strip():
                    final_chunks.append(Chunk(
                        content=sub_s,
                        file_path=file_path,
                        start_line=start_line + sub_ss - 1,
                        end_line=start_line + sub_es - 1,
                        chunk_index=len(final_chunks),
                    ))

    return final_chunks
