"""File scanning - discover and read code files from a project."""

import fnmatch
import os
from pathlib import Path

import pathspec

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, CONFIG_EXTENSIONS, MAX_FILE_SIZE, SKIP_DIRS

ALL_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS


def _load_ignore_patterns(project_root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore and .codechatignore patterns if available."""
    lines = []
    
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        lines.extend(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
        
    codechatignore = project_root / ".codechatignore"
    if codechatignore.exists():
        lines.extend(codechatignore.read_text(encoding="utf-8", errors="ignore").splitlines())
        
    # Clean up lines
    valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        
    if valid_lines:
        return pathspec.PathSpec.from_lines("gitignore", valid_lines)
    return None


def _should_skip_dir(dirname: str) -> bool:
    """Check if directory should be skipped."""
    if dirname.startswith(".") and dirname not in {".github", ".gitlab"}:
        return True
    for pattern in SKIP_DIRS:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    return False


def scan_files(project_root: Path, extra_extensions: set[str] | None = None) -> list[Path]:
    """
    Scan a project directory and return all code/doc files.

    Uses os.walk with directory pruning for efficient traversal.
    Respects .gitignore, skips binary/build directories.
    """
    project_root = project_root.resolve()
    ignore_patterns = _load_ignore_patterns(project_root)
    extensions = ALL_EXTENSIONS | (extra_extensions or set())

    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(str(project_root)):
        # Prune directories in-place (os.walk supports this)
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(d) and d != ".codechat"
        ]

        rel_dir = Path(dirpath).relative_to(project_root)

        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = rel_dir / fname

            # Check ignore patterns
            if ignore_patterns and ignore_patterns.match_file(str(rel)):
                continue

            # Check extension
            if fpath.suffix.lower() not in extensions:
                continue

            # Check file size
            try:
                size = fpath.stat().st_size
                if size == 0 or size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            files.append(fpath)

    return sorted(files)


def read_file(path: Path) -> str | None:
    """Read a file, returning None if it's binary or unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        return text
    except (UnicodeDecodeError, OSError):
        return None
