"""Repository map and symbol graph utilities for agent grounding."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .ast_chunker import get_language_for_file
from .config import get_snowcode_dir
from .scanner import scan_files
from .tree_gen import _extract_file_info, _is_internal_dep


_REPO_MAP_VERSION = 1
_MODULE_SUFFIXES = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs")
_INDEX_FILES = ("index.js", "index.jsx", "index.ts", "index.tsx")


@dataclass
class RepoFileRecord:
    """Compact per-file entry used by the repository map."""

    path: str
    language: str
    size: int
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    internal_deps: list[str] = field(default_factory=list)
    reverse_deps: list[str] = field(default_factory=list)

    @property
    def symbol_count(self) -> int:
        return len(self.classes) + len(self.functions)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "size": self.size,
            "classes": self.classes,
            "functions": self.functions,
            "imports": self.imports,
            "internal_deps": self.internal_deps,
            "reverse_deps": self.reverse_deps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RepoFileRecord":
        return cls(
            path=str(data.get("path", "")),
            language=str(data.get("language", "text")),
            size=int(data.get("size", 0)),
            classes=list(data.get("classes", [])),
            functions=list(data.get("functions", [])),
            imports=list(data.get("imports", [])),
            internal_deps=list(data.get("internal_deps", [])),
            reverse_deps=list(data.get("reverse_deps", [])),
        )


class RepositoryMap:
    """Build and cache a lightweight repository-wide symbol/dependency map."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.codechat_dir = get_snowcode_dir(self.project_root)
        self.cache_path = self.codechat_dir / "repo_map.json"
        self._snapshot: list[RepoFileRecord] | None = None

    def get_snapshot(self, force: bool = False) -> list[RepoFileRecord]:
        """Return the cached snapshot, rebuilding it if files changed."""
        if self._snapshot is not None and not force:
            return self._snapshot

        files = scan_files(self.project_root)
        signatures = self._compute_signatures(files)

        if not force:
            payload = self._load_cache()
            if payload and self._is_cache_valid(payload, signatures):
                records = [RepoFileRecord.from_dict(item) for item in payload.get("files", [])]
                self._snapshot = records
                return records

        records = self._build_snapshot(files)
        self._snapshot = records
        self._save_cache(signatures, records)
        return records

    def prompt_context(self, query: str, max_files: int = 4, max_symbols: int = 3) -> str:
        """Return a compact context block for agent prompts."""
        snapshot = self.get_snapshot()
        if not snapshot:
            return ""

        lines = ["## Repository Map"]
        languages = Counter(record.language for record in snapshot)
        top_languages = ", ".join(f"{name}:{count}" for name, count in languages.most_common(3))
        lines.append(f"Files indexed: {len(snapshot)} | Languages: {top_languages}")

        matches = self.find_symbols(query, limit=max_files)
        if matches:
            lines.append("Relevant symbols:")
            for match in matches[:max_files]:
                lines.append(
                    f"- {match['kind']} `{match['name']}` in `{match['file_path']}`"
                )
            return "\n".join(lines) + "\n"

        hotspots = sorted(
            snapshot,
            key=lambda item: (len(item.reverse_deps), item.symbol_count, item.size),
            reverse=True,
        )[:max_files]
        lines.append("Hot files:")
        for record in hotspots:
            symbols = ", ".join((record.classes + record.functions)[:max_symbols]) or "no top-level symbols"
            lines.append(f"- `{record.path}` [{record.language}] {symbols}")
        return "\n".join(lines) + "\n"

    def render(self, focus: str = "", max_files: int = 8, max_symbols: int = 4) -> str:
        """Render a human-readable repository map or symbol graph."""
        snapshot = self.get_snapshot()
        if not snapshot:
            return "Repository map unavailable."

        focus = (focus or "").strip()
        if focus:
            symbol_graph = self.render_symbol_graph(focus, limit=max_files)
            if symbol_graph:
                return symbol_graph

        lines = [f"Repository map for `{self.project_root.name}`"]
        lines.append(f"Indexed files: {len(snapshot)}")

        languages = Counter(record.language for record in snapshot)
        if languages:
            lines.append(
                "Languages: " + ", ".join(f"{name}={count}" for name, count in languages.most_common(5))
            )

        directories = Counter(Path(record.path).parts[0] if Path(record.path).parts else "." for record in snapshot)
        if directories:
            lines.append(
                "Top directories: " + ", ".join(f"{name}={count}" for name, count in directories.most_common(5))
            )

        focus_lower = focus.lower()
        if focus:
            selected = [
                record for record in snapshot
                if focus_lower in record.path.lower() or focus_lower in str(Path(record.path).parent).lower()
            ]
            if selected:
                lines.append(f"Focus matched `{focus}`:")
            else:
                lines.append(f"No direct symbol match for `{focus}`; showing general hotspots.")
                selected = []
        else:
            selected = []

        if not selected:
            selected = sorted(
                snapshot,
                key=lambda item: (len(item.reverse_deps), item.symbol_count, item.size),
                reverse=True,
            )[:max_files]
        else:
            selected = selected[:max_files]

        for record in selected:
            symbols = ", ".join((record.classes + record.functions)[:max_symbols]) or "no top-level symbols"
            deps = ", ".join(record.internal_deps[:3]) or "no internal deps"
            imported_by = len(record.reverse_deps)
            lines.append(
                f"- `{record.path}` [{record.language}] symbols: {symbols} | deps: {deps} | imported by: {imported_by}"
            )

        return "\n".join(lines)

    def render_symbol_graph(self, query: str, limit: int = 6) -> str:
        """Render symbol matches plus their local dependency context."""
        matches = self.find_symbols(query, limit=limit)
        if not matches:
            return ""

        by_path = {record.path: record for record in self.get_snapshot()}
        lines = [f"Symbol graph for `{query}`"]
        for match in matches:
            record = by_path.get(match["file_path"])
            if not record:
                continue

            siblings = [name for name in (record.classes + record.functions) if name != match["name"]][:4]
            lines.append(f"- {match['kind']} `{match['name']}` in `{record.path}`")
            lines.append(f"  siblings: {', '.join(siblings) if siblings else 'none'}")
            lines.append(f"  internal deps: {', '.join(record.internal_deps[:4]) if record.internal_deps else 'none'}")
            lines.append(f"  imported by: {', '.join(record.reverse_deps[:4]) if record.reverse_deps else 'none'}")
        return "\n".join(lines)

    def find_symbols(self, query: str, limit: int = 8) -> list[dict]:
        """Find matching classes/functions/files for a free-form query."""
        query = (query or "").strip().lower()
        if not query:
            return []

        hits: list[tuple[int, dict]] = []
        for record in self.get_snapshot():
            for name in record.classes:
                score = self._score_name(name, query)
                if score:
                    hits.append((score + 20, {"kind": "class", "name": name, "file_path": record.path}))

            for name in record.functions:
                score = self._score_name(name, query)
                if score:
                    hits.append((score + 10, {"kind": "function", "name": name, "file_path": record.path}))

            path_score = self._score_name(record.path, query)
            if path_score:
                hits.append((path_score, {"kind": "file", "name": Path(record.path).name, "file_path": record.path}))

        hits.sort(key=lambda item: (-item[0], item[1]["file_path"], item[1]["name"]))

        seen: set[tuple[str, str, str]] = set()
        results: list[dict] = []
        for _, hit in hits:
            key = (hit["kind"], hit["name"], hit["file_path"])
            if key in seen:
                continue
            seen.add(key)
            results.append(hit)
            if len(results) >= limit:
                break
        return results

    def _build_snapshot(self, files: list[Path]) -> list[RepoFileRecord]:
        records: list[RepoFileRecord] = []
        module_index: defaultdict[str, set[str]] = defaultdict(set)

        for file_path in files:
            rel = file_path.relative_to(self.project_root).as_posix()
            info = _extract_file_info(file_path)
            record = RepoFileRecord(
                path=rel,
                language=get_language_for_file(str(file_path)) or (file_path.suffix.lower().lstrip(".") or "text"),
                size=file_path.stat().st_size,
                classes=info.get("classes", []),
                functions=info.get("functions", []),
                imports=info.get("imports", []),
            )
            records.append(record)

            for module_name in self._module_names_for_path(rel):
                module_index[module_name].add(rel)

        for record in records:
            deps: set[str] = set()
            for imp in record.imports:
                deps.update(self._resolve_internal_import(record.path, imp, module_index))
            record.internal_deps = sorted(dep for dep in deps if dep != record.path)

        reverse_map: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            for dep in record.internal_deps:
                reverse_map[dep].add(record.path)

        for record in records:
            record.reverse_deps = sorted(reverse_map.get(record.path, set()))

        records.sort(key=lambda item: item.path)
        return records

    def _module_names_for_path(self, rel_path: str) -> list[str]:
        path = Path(rel_path)
        if path.name == "__init__.py":
            base = path.parent.as_posix().replace("/", ".")
            return [base] if base else []

        stem = path.with_suffix("").as_posix().replace("/", ".")
        return [stem] if stem else []

    def _resolve_internal_import(
        self,
        source_path: str,
        imp: str,
        module_index: dict[str, set[str]],
    ) -> set[str]:
        imp = imp.strip().strip("'\"")
        if not imp:
            return set()

        results: set[str] = set()
        if imp.startswith("."):
            results.update(self._resolve_relative_python_import(source_path, imp, module_index))
        elif imp.startswith("./") or imp.startswith("../"):
            results.update(self._resolve_relative_path_import(source_path, imp))
        elif imp.startswith("@/") or imp.startswith("~/"):
            results.update(self._expand_path_candidates(Path(imp[2:])))
        else:
            if _is_internal_dep(imp, self.project_root.name):
                results.update(module_index.get(imp, set()))
                results.update(self._expand_path_candidates(Path(*imp.split("."))))
                results.update(self._expand_path_candidates(Path(imp)))

        return {path for path in results if (self.project_root / path).exists()}

    def _resolve_relative_python_import(
        self,
        source_path: str,
        imp: str,
        module_index: dict[str, set[str]],
    ) -> set[str]:
        source_module = Path(source_path).with_suffix("").as_posix().replace("/", ".")
        if source_module.endswith(".__init__"):
            source_module = source_module[: -len(".__init__")]

        dots = len(imp) - len(imp.lstrip("."))
        remainder = imp[dots:]
        base_parts = [part for part in source_module.split(".") if part]
        if Path(source_path).name != "__init__.py":
            base_parts = base_parts[:-1]

        up_levels = max(dots - 1, 0)
        if up_levels:
            base_parts = base_parts[: max(0, len(base_parts) - up_levels)]

        if remainder:
            target_module = ".".join(base_parts + remainder.split("."))
        else:
            target_module = ".".join(base_parts)

        return set(module_index.get(target_module, set()))

    def _resolve_relative_path_import(self, source_path: str, imp: str) -> set[str]:
        source_dir = Path(source_path).parent
        try:
            target = (self.project_root / source_dir / imp).resolve().relative_to(self.project_root)
        except Exception:
            return set()
        return self._expand_path_candidates(target)

    def _expand_path_candidates(self, relative_path: Path) -> set[str]:
        results: set[str] = set()
        rel = Path(str(relative_path).replace("\\", "/"))
        candidates = {rel}

        if rel.suffix:
            candidates.add(rel.with_suffix(rel.suffix))
        else:
            for suffix in _MODULE_SUFFIXES:
                candidates.add(rel.with_suffix(suffix))
            for index_name in _INDEX_FILES:
                candidates.add(rel / index_name)
            candidates.add(rel / "__init__.py")

        for candidate in candidates:
            full = self.project_root / candidate
            if full.exists() and full.is_file():
                results.add(candidate.as_posix())
        return results

    def _compute_signatures(self, files: list[Path]) -> dict[str, str]:
        signatures = {}
        for file_path in files:
            stat = file_path.stat()
            rel = file_path.relative_to(self.project_root).as_posix()
            signatures[rel] = f"{stat.st_mtime_ns}:{stat.st_size}"
        return signatures

    def _load_cache(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, signatures: dict[str, str], records: list[RepoFileRecord]) -> None:
        self.codechat_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REPO_MAP_VERSION,
            "generated_at": time.time(),
            "signatures": signatures,
            "files": [record.to_dict() for record in records],
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _is_cache_valid(self, payload: dict, signatures: dict[str, str]) -> bool:
        return (
            payload.get("version") == _REPO_MAP_VERSION
            and isinstance(payload.get("files"), list)
            and payload.get("signatures") == signatures
        )

    @staticmethod
    def _score_name(candidate: str, query: str) -> int:
        haystack = candidate.lower()
        if haystack == query:
            return 100
        if haystack.endswith(f".{query}") or haystack.endswith(f"/{query}"):
            return 80
        if query in haystack:
            return 50 - min(len(haystack) - len(query), 20)
        query_tokens = [token for token in query.replace("/", " ").replace(".", " ").split() if token]
        if query_tokens and all(token in haystack for token in query_tokens):
            return 25
        return 0
