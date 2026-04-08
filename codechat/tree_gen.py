"""Project structure and dependency graph generation using rich Tree."""

from __future__ import annotations

import os
from pathlib import Path
from rich.tree import Tree
from rich.text import Text
from rich.console import Console

from .scanner import scan_files, _should_skip_dir, _load_ignore_patterns
from .ast_chunker import get_language_for_file, _get_parser, _get_language


def _extract_file_info(file_path: Path) -> dict:
    """Extract top-level classes, functions, and imports using AST or fallback regex."""
    lang_name = get_language_for_file(str(file_path))
    result = {"classes": [], "functions": [], "imports": []}
    
    if not lang_name:
        return result
        
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return result

    try:
        parser = _get_parser(lang_name)
        lang_obj = _get_language(lang_name) if parser else None
    except Exception:
        parser = None
        lang_obj = None

    # Use Tree-sitter if available
    if parser and lang_obj:
        try:
            tree = parser.parse(content.encode("utf-8"))
            if lang_name == "python":
                query = lang_obj.query("""
                    (class_definition name: (identifier) @class)
                    (function_definition name: (identifier) @function)
                    (import_statement name: (dotted_name) @import)
                    (import_from_statement module_name: (dotted_name) @import_from)
                """)
                for node, tag in query.captures(tree.root_node):
                    text = node.text.decode("utf-8")
                    if tag == "class": result["classes"].append(text)
                    elif tag == "function": result["functions"].append(text)
                    elif tag in ("import", "import_from"): result["imports"].append(text)
            
            elif lang_name in ("javascript", "typescript", "tsx"):
                query = lang_obj.query("""
                    (class_declaration name: (identifier) @class)
                    (function_declaration name: (identifier) @function)
                    (import_statement source: (string) @import)
                    (call_expression function: (identifier) @req (#eq? @req "require") arguments: (arguments (string) @import))
                """)
                for node, tag in query.captures(tree.root_node):
                    text = node.text.decode("utf-8").strip("'\"")
                    if tag == "class": result["classes"].append(text)
                    elif tag == "function": result["functions"].append(text)
                    elif tag == "import": result["imports"].append(text)

            elif lang_name == "go":
                query = lang_obj.query("""
                    (type_declaration (type_spec name: (type_identifier) @class))
                    (function_declaration name: (identifier) @function)
                    (method_declaration name: (field_identifier) @function)
                    (import_spec path: (interpreted_string_literal) @import)
                """)
                for node, tag in query.captures(tree.root_node):
                    text = node.text.decode("utf-8").strip('"')
                    if tag == "class": result["classes"].append(text)
                    elif tag == "function": result["functions"].append(text)
                    elif tag == "import": result["imports"].append(text)

            elif lang_name == "rust":
                query = lang_obj.query("""
                    (struct_item name: (type_identifier) @class)
                    (enum_item name: (type_identifier) @class)
                    (trait_item name: (type_identifier) @class)
                    (function_item name: (identifier) @function)
                    (use_declaration argument: (_) @import)
                """)
                for node, tag in query.captures(tree.root_node):
                    text = node.text.decode("utf-8")
                    if tag == "class": result["classes"].append(text)
                    elif tag == "function": result["functions"].append(text)
                    elif tag == "import": result["imports"].append(text)

            # Deduplicate and limit symbols
            result["classes"] = list(dict.fromkeys(result["classes"]))[:15]
            result["functions"] = list(dict.fromkeys(result["functions"]))[:15]
            result["imports"] = list(dict.fromkeys(result["imports"]))
            return result
        except Exception:
            pass # Fallback to regex if query fails

    # Fallback to simple regex if query is not available or too complex
    import re
    if lang_name == "python":
        for line in content.splitlines():
            m_cls = re.match(r"^class\s+(\w+)", line)
            if m_cls: result["classes"].append(m_cls.group(1))
            m_func = re.match(r"^(?:async\s+)?def\s+(\w+)", line)
            if m_func: result["functions"].append(m_func.group(1))
            m_imp = re.match(r"^import\s+([a-zA-Z0-9_.]+)", line)
            if m_imp: result["imports"].append(m_imp.group(1))
            m_imp_from = re.match(r"^from\s+([a-zA-Z0-9_.]+)\s+import", line)
            if m_imp_from: result["imports"].append(m_imp_from.group(1))
            
    elif lang_name in ("javascript", "typescript", "tsx"):
        for line in content.splitlines():
            m_cls = re.match(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)", line)
            if m_cls: result["classes"].append(m_cls.group(1))
            m_func = re.match(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)", line)
            if m_func: result["functions"].append(m_func.group(1))
            m_imp = re.search(r"import\s+.*from\s+['\"]([^'\"]+)['\"]", line)
            if m_imp: result["imports"].append(m_imp.group(1))
            m_req = re.search(r"require\(['\"]([^'\"]+)['\"]\)", line)
            if m_req: result["imports"].append(m_req.group(1))
            
    elif lang_name == "go":
        for line in content.splitlines():
            m_cls = re.match(r"^type\s+(\w+)\s+(?:struct|interface)", line)
            if m_cls: result["classes"].append(m_cls.group(1))
            m_func = re.match(r"^func\s+(?:\([^)]+\)\s+)?(\w+)", line)
            if m_func: result["functions"].append(m_func.group(1))
            
    elif lang_name == "rust":
        for line in content.splitlines():
            m_cls = re.match(r"^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", line)
            if m_cls: result["classes"].append(m_cls.group(1))
            m_func = re.match(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", line)
            if m_func: result["functions"].append(m_func.group(1))
            m_imp = re.match(r"^use\s+([^;]+);", line)
            if m_imp: result["imports"].append(m_imp.group(1))

    result["classes"] = list(dict.fromkeys(result["classes"]))[:15]
    result["functions"] = list(dict.fromkeys(result["functions"]))[:15]
    result["imports"] = list(dict.fromkeys(result["imports"]))
    return result


def build_project_tree(project_root: Path, show_symbols: bool = False) -> Tree:
    """
    Build a Rich Tree representing the project directory structure.
    If show_symbols is True, it will parse files and attach classes/functions as leaves.
    """
    project_root = project_root.resolve()
    ignore_patterns = _load_ignore_patterns(project_root)
    
    root_name = project_root.name
    tree = Tree(
        f"[bold blue]📁 {root_name}[/]",
        guide_style="bold bright_black",
    )

    def add_node(dir_path: Path, current_tree: Tree):
        try:
            entries = sorted(os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            # Skip hidden files/dirs and ignored dirs
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and _should_skip_dir(entry.name):
                continue
                
            rel_path = Path(entry.path).relative_to(project_root)
            
            # Check gitignore
            if ignore_patterns and ignore_patterns.match_file(str(rel_path)):
                continue

            if entry.is_dir():
                branch = current_tree.add(f"[bold cyan]📁 {entry.name}[/]")
                add_node(Path(entry.path), branch)
            else:
                # File node
                text = Text(f"📄 {entry.name}", style="green")
                file_size = entry.stat().st_size
                size_str = f"{file_size:,}B" if file_size < 1024 else f"{file_size/1024:.1f}KB"
                text.append(f" ({size_str})", style="dim")
                
                file_node = current_tree.add(text)
                
                if show_symbols:
                    info = _extract_file_info(Path(entry.path))
                    for cls in info["classes"]:
                        file_node.add(Text(f"📦 {cls}", style="yellow"))
                    for func in info["functions"]:
                        file_node.add(Text(f"ƒ {func}", style="magenta"))

    add_node(project_root, tree)
    return tree

def _is_internal_dep(imp: str, project_name: str) -> bool:
    """Heuristic to determine if an import is an internal project dependency."""
    imp = imp.strip()
    if imp.startswith("."):
        return True
    if imp.startswith("@/") or imp.startswith("~/"):
        return True
    # Python/JS/Go absolute imports starting with project name
    if imp.startswith(project_name):
        return True
    if "/" in imp and project_name in imp.split("/"):
        return True
    return False

def build_dependency_graph(project_root: Path, internal_only: bool = False) -> Tree:
    """
    Build a Rich Tree representing the project's dependency graph.
    Shows what each file imports.
    """
    project_root = project_root.resolve()
    ignore_patterns = _load_ignore_patterns(project_root)
    
    root_name = project_root.name
    tree = Tree(
        f"[bold blue]🕸️ {root_name} Dependencies[/]",
        guide_style="bold bright_black",
    )

    def add_node(dir_path: Path, current_tree: Tree):
        try:
            entries = sorted(os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and _should_skip_dir(entry.name):
                continue
                
            rel_path = Path(entry.path).relative_to(project_root)
            
            if ignore_patterns and ignore_patterns.match_file(str(rel_path)):
                continue

            if entry.is_dir():
                branch = current_tree.add(f"[bold cyan]📁 {entry.name}[/]")
                add_node(Path(entry.path), branch)
            else:
                info = _extract_file_info(Path(entry.path))
                imports = info.get("imports", [])
                
                lang_name = get_language_for_file(str(entry.path))
                if lang_name:
                    text = Text(f"📄 {entry.name}", style="green")
                    file_node = current_tree.add(text)
                    
                    added_any = False
                    if imports:
                        for imp in imports:
                            is_internal = _is_internal_dep(imp, root_name)
                            if internal_only and not is_internal:
                                continue
                            
                            style = "bold cyan" if is_internal else "dim white"
                            icon = "🔗" if is_internal else "📦"
                            file_node.add(Text(f"{icon} {imp}", style=style))
                            added_any = True
                            
                    if not added_any:
                        file_node.add(Text("(no dependencies)", style="dim italic"))

    add_node(project_root, tree)
    return tree

def generate_mermaid_graph(project_root: Path, internal_only: bool = False) -> str:
    """Generate a Mermaid.js directed graph of the project dependencies."""
    project_root = project_root.resolve()
    ignore_patterns = _load_ignore_patterns(project_root)
    root_name = project_root.name
    
    lines = ["graph TD", "    %% Nodes"]
    
    edges = []
    node_map = {}
    node_counter = 0
    
    def get_node_id(name: str) -> str:
        nonlocal node_counter
        if name not in node_map:
            node_map[name] = f"N{node_counter}"
            node_counter += 1
            # Add node definition
            safe_name = name.replace('"', '').replace("'", "")
            lines.append(f"    {node_map[name]}[{safe_name}]")
        return node_map[name]

    for root, dirs, files in os.walk(project_root):
        # Filter dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and not _should_skip_dir(d)]
        
        for file in files:
            if file.startswith("."): continue
            file_path = Path(root) / file
            rel_path = file_path.relative_to(project_root)
            
            if ignore_patterns and ignore_patterns.match_file(str(rel_path)):
                continue
                
            lang_name = get_language_for_file(str(file_path))
            if not lang_name:
                continue
                
            info = _extract_file_info(file_path)
            imports = info.get("imports", [])
            
            if imports:
                source_id = get_node_id(file)
                for imp in imports:
                    is_internal = _is_internal_dep(imp, root_name)
                    if internal_only and not is_internal:
                        continue
                    
                    target_id = get_node_id(imp)
                    # Different arrow style for internal vs external
                    arrow = "-->" if is_internal else "-.->"
                    edges.append(f"    {source_id} {arrow} {target_id}")

    lines.append("\n    %% Edges")
    lines.extend(edges)
    
    # Styling for nodes
    lines.append("\n    %% Styling")
    lines.append("    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;")
    lines.append("    classDef external fill:#eef,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5;")
    
    for name, nid in node_map.items():
        if not name.endswith((".py", ".js", ".ts", ".go", ".rs")) and not _is_internal_dep(name, root_name):
            lines.append(f"    class {nid} external;")

    return "\n".join(lines)

