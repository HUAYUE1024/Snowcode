"""CLI - Command-line interface for snowcode."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text

from . import __version__

from .chunker import chunk_file
from .config import get_snowcode_dir, load_config, save_config, DEFAULT_EMBEDDING_MODEL, get_llm_config_from_file, save_llm_config
from .rag import answer_question, answer_question_stream, _get_llm_config
from .scanner import scan_files, read_file
from .agent import CodeAgent, CoordinatorAgent
from .agent_v2 import (
    create_agent as create_agent_v2,
    create_coordinator as create_coordinator_v2,
    build_default_registry as build_registry_v2,
)
from .skills import run_skill, run_skill_stream, SKILL_QUERIES
from .store import VectorStore

console = Console(legacy_windows=False)
_TOOL_CONFIRM_LOCK = threading.Lock()
_TOOL_AUTO_APPROVE = os.environ.get("SNOWCODE_AUTO_APPROVE", "").strip().lower() in {
    "1", "true", "yes", "y", "on",
}


def _set_tool_auto_approve(enabled: bool):
    """Configure whether tool confirmations should be auto-approved."""
    global _TOOL_AUTO_APPROVE
    _TOOL_AUTO_APPROVE = bool(enabled)


def _ask_tool_confirmation(prompt: str, default: bool = False) -> bool:
    """Read a confirmation choice robustly, with a Windows console fallback."""
    if _TOOL_AUTO_APPROVE:
        console.print("[dim]Auto-approved by CLI setting.[/]")
        return True

    if sys.stdin and sys.stdin.isatty():
        if sys.platform == "win32":
            try:
                import msvcrt

                default_label = "y" if default else "n"
                console.print(f"{prompt} [y/n] ({default_label}): ", end="")
                while True:
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        console.print(default_label)
                        return default
                    lower = ch.lower()
                    if lower in {"y", "n"}:
                        console.print(lower)
                        return lower == "y"
                    if ch == "\x03":
                        raise KeyboardInterrupt
            except (EOFError, KeyboardInterrupt):
                console.print()
                return default
            except Exception:
                pass

        try:
            return Confirm.ask(prompt, default=default, console=console)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return default

    console.print(
        "[dim]Interactive input is unavailable, so this confirmation defaults to deny. "
        "Use --auto-approve to skip prompts.[/]"
    )
    return default


def _confirm_tool_use(tool_name: str, params: dict, permission, details: str | None = None) -> bool:
    """Ask the user to approve a tool call at runtime."""
    with _TOOL_CONFIRM_LOCK:
        level = "dangerous" if getattr(permission, "value", "") == "dangerous" else "write"
        console.print()
        console.print(f"[bold yellow]Tool confirmation required[/]  [dim]({level})[/]")
        console.print(f"  [cyan]{tool_name}[/]")
        for key, value in params.items():
            preview = str(value)
            if len(preview) > 160:
                preview = preview[:157] + "..."
            console.print(f"  [dim]{key}[/]: {preview}")
        if details:
            console.print(Panel(Text(details), title="Diff Preview", border_style="magenta"))
        return _ask_tool_confirmation("Allow this tool call?", default=False)


def _find_project_root() -> Path:
    """Find project root by looking for .git, pyproject.toml, package.json, etc."""
    markers = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    return cwd


def _generate_banner() -> list[str]:
    """Generate ASCII art banner for snowcode."""
    banner = [
        "",
        "   " + "\u2500" * 40,
        "   SNOWCODE",
        "   " + "\u2500" * 40,
        "",
        "   Local RAG Code Intelligence",
        "",
        "   " + "\u2500" * 40,
        "",
    ]
    return banner


class _ColoredGroup(click.Group):
    """Click group with colored help output."""

    def format_help(self, ctx, formatter):
        # Banner
        formatter.write("\n")
        banner = _generate_banner()
        colors = [
            "\x1b[0m",       # empty
            "\x1b[1;37m",    # title
            "\x1b[1;30m",    # separator
            "\x1b[0m",       # empty
            "\x1b[1;36m",    # tagline
            "\x1b[0m",       # empty
            # box lines
            "\x1b[1;37m",    # top-left corner
            "\x1b[1;36m",    # art row 1
            "\x1b[1;34m",    # art row 2
            "\x1b[1;35m",    # art row 3
            "\x1b[1;34m",    # art row 4
            "\x1b[1;36m",    # art row 5
            "\x1b[1;37m",    # empty row
            "\x1b[1;33m",    # tagline
            "\x1b[1;37m",    # empty row
            "\x1b[1;37m",    # bottom
            "\x1b[0m",       # empty
        ]
        for i, line in enumerate(banner):
            color = colors[i] if i < len(colors) else "\x1b[0m"
            formatter.write(f"{color}{line}\x1b[0m\n")
        formatter.write("\n")
        formatter.write(f"\x1b[1;32m  Version: {__version__}\x1b[0m\n")
        formatter.write("\n")
        formatter.write("\x1b[90m  Usage: snowcode [OPTIONS] COMMAND [ARGS]...\x1b[0m\n")
        formatter.write("\n")

        # Options
        formatter.write("\x1b[1;33m  Options:\x1b[0m\n")
        formatter.write("    \x1b[32m--version\x1b[0m    Show version\n")
        formatter.write("    \x1b[32m--help\x1b[0m       Show this help\n")
        formatter.write("\n")

        # Commands grouped by category
        commands = list(self.commands.items())

        categories = {
            "\x1b[1;35m  Core:\x1b[0m": [
                ("ingest",       "Build vector index (incremental)"),
                ("ask",          "Ask questions about the codebase"),
                ("chat",         "Interactive REPL with memory"),
                ("status",       "Show index status"),
                ("clean",        "Delete the index"),
                ("config",       "Set API Key, Model, URL"),
            ],
            "\x1b[1;34m  Agent:\x1b[0m": [
                ("agent",        "Multi-step: Plan -> Tools -> Memory -> Answer"),
                ("agent2",       "Enhanced agent: Better tools, memory, multi-agent"),
                ("agent-chat",   "Interactive agent session (keep chatting)"),
                ("agent-help",   "Detailed guide for agent and agent2 commands"),
            ],
            "\x1b[1;33m  Skills:\x1b[0m": [
                ("explain",      "Explain a function/class/file"),
                ("review",       "Code review (bugs, security, perf)"),
                ("find",         "Search code patterns (regex)"),
                ("summary",      "Architecture overview"),
                ("trace",        "Trace function call chain"),
                ("compare",      "Compare two files"),
                ("test-suggest", "Suggest test cases"),
                ("tree",         "Project structure tree"),
            ],
        }

        for cat_name, cmds in categories.items():
            formatter.write(f"{cat_name}\n")
            for cmd_name, desc in cmds:
                formatter.write(f"    \x1b[1m{cmd_name:<16}\x1b[0m{desc}\n")
            formatter.write("\n")

        formatter.write("\x1b[90m  More: snowcode COMMAND --help\x1b[0m\n\n")


@click.group(cls=_ColoredGroup)
@click.version_option(__version__, prog_name="snowcode")
def cli():
    """snowcode - Chat with your codebase using local RAG."""


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
def config(path: str | None):
    """Show and edit snowcode configuration (API Key, Model, URL, etc.)."""
    root = Path(path).resolve() if path else _find_project_root()
    llm_config = get_llm_config_from_file(root)
    
    # Get current API key from environment
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or llm_config["api_base_url"]
    model = os.environ.get("LLM_MODEL") or os.environ.get("CODECHAT_MODEL") or llm_config["default_model"]
    
    # Mask API key for display
    masked_key = ""
    if api_key:
        if len(api_key) > 8:
            masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        else:
            masked_key = "****"
    
    console.print("\n[bold cyan]╔══════════════════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║              Snowcode Configuration                         ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════╝[/]")
    
    # Current config display
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_row("API Key", masked_key or "[red]Not set[/]")
    table.add_row("API Base URL", base_url)
    table.add_row("Default Model", model)
    table.add_row("Thinking Mode", str(llm_config["thinking_enabled"]))
    console.print(table)
    
    # Environment variable status
    console.print("\n[dim]Environment Variables:[/]")
    env_table = Table(show_header=False, box=None, padding=(0, 2))
    env_table.add_column("Var", style="yellow", width=25)
    env_table.add_column("Status", width=10)
    env_table.add_column("Value", style="dim")
    
    for var_name in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "LLM_MODEL"]:
        var_val = os.environ.get(var_name, "")
        if var_val:
            if "KEY" in var_name:
                display = var_val[:4] + "****" if len(var_val) > 4 else "****"
            else:
                display = var_val
            env_table.add_row(var_name, "[green] set[/]", display)
        else:
            env_table.add_row(var_name, "[red] not set[/]", "")
    console.print(env_table)
    
    # Menu
    console.print("\n[bold yellow]Options:[/]")
    console.print("  [cyan]1[/] - Set API Key")
    console.print("  [cyan]2[/] - Set API Base URL")
    console.print("  [cyan]3[/] - Set Default Model")
    console.print("  [cyan]4[/] - Toggle Thinking Mode")
    console.print("  [cyan]5[/] - Quick Setup (API Key + URL + Model)")
    console.print("  [cyan]q[/] - Quit")
    
    while True:
        choice = input("\n\033[1;36mSelect option: \033[0m").strip().lower()
        
        if choice == "q" or choice == "":
            break
        
        elif choice == "1":
            # Set API Key
            new_key = click.prompt("Enter API Key", hide_input=True, confirmation_prompt=True)
            if new_key:
                # Save to .env file
                env_path = root / ".env"
                _update_env_file(env_path, "OPENAI_API_KEY", new_key)
                # Also set in current process
                os.environ["OPENAI_API_KEY"] = new_key
                console.print(f"[green] API Key saved to {env_path}[/]")
                masked = new_key[:4] + "*" * (len(new_key) - 8) + new_key[-4:] if len(new_key) > 8 else "****"
                console.print(f"  Key: {masked}")
        
        elif choice == "2":
            # Set Base URL
            new_url = click.prompt("Enter API Base URL", default=base_url)
            if new_url:
                env_path = root / ".env"
                _update_env_file(env_path, "OPENAI_BASE_URL", new_url)
                os.environ["OPENAI_BASE_URL"] = new_url
                console.print(f"[green] Base URL saved: {new_url}[/]")
        
        elif choice == "3":
            # Set Model
            console.print("[dim]Common models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, claude-3-5-sonnet[/]")
            new_model = click.prompt("Enter Model Name", default=model)
            if new_model:
                env_path = root / ".env"
                _update_env_file(env_path, "LLM_MODEL", new_model)
                os.environ["LLM_MODEL"] = new_model
                save_llm_config(root, {**llm_config, "default_model": new_model})
                console.print(f"[green] Default model saved: {new_model}[/]")
        
        elif choice == "4":
            # Toggle thinking mode
            new_thinking = not llm_config["thinking_enabled"]
            save_llm_config(root, {**llm_config, "thinking_enabled": new_thinking})
            console.print(f"[green] Thinking mode: {'enabled' if new_thinking else 'disabled'}[/]")
        
        elif choice == "5":
            # Quick setup
            console.print("\n[bold]Quick Setup[/]")
            console.print("[dim]Enter your API credentials (press Enter to skip):[/]\n")
            
            new_key = click.prompt("API Key", default="", show_default=False, hide_input=True)
            new_url = click.prompt("API Base URL", default="https://aiaiai.213891.xyz/v1", show_default=True)
            new_model = click.prompt("Model", default="gpt-4o", show_default=True)
            
            env_path = root / ".env"
            changes = []
            
            if new_key:
                _update_env_file(env_path, "OPENAI_API_KEY", new_key)
                os.environ["OPENAI_API_KEY"] = new_key
                changes.append("API Key")
            
            if new_url:
                _update_env_file(env_path, "OPENAI_BASE_URL", new_url)
                os.environ["OPENAI_BASE_URL"] = new_url
                changes.append("Base URL")
            
            if new_model:
                _update_env_file(env_path, "LLM_MODEL", new_model)
                os.environ["LLM_MODEL"] = new_model
                save_llm_config(root, {**llm_config, "default_model": new_model})
                changes.append("Model")
            
            if changes:
                console.print(f"\n[green] Saved: {', '.join(changes)}[/]")
                console.print(f"  Config file: {env_path}")
            else:
                console.print("[yellow]No changes made.[/]")
            break
        
        else:
            console.print("[red]Invalid option. Enter 1-5 or q.[/]")
    
    console.print("[dim]Done.[/]\n")


def _update_env_file(env_path: Path, key: str, value: str):
    """Update or add a key-value pair in .env file."""
    lines = []
    found = False
    
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"{key}={value}")
    
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--reset", is_flag=True, help="Full rebuild: clear and re-index everything")
@click.option("--chunk-size", default=None, type=int, help="Chunk size in characters")
@click.option("--chunk-overlap", default=None, type=int, help="Chunk overlap in lines")
@click.option("--model", "-m", default=None, help="Embedding model name")
def ingest(
    path: str | None,
    reset: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    model: str | None,
):
    """Ingest a project — full or incremental (only changed files)."""
    from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    root = Path(path).resolve() if path else _find_project_root()
    emb_model = model or DEFAULT_EMBEDDING_MODEL
    c_size = chunk_size or DEFAULT_CHUNK_SIZE
    c_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    console.print(f"\n[bold cyan]snowcode[/] v{__version__}")
    console.print(f"  Project: [green]{root}[/]")
    console.print(f"  Model:   [dim]{emb_model}[/]\n")

    # Initialize vector store
    with console.status("[bold green]Initializing embedding model...", spinner="dots"):
        store = VectorStore(root, embedding_model=emb_model)

    # Full rebuild
    if reset:
        store.reset()
        old_hashes: dict[str, str] = {}
        console.print("  [yellow]Index reset — full rebuild.[/]\n")
    else:
        old_hashes = store.load_hashes()

    # Scan files and compute hashes
    with console.status("[bold green]Scanning project files...", spinner="dots"):
        files = scan_files(root)

    if not files:
        console.print("[red]No code files found in the project.[/]")
        return

    # Compute current hashes and diff
    new_hashes: dict[str, str] = {}
    files_to_process: list[Path] = []
    unchanged_count = 0

    for f in files:
        rel = str(f.relative_to(root))
        h = store.file_hash(f)
        if not h:
            continue
        new_hashes[rel] = h
        if old_hashes.get(rel) == h:
            unchanged_count += 1
        else:
            files_to_process.append(f)

    # Find deleted files
    current_rels = set(new_hashes.keys())
    deleted_rels = set(old_hashes.keys()) - current_rels

    # Remove chunks for deleted and changed files
    paths_to_remove = list(deleted_rels) + [str(f.relative_to(root)) for f in files_to_process]
    removed_count = store.remove_by_files(paths_to_remove)

    # Report diff
    if old_hashes:
        console.print(f"  Files: [bold]{len(files)}[/] total, "
                       f"[green]{unchanged_count}[/] unchanged, "
                       f"[yellow]{len(files_to_process)}[/] changed/new, "
                       f"[red]{len(deleted_rels)}[/] deleted")
        if not files_to_process and not deleted_rels:
            console.print("  [green]Index is up to date.[/]")
            # Still save hashes in case format changed
            store.save_hashes(new_hashes)
            save_config(root, {
                "embedding_model": emb_model,
                "chunk_size": c_size,
                "chunk_overlap": c_overlap,
                "last_ingest": time.time(),
                "files_count": len(files),
                "chunks_count": store.count(),
            })
            return
        console.print()
    else:
        console.print(f"  Found [bold]{len(files)}[/] files (first ingest)\n")

    # Chunk only changed/new files
    new_chunks = []
    if files_to_process:
        import concurrent.futures
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(files_to_process))
            
            # Use ThreadPoolExecutor to parallelize file reading and chunking
            with concurrent.futures.ThreadPoolExecutor() as executor:
                def process_file(f):
                    rel = str(f.relative_to(root))
                    content = read_file(f)
                    if content:
                        return chunk_file(rel, content, c_size, c_overlap)
                    return []

                # Use map to preserve file order for better embedding caching/batching
                results = executor.map(process_file, files_to_process)
                for chunks in results:
                    if chunks:
                        new_chunks.extend(chunks)
                    progress.advance(task)

    if new_chunks:
        console.print(f"  Generated [bold]{len(new_chunks)}[/] chunks from {len(files_to_process)} files\n")

        # Embed and add
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding & indexing...", total=1)
            store.add_chunks(new_chunks)
            progress.advance(task)

    # Save hashes and config
    store.save_hashes(new_hashes)
    save_config(root, {
        "embedding_model": emb_model,
        "chunk_size": c_size,
        "chunk_overlap": c_overlap,
        "last_ingest": time.time(),
        "files_count": len(files),
        "chunks_count": store.count(),
    })

    # Summary
    table = Table(title="Ingestion Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files scanned", str(len(files)))
    table.add_row("Files processed", str(len(files_to_process)))
    table.add_row("Files unchanged (skipped)", str(unchanged_count))
    if deleted_rels:
        table.add_row("Files deleted", str(len(deleted_rels)))
    if removed_count > 0:
        table.add_row("Chunks removed", str(removed_count))
    table.add_row("New chunks", str(len(new_chunks)))
    table.add_row("Total chunks in store", str(store.count()))
    console.print(table)
    console.print("\n[bold green]Done![/] You can now run [cyan]snowcode ask \"...\"[/]\n")


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--context", "-k", default=5, type=int, help="Number of context chunks to retrieve")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files in output")
@click.option("--show-thinking", is_flag=True, help="Show LLM thinking/reasoning process")
def ask(
    question: tuple[str, ...],
    path: str | None,
    context: int,
    model: str | None,
    show_sources: bool,
    show_thinking: bool,
):
    """Ask a question about the codebase."""
    root = Path(path).resolve() if path else _find_project_root()
    q = " ".join(question)

    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]codechat ingest[/] first.[/]")
        return

    api_key, base_url, llm_model, thinking = _get_llm_config(model)

    if api_key:
        # Streaming mode with thinking support
        console.print(f"\n  [dim]LLM: {llm_model} @ {base_url}[/]")
        console.print()
        if thinking and show_thinking:
            console.print("[dim]=== Thinking ===[/]")

        think_buf: list[str] = []
        markdown_content = ""
        in_answer = False

        with Live(Markdown(""), console=console, refresh_per_second=10) as live:
            def on_think(token: str):
                nonlocal in_answer
                if not in_answer and token.strip():
                    think_buf.append(token)

            def on_answer(token: str):
                nonlocal in_answer, markdown_content
                if not in_answer:
                    in_answer = True
                        
                markdown_content += token
                live.update(Markdown(markdown_content))

            result = answer_question_stream(
                store, q, n_context=context, model=model,
                on_think=on_think, on_answer=on_answer,
            )
        console.print()
    else:
        # No LLM - use non-streaming fallback
        with console.status("[bold cyan]Searching...", spinner="dots"):
            result = answer_question(store, q, n_context=context, model=model)
        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold]Answer[/]",
            border_style="green",
            padding=(1, 2),
        ))

    # Show sources
    if show_sources or not api_key:
        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
    console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--context", "-k", default=5, type=int, help="Number of context chunks")
@click.option("--model", "-m", default=None, help="LLM model to use")
def chat(path: str | None, context: int, model: str | None):
    """Start an interactive chat session with the codebase."""
    root = Path(path).resolve() if path else _find_project_root()
    store = VectorStore(root)

    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]snowcode ingest[/] first.[/]")
        return

    codechat_dir = get_snowcode_dir(root)
    history_file = codechat_dir / "history.json"

    def _save_history():
        """Save conversation history to disk."""
        try:
            history_file.write_text(json.dumps(conversation_history, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_history():
        """Load conversation history from disk."""
        if history_file.exists():
            try:
                return json.loads(history_file.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    conversation_history: list[dict] = _load_history()

    console.print(f"\n[bold cyan]snowcode[/] v{__version__} - Interactive Mode")
    console.print(f"  Project: [green]{root}[/]")
    console.print(f"  Chunks:  [dim]{store.count()}[/]")
    if conversation_history:
        console.print(f"  History: [dim]Loaded {len(conversation_history)} past messages.[/]")
    console.print(f"  Type your questions, or [dim]/quit[/] to exit.\n")

    # Try to enable readline history
    try:
        import readline
        readline.parse_and_bind('tab: complete')
    except ImportError:
        pass

    while True:
        try:
            # Use simple input instead of prompt_toolkit
            q = input("\033[1;36msnowcode> \033[0m")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/]")
            break

        q = q.strip()
        if not q:
            continue

        if q in {"/quit", "/exit", "/q"}:
            console.print("[dim]Bye![/]")
            break

        if q in {"/cls", "/clear_screen"}:
            console.clear()
            continue

        if q == "/reset":
            store.reset()
            conversation_history.clear()
            console.print("[yellow]Index and conversation history reset.[/]")
            continue

        if q.startswith("/export"):
            parts = q.split(" ", 1)
            filename = parts[1].strip() if len(parts) > 1 else "snowcode_export.md"
            
            if not conversation_history:
                console.print("[yellow]No conversation history to export.[/]")
                continue
                
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# snowcode Q&A Export\n\n")
                    for msg in conversation_history:
                        role = "User" if msg["role"] == "user" else "snowcode"
                        f.write(f"## {role}\n\n{msg['content']}\n\n")
                console.print(f"[green]Successfully exported Q&A to [bold]{filename}[/][/]")
            except Exception as e:
                console.print(f"[red]Failed to export: {e}[/]")
            continue

        if q == "/clear":
            conversation_history.clear()
            _save_history()
            console.print("[yellow]Conversation history cleared.[/]")
            continue

        if q == "/load":
            conversation_history = _load_history()
            if conversation_history:
                console.print(f"[green]Loaded {len(conversation_history)} messages from history.[/]")
            else:
                console.print("[yellow]No history found.[/]")
            continue

        if q == "/stats":
            console.print(f"  Chunks: [bold]{store.count()}[/]")
            console.print(f"  History messages: [bold]{len(conversation_history)}[/]")
            continue

        if q == "/help":
            console.print("  [cyan]/quit[/]    Exit")
            console.print("  [cyan]/cls[/]     Clear the terminal screen")
            console.print("  [cyan]/clear[/]   Clear conversation history")
            console.print("  [cyan]/load[/]    Load history from disk")
            console.print("  [cyan]/export[/]  Export Q&A to Markdown (e.g. /export file.md)")
            console.print("  [cyan]/reset[/]   Clear the index and history")
            console.print("  [cyan]/stats[/]   Show index stats")
            console.print("  [cyan]/help[/]    Show this help")
            continue

        with console.status("[bold cyan]Thinking...", spinner="dots"):
            api_key, _, _, thinking = _get_llm_config(model)

        if api_key:
            # Streaming mode
            if thinking:
                console.print("[dim]=== Thinking ===[/]")
            is_answer = False
            
            markdown_content = ""
            think_buf: list[str] = []
            
            with Live(Markdown(""), console=console, refresh_per_second=10) as live:
                def _on_think(t: str):
                    nonlocal is_answer
                    if not is_answer and t.strip():
                        think_buf.append(t)
                        # Optional: Could update live with thinking tokens if desired
                        # but simple buffering might be safer for rendering

                def _on_answer(t: str):
                    nonlocal is_answer, markdown_content
                    if not is_answer:
                        is_answer = True
                        if thinking:
                            # Print thinking buf before answer starts
                            if think_buf:
                                console.print(f"[dim]{''.join(think_buf)}[/]")
                            console.print("[dim]=== Answer ===[/]")
                    
                    markdown_content += t
                    live.update(Markdown(markdown_content))

                result = answer_question_stream(
                    store, q, n_context=context, model=model,
                    on_think=_on_think, on_answer=_on_answer,
                    history=conversation_history
                )
            console.print()
        else:
            result = answer_question(store, q, n_context=context, model=model, history=conversation_history)
            console.print()
            console.print(Markdown(result["answer"]))

        if result["answer"] and not result["answer"].startswith("No LLM configured"):
            conversation_history.append({"role": "user", "content": q})
            conversation_history.append({"role": "assistant", "content": result["answer"]})
            # Keep history configurable to prevent file from growing infinitely
            limit = int(os.environ.get("CODECHAT_HISTORY_LIMIT", "10"))
            if len(conversation_history) > limit * 2:
                conversation_history = conversation_history[-limit * 2:]
            _save_history()

        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
        console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
def status(path: str | None):
    """Show the current index status."""
    root = Path(path).resolve() if path else _find_project_root()
    config = load_config(root)

    if not config:
        console.print("[dim]No index found. Run [cyan]snowcode ingest[/] to get started.[/]")
        return

    store = VectorStore(root)

    table = Table(title="snowcode Status", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project", str(root))
    table.add_row("Embedding model", config.get("embedding_model", "N/A"))
    table.add_row("Chunk size", str(config.get("chunk_size", "N/A")))
    table.add_row("Chunk overlap", str(config.get("chunk_overlap", "N/A")))
    table.add_row("Files indexed", str(config.get("files_count", "N/A")))
    table.add_row("Chunks in store", str(store.count()))
    if config.get("last_ingest"):
        import datetime
        ts = datetime.datetime.fromtimestamp(config["last_ingest"])
        table.add_row("Last ingested", ts.strftime("%Y-%m-%d %H:%M:%S"))

    console.print()
    console.print(table)
    console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.confirmation_option(prompt="Are you sure you want to delete the index?")
def clean(path: str | None):
    """Delete the vector index for the project."""
    root = Path(path).resolve() if path else _find_project_root()
    codechat_dir = get_snowcode_dir(root)

    import shutil
    if codechat_dir.exists():
        shutil.rmtree(codechat_dir)
        console.print("[green]Index deleted.[/]")
    else:
        console.print("[dim]No index found.[/]")


# ================================================================== Skills

def _run_skill_common(skill_name: str, target: str, path: str | None, model: str | None, show_sources: bool):
    """Shared logic for all skill commands."""
    root = Path(path).resolve() if path else _find_project_root()
    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]snowcode ingest[/] first.[/]")
        return

    api_key, _, _, thinking = _get_llm_config(model)

    if api_key:
        console.print()
        markdown_content = ""
        is_answer = False
        think_buf: list[str] = []

        with Live(Markdown(""), console=console, refresh_per_second=10) as live:
            def on_think(t: str):
                nonlocal is_answer
                if not is_answer and t.strip():
                    think_buf.append(t)

            def on_answer(t: str):
                nonlocal is_answer, markdown_content
                if not is_answer:
                    is_answer = True
                    if thinking:
                        if think_buf:
                            console.print(f"[dim]{''.join(think_buf)}[/]")
                        console.print("[dim]=== Answer ===[/]")
                
                markdown_content += t
                live.update(Markdown(markdown_content))

            result = run_skill_stream(store, skill_name, target, model=model, on_think=on_think, on_answer=on_answer)
        console.print()
    else:
        with console.status("[bold cyan]Analyzing...", spinner="dots"):
            result = run_skill(store, skill_name, target, model=model)
        console.print()
        console.print(Panel(Markdown(result["answer"]), border_style="green", padding=(1, 2)))

    if show_sources or not api_key:
        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
    console.print()


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def explain(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Explain a function, class, or file in detail."""
    _run_skill_common("explain", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=False)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def review(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Code review: find bugs, security issues, and quality problems."""
    t = " ".join(target) if target else "审查整个项目的核心代码"
    _run_skill_common("review", t, path, model, show_sources)


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def find(query: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Find code patterns: TODOs, FIXMEs, security risks, specific logic."""
    _run_skill_common("find", " ".join(query), path, model, show_sources)


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def summary(path: str | None, model: str | None, show_sources: bool):
    """Generate project architecture summary."""
    _run_skill_common("summary", "生成项目架构概览", path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def trace(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Trace function call chain."""
    _run_skill_common("trace", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("file_a", required=True)
@click.argument("file_b", required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def compare(file_a: str, file_b: str, path: str | None, model: str | None, show_sources: bool):
    """Compare two files or modules."""
    target = f"对比 {file_a} 和 {file_b} 的异同"
    _run_skill_common("compare", target, path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def test_suggest(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Suggest test cases for a function or module."""
    _run_skill_common("test", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--steps", "-s", default=0, type=int, help="Max agent steps (0=unlimited, default 0)")
@click.option("--no-plan", is_flag=True, help="Skip planning phase")
@click.option("--coordinator", is_flag=True, help="Use coordinator-worker architecture for complex tasks")
@click.option("--workers", "-w", default=2, type=int, help="Number of worker agents (default: 2)")
def agent(question: tuple[str, ...], path: str | None, model: str | None, steps: int, 
          no_plan: bool, coordinator: bool, workers: int):
    """Multi-step agent: Planning + Tools + Memory + Action + LLM."""
    root = Path(path).resolve() if path else _find_project_root()
    q = " ".join(question)

    console.print(f"\n  [dim]Project root: {root}[/]")

    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]snowcode ingest[/] first.[/]")
        return

    api_key, _, llm_model, _ = _get_llm_config(model)

    if not api_key:
        # No LLM, just search
        with console.status("[bold cyan]Searching...", spinner="dots"):
            results = store.query(q, n_results=5)
        if results:
            from .rag import _format_context
            console.print(Panel(Markdown("相关代码：\n\n" + _format_context(results)), border_style="green", padding=(1, 2)))
        else:
            console.print("[red]未找到相关代码。[/]")
        return

    # Choose agent architecture
    if coordinator:
        console.print(f"\n  [dim]LLM: {llm_model} | Mode: Coordinator-Worker | Workers: {workers}[/]\n")
        a = CoordinatorAgent(store, root, model=model, max_workers=workers)
    else:
        console.print(f"\n  [dim]LLM: {llm_model} | Steps: {steps} | Planning: {'on' if not no_plan else 'off'}[/]\n")
        a = CodeAgent(store, root, model=model, max_steps=steps, use_planning=not no_plan)

    def on_step(num, tool_name, preview):
        console.print(f"  [cyan]Step {num}[/] [yellow]-> {tool_name}[/]")
        console.print(f"  [dim]{preview}[/]\n")

    def on_think(t):
        console.print(f"  [dim]Think: {t}[/]")

    def on_answer(t):
        console.print()
        console.print(Panel(Markdown(t), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))
    
    def on_progress(t):
        console.print(f"  [cyan]Progress:[/] {t}")

    # Execute based on agent type
    if coordinator:
        # CoordinatorAgent uses plan_and_execute
        result = a.plan_and_execute(q, on_progress=on_progress)
        # Print the final answer
        console.print()
        console.print(Panel(Markdown(result.answer), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))
    else:
        # CodeAgent uses run
        result = a.run(q, on_step=on_step, on_think=on_think, on_answer=on_answer)

    # Show execution summary
    if result.actions:
        console.print(f"  [dim]Actions: {result.steps_taken} steps | Memory: {result.memory_entries} entries[/]")
    console.print()


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--steps", "-s", default=0, type=int, help="Max agent steps (default: 0=unlimited)")
@click.option("--no-plan", is_flag=True, help="Skip planning phase")
@click.option("--multi-agent", is_flag=True, help="Use multi-agent coordinator for complex tasks")
@click.option("--workers", "-w", default=2, type=int, help="Number of worker agents for multi-agent mode (default: 2)")
@click.option("--auto-approve", "-y", is_flag=True, help="Automatically approve tool confirmation prompts")
def agent2(question: tuple[str, ...], path: str | None, model: str | None, steps: int,
           no_plan: bool, multi_agent: bool, workers: int, auto_approve: bool):
    """Enhanced agent v2: Better tools, memory, and multi-agent support."""
    root = Path(path).resolve() if path else _find_project_root()
    q = " ".join(question)
    _set_tool_auto_approve(auto_approve)

    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]snowcode ingest[/] first.[/]")
        return

    api_key, _, llm_model, _ = _get_llm_config(model)

    if not api_key:
        # No LLM, just search
        with console.status("[bold cyan]Searching...", spinner="dots"):
            results = store.query(q, n_results=5)
        if results:
            from .rag import _format_context
            console.print(Panel(Markdown("相关代码：\n\n" + _format_context(results)), border_style="green", padding=(1, 2)))
        else:
            console.print("[red]未找到相关代码。[/]")
        return

    # Show tool registry
    registry = build_registry_v2()
    tools_info = ", ".join([t.name for t in registry.list_tools()])
    
    # Choose agent architecture
    if multi_agent:
        console.print(f"\n  [dim]LLM: {llm_model} | Mode: Multi-Agent Coordinator | Workers: {workers}[/]")
        console.print(f"  [dim]Tools: {tools_info}[/]\n")
        agent = create_coordinator_v2(
            store,
            root,
            model=model,
            num_workers=workers,
            confirm_tool=_confirm_tool_use,
        )
    else:
        console.print(f"\n  [dim]LLM: {llm_model} | Steps: {steps} | Planning: {'on' if not no_plan else 'off'}[/]")
        console.print(f"  [dim]Tools: {tools_info}[/]\n")
        agent = create_agent_v2(
            store,
            root,
            model=model,
            max_steps=steps,
            use_planning=not no_plan,
            confirm_tool=_confirm_tool_use,
        )

    # Progress callbacks
    step_count = [0]
    
    def on_step(num, tool_name, preview):
        step_count[0] = num
        console.print(f"  [cyan]Step {num}[/] [yellow]→ {tool_name}[/]")
        # Truncate preview
        preview_str = preview[:100] + "..." if len(preview) > 100 else preview
        console.print(f"  [dim]{preview_str}[/]\n")

    def on_think(t):
        console.print(f"  [dim]Think: {t}[/]")

    def on_answer(t):
        console.print()
        console.print(Panel(Markdown(t), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))
    
    def on_progress(t):
        console.print(f"  [cyan]{t}[/]")

    # Execute. Avoid Rich Live here because it interferes with interactive confirmations
    # on some Windows terminals.
    if multi_agent:
        answer = agent.coordinate(q, on_progress=on_progress)
        result = type('Result', (), {
            'answer': answer,
            'steps_taken': step_count[0],
            'memory_entries': 0,
            'tools_used': [],
            'actions': []
        })()
        console.print()
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))
    else:
        result = agent.run(
            q,
            on_step=on_step,
            on_think=on_think,
            on_answer=on_answer,
            on_progress=on_progress
        )

    # Show execution summary
    console.print()
    summary_table = Table(title="Execution Summary", show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Steps", str(result.steps_taken))
    if hasattr(result, 'tools_used') and result.tools_used:
        summary_table.add_row("Tools used", ", ".join(result.tools_used))
    if hasattr(result, 'total_elapsed_ms'):
        summary_table.add_row("Time", f"{result.total_elapsed_ms:.0f}ms")
    console.print(summary_table)
    console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--steps", "-s", default=0, type=int, help="Max agent steps per turn (default: 0=unlimited)")
@click.option("--multi-agent", is_flag=True, help="Use multi-agent coordinator")
@click.option("--workers", "-w", default=2, type=int, help="Number of worker agents")
@click.option("--auto-approve", "-y", is_flag=True, help="Automatically approve tool confirmation prompts")
def agent_chat(path: str | None, model: str | None, steps: int,
               multi_agent: bool, workers: int, auto_approve: bool):
    """Interactive agent session: keep chatting without re-entering commands."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    
    root = Path(path).resolve() if path else _find_project_root()
    _set_tool_auto_approve(auto_approve)
    
    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]snowcode ingest[/] first.[/]")
        return
    
    api_key, _, llm_model, _ = _get_llm_config(model)
    if not api_key:
        console.print("[red]No LLM configured. Run [cyan]snowcode config[/] or set DASHSCOPE_API_KEY.[/]")
        return
    
    if multi_agent:
        console.print(f"\n  [dim]LLM: {llm_model} | Mode: Multi-Agent Coordinator | Workers: {workers}[/]\n")
        agent = create_coordinator_v2(
            store,
            root,
            model=model,
            num_workers=workers,
            confirm_tool=_confirm_tool_use,
        )
    else:
        console.print(f"\n  [dim]LLM: {llm_model} | Steps: {steps}[/]\n")
        # Create agent once and reuse (preserves memory across turns)
        agent = create_agent_v2(
            store,
            root,
            model=model,
            max_steps=steps,
            use_planning=True,
            confirm_tool=_confirm_tool_use,
        )
    
    console.print(f"\n[bold cyan]snowcode[/] v{__version__} - Agent Chat")
    console.print("[dim]Type your question. /exit or Ctrl+D to quit.[/]\n")
    
    codechat_dir = get_snowcode_dir(root)
    history_file = codechat_dir / "agent_history.txt"
    
    session = PromptSession(
        history=FileHistory(str(history_file)) if codechat_dir.exists() else None
    )
    
    while True:
        try:
            q = session.prompt("snowcode-agent> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break
        
        q = q.strip()
        if not q:
            continue
        if q.lower() in ("/exit", "/quit", "/q"):
            console.print("[dim]Goodbye.[/]")
            break
        if q.lower() == "/clear":
            if hasattr(agent, "reset_memory"):
                agent.reset_memory()
                console.print("[dim]Memory cleared.[/]")
            else:
                console.print("[dim]Multi-agent mode does not keep shared chat memory.[/]")
            continue
        if q.lower() == "/help":
            console.print("[dim]Commands: /exit, /clear, /help[/]")
            continue
        
        def on_step(num, tool_name, preview):
            preview_str = preview[:100] + "..." if len(preview) > 100 else preview
            console.print(f"  [cyan]Step {num}[/] [yellow]-> {tool_name}[/]")
            console.print(f"  [dim]{preview_str}[/]\n")
        
        def on_think(t):
            console.print(f"  [dim]Think: {t}[/]")
        
        def on_answer(t):
            console.print()
            console.print(Panel(Markdown(t), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))

        def on_progress(t):
            console.print(f"  [cyan]{t}[/]")
        
        if multi_agent:
            answer = agent.coordinate(q, on_progress=on_progress)
            on_answer(answer)
        else:
            result = agent.run(
                q,
                on_step=on_step,
                on_think=on_think,
                on_answer=on_answer,
                on_progress=on_progress,
            )
            
            if result.actions:
                console.print(f"  [dim]Steps: {result.steps_taken} | Memory: {result.memory_entries}[/]\n")


@cli.command()
def agent_help():
    """Detailed guide for agent and agent2 commands."""
    from rich.panel import Panel
    from rich.table import Table

    console.print()
    console.print(Panel(
        f"[bold cyan]Snowcode Agent Guide v{__version__}[/]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()

    console.print("[bold yellow]agent vs agent2[/]\n")
    table = Table(show_header=True, border_style="dim")
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("agent", style="green", width=28)
    table.add_column("agent2 (recommended)", style="bright_green", width=42)
    table.add_row("Architecture", "Single-agent ReAct", "Planner + tools + verifier + optional multi-agent")
    table.add_row("Planning", "Basic", "Step planning with tool hints")
    table.add_row("Repo awareness", "Search-oriented", "Repo map + symbol graph grounding")
    table.add_row("Write safety", "Lightweight", "Runtime confirmation + diff preview")
    table.add_row("Verification", "Manual follow-up", "Auto verification after edits")
    table.add_row("Tooling", "Core coding tools", "20+ coding, skill, multimodal, and data tools")
    table.add_row("Data readers", "Limited", "JSON/YAML/TOML/MAT/NetCDF/HDF5/Parquet and more")
    table.add_row("Best for", "Quick exploration", "Real coding tasks and deeper repository workflows")
    console.print(table)
    console.print()

    console.print(Panel(
        """[bold]agent[/]
[dim]Lightweight ReAct agent for simple one-shot exploration.[/]

[bold cyan]Usage[/]
  snowcode agent "your question"

[bold cyan]Useful options[/]
  [green]--steps, -s[/]      Max steps per turn
  [green]--no-plan[/]        Skip planning
  [green]--coordinator[/]    Use coordinator-worker mode
  [green]--workers, -w[/]    Worker count for coordinator mode
  [green]--model, -m[/]      Override model
  [green]--path, -p[/]       Project root

[bold cyan]Examples[/]
  snowcode agent "Explain the auth flow"
  snowcode agent --coordinator --workers 3 "Review the project for security issues"
""",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print()

    console.print(Panel(
        """[bold]agent2[/]
[dim]Full-featured coding agent with planning, repo grounding, verification, and richer tools.[/]

[bold cyan]Usage[/]
  snowcode agent2 "your question"

[bold cyan]Useful options[/]
  [green]--steps, -s[/]          Max steps per turn (`0` = unlimited)
  [green]--no-plan[/]            Skip planning
  [green]--multi-agent[/]        Enable coordinator-worker mode
  [green]--workers, -w[/]        Worker count for multi-agent mode
  [green]--auto-approve, -y[/]   Auto-approve runtime tool confirmations
  [green]--model, -m[/]          Override model
  [green]--path, -p[/]           Project root

[bold cyan]Examples[/]
  snowcode agent2 "Explain the repository structure"
  snowcode agent2 -y "Refactor the config loader and run tests"
  snowcode agent2 --multi-agent --workers 3 "Audit caching and data flow"
""",
        border_style="green",
        padding=(1, 2)
    ))
    console.print()

    console.print(Panel(
        """[bold yellow]How agent2 works[/]

1. Build a plan from the user request.
2. Prefer repository-aware tools like [cyan]repo_map[/] for structure questions.
3. Execute tools and verify each step before moving on.
4. Ask for confirmation on writes and shell commands when needed.
5. Run inferred checks after edits before giving the final answer.
""",
        border_style="yellow",
        padding=(1, 2)
    ))
    console.print()

    console.print(Panel(
        """[bold magenta]Notable agent2 tools[/]

[bold]Core[/]
  [cyan]search[/], [cyan]read_file[/], [cyan]find_pattern[/], [cyan]list_dir[/], [cyan]repo_map[/]

[bold]Write / execute[/]
  [cyan]write_file[/], [cyan]search_replace[/], [cyan]shell[/]

[bold]Analysis[/]
  [cyan]explain[/], [cyan]review[/], [cyan]summary[/], [cyan]trace[/], [cyan]compare[/], [cyan]test_suggest[/]

[bold]Multimodal / data[/]
  [cyan]image_reader[/], [cyan]pdf_reader[/], [cyan]document_reader[/], [cyan]data_reader[/], [cyan]mat_reader[/], [cyan]file_browser[/], [cyan]nc_reader[/]""",
        border_style="magenta",
        padding=(1, 2)
    ))
    console.print()

    console.print(Panel(
        """[bold]Quick recommendations[/]

  Simple question            -> [green]snowcode agent2 "question"[/]
  Fast search                -> [green]snowcode agent2 --no-plan "question"[/]
  Deep repository analysis   -> [green]snowcode agent2 --multi-agent --workers 3 "question"[/]
  Safe automated edits       -> [green]snowcode agent2 "task"[/]
  Faster automated edits     -> [green]snowcode agent2 -y "task"[/]
  Interactive session        -> [green]snowcode agent-chat[/]
  Interactive auto-approve   -> [green]snowcode agent-chat -y[/]""",
        border_style="white",
        padding=(1, 2)
    ))
    console.print()

@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--symbols", "-s", is_flag=True, help="Show classes and functions in the tree")
@click.option("--deps", "-d", is_flag=True, help="Show dependency graph (imports) for the project")
@click.option("--internal", "-i", is_flag=True, help="Only show internal dependencies")
@click.option("--mermaid", "-m", is_flag=True, help="Output dependency graph in Mermaid format")
def tree(path: str | None, symbols: bool, deps: bool, internal: bool, mermaid: bool):
    """Generate a visual project structure and dependency tree."""
    from .tree_gen import build_project_tree, build_dependency_graph, generate_mermaid_graph
    
    root = Path(path).resolve() if path else _find_project_root()
    
    if mermaid:
        with console.status("[bold green]Generating Mermaid graph...", spinner="dots"):
            mermaid_output = generate_mermaid_graph(root, internal_only=internal)
        console.print()
        console.print(Syntax(mermaid_output, "markdown", theme="monokai", word_wrap=True))
        console.print("\n[dim]Tip: You can copy this output and paste it into any Markdown file to render the graph.[/]")
        return

    with console.status("[bold green]Analyzing project structure...", spinner="dots"):
        project_tree = build_project_tree(root, show_symbols=symbols)
        if deps:
            dep_tree = build_dependency_graph(root, internal_only=internal)
        else:
            dep_tree = None
        
    console.print()
    console.print(project_tree)
    if dep_tree:
        console.print()
        console.print(dep_tree)
    console.print()


def main():
    cli()


if __name__ == "__main__":
    main()
