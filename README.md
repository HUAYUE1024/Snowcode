<div align="center">

# codechat

**Local RAG-powered codebase Q&A — chat with your code in the terminal**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/Storage-NumPy-orange.svg)](https://numpy.org)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#privacy--security)

</div>

---

When you take over a complex open-source project or revisit your own codebase from months ago, understanding the architecture is painfully slow. **codechat** vectorizes your entire project locally so you can "talk" to your codebase directly from the terminal.

---

## Quick Start

```bash
# Install
git clone https://github.com/HUAYUE1024/codechat.git
cd codechat
pip install -e .

# Use
cd /path/to/your-project
codechat ingest                              # build vector index
codechat ask "how does auth work?"           # ask questions
codechat agent "trace the request lifecycle" # multi-step exploration
codechat chat                                # interactive REPL
```

## Commands

| Command | Description |
|---------|-------------|
| `ingest` | Scan project files, build local vector index (incremental by default) |
| `ask "question"` | Ask about the codebase (streaming output) |
| `agent "question"` | Multi-step agent: Plan → Tools → Memory → Answer |
| `chat` | Interactive REPL with history and auto-complete |
| `explain "target"` | Explain a function, class, or file |
| `review` | Code review: bugs, security, performance |
| `find "pattern"` | Search code patterns (regex, definitions, imports) |
| `summary` | Generate project architecture overview |
| `trace "target"` | Trace function call chains |
| `compare A B` | Compare two files or modules |
| `test-suggest "target"` | Suggest test cases |
| `status` | Show index status |
| `clean` | Delete the vector index |

## Chat REPL Commands

Inside the interactive `codechat chat` mode, you can use the following commands:

| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, `/q` | Exit the chat session |
| `/cls` | Clear the terminal screen |
| `/clear` | Clear the current conversation history (memory) |
| `/load` | Reload the conversation history from disk |
| `/export [file]` | Export the Q&A session to a Markdown file (default: `codechat_export.md`) |
| `/reset` | Clear the vector index and conversation history |
| `/stats` | Show index and history statistics |
| `/help` | Show available commands |

## Setup LLM

```cmd
:: Windows
set DASHSCOPE_API_KEY=sk-xxx

# Linux / Mac
export DASHSCOPE_API_KEY=sk-xxx
```

Default model: `qwen-flash`. See [LLM Config](#llm-config) for more options.

**Smart chunking:** AST-first (Tree-sitter parses real function/class boundaries) → regex fallback → line-based fallback. Supports 20+ languages.

## Incremental Indexing

By default, `ingest` only processes changed files:

```bash
codechat ingest          # incremental: new/changed/deleted files only
codechat ingest --reset  # full rebuild
```

**How it works:**
1. Each file's hash (mtime + size) is stored in `.codechat/file_hashes.json`
2. On subsequent runs, hashes are compared to detect changes
3. Only changed/new files are re-chunked and re-embedded
4. Chunks from deleted files are automatically removed
5. Unchanged files are completely skipped

```
Files: 42 total, 38 unchanged, 3 changed/new, 1 deleted
```

## Agent Mode

The agent decomposes complex questions into steps and uses tools to explore the codebase:

```
codechat agent "how does the vector store persist data?"
```

```
Step 1 → search
Think: Need to find the storage implementation

Step 2 → read_file
Think: Found store.py, need to see _save and _load methods

Answer:
The vector store persists data as NumPy .npy + JSON files...
```

**5 built-in tools:**

| Tool | Description |
|------|-------------|
| `search` | Semantic code search |
| `read_file` | Read file content (with line range) |
| `find_pattern` | Regex search across codebase |
| `list_dir` | Browse directory structure |
| `read_multiple` | Read multiple files simultaneously |

**Memory system:**
- **Short-term**: Sliding window of tool calls within a session (default 20 entries)
- **Long-term**: Q&A sessions persisted to `.codechat/memory.jsonl`

```bash
codechat agent "question" -s 3       # limit to 3 steps
codechat agent "question" --no-plan  # skip planning phase
```

## Skills

7 specialized prompts optimized for specific analysis tasks:

| Command | Purpose | Example |
|---------|---------|---------|
| `explain` | Explain function/class/file | `codechat explain "VectorStore.query"` |
| `review` | Code review | `codechat review` or `codechat review store.py` |
| `find` | Search patterns | `codechat find "all exception handling"` |
| `summary` | Architecture overview | `codechat summary` |
| `trace` | Call chain tracing | `codechat trace "answer_question"` |
| `compare` | Compare files | `codechat compare store.py chunker.py` |
| `test-suggest` | Test case suggestions | `codechat test-suggest "chunk_file"` |

Each skill has its own system prompt, retrieval strategy, and context size.

## LLM Config

### DashScope (Recommended for China)

```cmd
set DASHSCOPE_API_KEY=sk-xxx
```

Default model: `qwen-flash`

### OpenAI Compatible

```cmd
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.openai.com/v1
```

### Domestic LLM Providers

```cmd
:: DeepSeek
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set CODECHAT_MODEL=deepseek-chat

:: Qwen via OpenAI compat
set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Ollama (Local)

```cmd
ollama pull qwen2.5-coder:7b
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen2.5-coder
```

### Thinking Mode

DashScope supports reasoning tokens. Off by default:

```cmd
set CODECHAT_THINKING=1
codechat ask "complex question" --show-thinking
```

### Embedding Models

```bash
codechat ingest -m all-mpnet-base-v2           # default, good quality
codechat ingest -m all-MiniLM-L6-v2            # faster, lower quality
codechat ingest -m paraphrase-multilingual-MiniLM-L12-v2  # multilingual
```

| Model | Dimensions | Size | Note |
|-------|-----------|------|------|
| `all-mpnet-base-v2` | 768 | 420MB | Default, best quality |
| `all-MiniLM-L6-v2` | 384 | 90MB | Fast, lower quality |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | Multilingual |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   CLI (Click + Rich)             │
├──────────┬────────────────┬──────────────────────┤
│   ask    │    agent       │   skills             │
│          │                │ explain/review/find  │
│          │ ┌────────────┐ │ summary/trace/       │
│          │ │ Planning   │ │ compare/test-suggest │
│          │ │ Memory     │ │                      │
│          │ │ Action     │ │                      │
│          │ │ Tools      │ │                      │
│          │ └────────────┘ │                      │
├──────────┴────────────────┴──────────────────────┤
│                   RAG Engine                     │
├──────────────────────────────────────────────────┤
│ Scanner → Chunker → VectorStore → LLM Client    │
│ (os.walk   (func-level  (NumPy .npy  (DashScope │
│  pruning)   split +      + JSON)     OpenAI     │
│            merge small)             Ollama)     │
└──────────────────────────────────────────────────┘
```

## Project Structure

```
codechat/
├── pyproject.toml       # Project config & dependencies
├── README.md
├── LICENSE              # MIT
├── .gitignore
└── codechat/
    ├── __init__.py      # Version
    ├── __main__.py      # python -m codechat
    ├── cli.py           # CLI commands (Click + Rich)
    ├── config.py        # Constants, file types, skip dirs
    ├── scanner.py       # File scanner (os.walk + pruning)
    ├── chunker.py       # Smart code chunking
    ├── store.py         # NumPy + JSON vector storage
    ├── rag.py           # RAG engine
    ├── agent.py         # Agent: Planning/Tools/Memory/Action/LLM
    ├── skills.py        # 7 specialized skill prompts
```

**Generated data:**
```
your-project/
├── .codechat/
│   ├── config.json      # Index config
│   ├── embeddings.npy   # Vector matrix
│   ├── metadata.json    # File paths + line numbers
│   └── memory.jsonl     # Agent long-term memory
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `click` | CLI framework |
| `numpy` | Vector storage |
| `sentence-transformers` | Embedding models |
| `prompt-toolkit` | Interactive REPL |
| `rich` | Terminal output |
| `pathspec` | .gitignore parsing |
| `openai` | OpenAI-compatible API |
| `httpx` | Ollama API |

## Supported File Types

**Code:** `.py` `.js` `.ts` `.tsx` `.go` `.rs` `.java` `.kt` `.c` `.cpp` `.cs` `.rb` `.php` `.swift` `.sh` `.sql` `.proto` — 40+ languages

**Docs:** `.md` `.rst` `.txt`

**Config:** `.json` `.yaml` `.toml` `.xml` `.env`

**Auto-skipped:** `.git` `node_modules` `__pycache__` `.venv` `dist` `build` `.codechat`

**Custom Ignore:** You can create a `.codechatignore` file in your project root to define custom ignore rules using standard `.gitignore` syntax.

## FAQ

**Q: Does it work without an LLM?**
Yes. Falls back to raw code retrieval — useful for locating where logic lives.

**Q: Chinese support?**
Full support. Embedding and LLM both handle Chinese/English mixed input.

**Q: Index size?**
Typically 20-50MB for 1000 files.

**Q: Re-index after code changes?**
Yes. `codechat ingest --reset` to rebuild.

## Privacy & Security

- All vector data stored locally in `.codechat/`
- Embedding runs locally via sentence-transformers
- Only LLM calls go to external API (DashScope/OpenAI/Ollama)
- File paths validated against project root (no path traversal)
- Regex patterns limited to prevent ReDoS

## Roadmap

- [x] RAG Q&A with semantic search
- [x] Agent with Planning + Tools + Memory
- [x] 7 specialized skills
- [x] Streaming output + thinking mode
- [x] Long-term memory persistence
- [x] Incremental indexing (only changed files)
- [x] AST-aware chunking (Tree-sitter, 20+ languages)
- [x] Multi-turn conversation memory
- [x] `.codechatignore` custom rules
- [x] Export Q&A to Markdown
- [x] Test suite
- [x] Hybrid Search (Vector + BM25)
- [x] Streaming Markdown rendering (Rich Live)

## License

[MIT](LICENSE)
