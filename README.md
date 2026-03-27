<div align="center">

# codechat

**A Local RAG-Powered Code Intelligence Engine for Terminal Environments**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-0DA338.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-ReAct_Agent-blueviolet.svg)](#agent-architecture)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#privacy--security)

*A Retrieval-Augmented Generation system designed for codebase comprehension, featuring AST-aware semantic chunking, hybrid vector-keyword retrieval, and an autonomous ReAct agent with full CRUD tool capabilities.*

</div>

---

## Abstract

**codechat** is a privacy-first, fully-local code intelligence engine that enables developers to query, analyze, and modify codebases through natural language in a terminal environment. Unlike cloud-based alternatives (GitHub Copilot, Cursor), codechat performs all embedding, indexing, and retrieval operations locally, with only optional LLM calls leaving the machine. The system employs a hybrid retrieval architecture combining dense vector similarity (sentence-transformers) with sparse keyword matching (BM25), reranked by a cross-encoder, and orchestrated through a ReAct agent with planning, memory, and 8 specialized tools supporting full CRUD operations.

## Key Contributions

| Contribution | Description |
|:-------------|:------------|
| **Hybrid Retrieval** | Dense (vector) + Sparse (BM25) + Cross-encoder reranking |
| **AST-Aware Chunking** | Tree-sitter parsing for 20+ languages, preserving semantic boundaries |
| **ReAct Agent** | Planning → Tools → Memory → Observation loop with repeat detection |
| **8 Tool Suite** | Full CRUD: search, read, write, search-replace, delete, list, find-pattern, read-multiple |
| **Incremental Indexing** | File hash tracking; only changed files re-processed |
| **Multi-LLM Backend** | DashScope / OpenAI-compatible / Ollama with streaming + thinking mode |
| **Privacy Guarantee** | Zero data exfiltration; all computation local except optional LLM API |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Terminal Interface                           │
│                    (Click CLI + Rich + Prompt Toolkit)               │
├─────────────┬────────────────────────┬───────────────────────────────┤
│   Direct    │    Agent Mode          │    Skill Mode                 │
│   Query     │                        │                               │
│             │ ┌──────────────────┐   │ explain / review / find       │
│   ask       │ │    Planner       │   │ summary / trace / compare     │
│   chat      │ │  (LLM-decomposed │   │ test-suggest / tree           │
│             │ │   sub-tasks)     │   │                               │
│             │ ├──────────────────┤   │ (7 specialized prompts with   │
│             │ │    Executor      │   │  optimized retrieval params)  │
│             │ │  (8 tools, retry │   │                               │
│             │ │   + dedup)       │   │                               │
│             │ ├──────────────────┤   │                               │
│             │ │    Memory        │   │                               │
│             │ │ Short: sliding   │   │                               │
│             │ │ Long: .jsonl     │   │                               │
│             │ └──────────────────┘   │                               │
├─────────────┴────────────────────────┴───────────────────────────────┤
│                      RAG Retrieval Engine                            │
│  ┌──────────────┐   ┌──────────────────────────────────────────┐    │
│  │  Query       │──▶│  Hybrid Search                           │    │
│  │  Embedding   │   │  ┌──────────┐  ┌──────┐  ┌───────────┐  │    │
│  │  (mpnet-768) │   │  │  Vector  │  │ BM25 │  │ Cross-    │  │    │
│  │              │   │  │  Search  │+ │      │─▶│ Encoder   │  │    │
│  │              │   │  │  (cosine)│  │(tfidf│  │ Reranker  │  │    │
│  │              │   │  └──────────┘  └──────┘  └───────────┘  │    │
│  └──────────────┘   └──────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                       Indexing Pipeline                              │
│                                                                      │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌───────────────┐  │
│  │ Scanner  │──▶│  Chunker   │──▶│ Embedder │──▶│  VectorStore  │  │
│  │          │   │            │   │          │   │               │  │
│  │ os.walk  │   │ AST-first  │   │ mpnet    │   │ .npy + JSON   │  │
│  │ pruning  │   │ → regex    │   │ 768-dim  │   │ + BM25 index  │  │
│  │ + gitign │   │ → lines    │   │ local    │   │ + file hashes │  │
│  │ + codech │   │ (20+ lang) │   │          │   │               │  │
│  └──────────┘   └────────────┘   └──────────┘   └───────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Methodology

### 1. Code-Aware Indexing Pipeline

**File Discovery** (`scanner.py`): Recursively traverses the project using `os.walk` with in-place directory pruning. Respects both `.gitignore` and `.codechatignore` patterns via the `pathspec` library. Skips 14 categories of non-source directories (`.git`, `node_modules`, `__pycache__`, `.venv`, etc.) and enforces a 1MB file size limit.

**Semantic Chunking** (`chunker.py`, `ast_chunker.py`): Employs a three-tier fallback strategy:

1. **AST Parsing** (Tree-sitter): Parses source into syntax trees for 20+ languages. Extracts top-level definitions (functions, classes, methods, interfaces, traits) as atomic chunks. Merges fragments shorter than 10 lines with adjacent chunks.

2. **Regex Heuristics**: Pattern-based detection of function/class boundaries using language-specific regular expressions (Python `def`/`class`, Go `func`, Rust `fn`/`impl`, etc.).

3. **Line-Based Sliding Window**: Overlapping window (default 1500 chars, 5-line overlap) as final fallback.

**Vector Storage** (`store.py`): Embeds chunks using `all-mpnet-base-v2` (768-dimensional) via sentence-transformers. Stores embeddings as NumPy `.npy` matrices and metadata as JSON, avoiding external database dependencies. Supports incremental indexing via content-based file hashing (mtime + size).

### 2. Hybrid Retrieval

The retrieval engine combines two complementary search strategies:

**Dense Retrieval**: Query embedding (mpnet-768) compared against stored chunk embeddings via cosine similarity. File-type weighting applied post-retrieval: code files ×1.0, documents ×0.4, config files ×0.7. Diversification ensures no more than 2 chunks per file in results.

**Sparse Retrieval (BM25)**: Token-based keyword matching using BM25 scoring with IDF weighting. Effective for exact term matching (function names, variable names, error messages).

**Cross-Encoder Reranking**: Top candidates from both methods reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` for final relevance ordering.

### 3. Agent Architecture

The agent follows the **ReAct** (Reasoning + Acting) paradigm:

```
┌─────────────────────────────────────────────────┐
│                   ReAct Loop                    │
│                                                 │
│  ┌─────────┐    ┌─────────┐    ┌────────────┐  │
│  │  Think   │───▶│  Act    │───▶│  Observe   │  │
│  │ (LLM)   │    │ (Tool)  │    │ (Result)   │  │
│  └─────────┘    └─────────┘    └─────┬──────┘  │
│       ▲                               │         │
│       └───────────────────────────────┘         │
│                                                 │
│  Termination: answer found | repeat detected    │
│               max steps    | no results ×3      │
└─────────────────────────────────────────────────┘
```

**Planning**: LLM decomposes user goals into 2-5 executable steps with tool hints.

**Memory**: 
- Short-term: Sliding window (20 entries, 30K chars) of tool calls and observations within a session.
- Long-term: Q&A sessions persisted to `.codechat/memory.jsonl` for cross-session recall.

**Tool Suite** (8 tools, full CRUD):

| Tool | Class | Operation | Safety |
|:-----|:------|:----------|:-------|
| `search` | SearchTool | Semantic search | Read-only |
| `read_file` | ReadFileTool | Read full file (≤2000 lines) | Path validated |
| `find_pattern` | FindPatternTool | Regex search (≤200 char pattern, ≤500 char lines) | ReDoS protected |
| `list_dir` | ListDirTool | Browse directory | Skip dirs applied |
| `read_multiple` | ReadMultipleTool | Batch file reads | Path validated |
| `write_file` | WriteFileTool | Create/overwrite file | `.bak` backup |
| `search_replace` | SearchReplaceTool | Find-and-replace code blocks | `.bak` backup |
| `delete_file` | DeleteFileTool | Delete file | `.deleted` backup |

**Safety Mechanisms**:
- Path traversal prevention: `resolve()` + `is_relative_to(root)` on all file operations
- ReDoS protection: Pattern length ≤200 chars, search line ≤500 chars
- Repeat detection: Auto-exits if identical tool+params called 2× consecutively
- Hard step cap: 50 steps maximum (configurable via `--steps`)
- All destructive operations create backups before execution

### 4. Multi-LLM Backend

| Backend | Environment Variables | Features |
|:--------|:--------------------|:---------|
| **DashScope** | `DASHSCOPE_API_KEY` | Streaming, thinking/reasoning tokens |
| **OpenAI Compat** | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Any OpenAI-compatible API |
| **Ollama** | `OLLAMA_URL`, `OLLAMA_MODEL` | Fully local, zero network |

Default: `qwen-flash` via DashScope. Thinking mode (reasoning tokens) off by default, enabled via `CODECHAT_THINKING=1`.

---

## Commands Reference

### Core

| Command | Description | Example |
|:--------|:------------|:--------|
| `ingest` | Build vector index (incremental) | `codechat ingest --reset` |
| `ask` | Direct Q&A with streaming | `codechat ask "how does auth work?"` |
| `chat` | Interactive REPL with persistent memory | `codechat chat` |
| `status` | Index statistics | `codechat status` |
| `clean` | Delete index | `codechat clean` |

### Agent

| Command | Description | Example |
|:--------|:------------|:--------|
| `agent` | Multi-step autonomous exploration | `codechat agent "trace request lifecycle"` |

Options: `-s N` max steps, `--no-plan` skip planning, `-m MODEL` LLM

### Skills (Specialized Prompts)

| Command | Purpose | Optimal Use Case |
|:--------|:--------|:-----------------|
| `explain` | Function/class/file explanation | Onboarding to unfamiliar code |
| `review` | Bug/security/performance audit | Pre-commit quality gate |
| `find` | Pattern search (regex, definitions) | Locating specific logic |
| `summary` | Architecture overview | Project documentation |
| `trace` | Call chain tracing | Debugging, impact analysis |
| `compare` | File diff with analysis | Refactoring, merge review |
| `test-suggest` | Test case generation | Test planning |
| `tree` | Visual project structure | Quick orientation |

---

## Installation

```bash
git clone https://github.com/HUAYUE1024/codechat.git
cd codechat
pip install -e .
```

### Dependencies

| Package | Version | Role |
|:--------|:--------|:-----|
| `click` | ≥8.1 | CLI framework |
| `numpy` | ≥1.24 | Vector storage |
| `sentence-transformers` | ≥3.0 | Embedding + reranking |
| `tree-sitter` | ≥0.22 | AST parsing |
| `tree-sitter-languages` | ≥1.10 | Pre-built grammars |
| `prompt-toolkit` | ≥3.0 | Interactive REPL |
| `rich` | ≥13.0 | Terminal rendering |
| `pathspec` | ≥0.12 | .gitignore parsing |
| `openai` | ≥1.0 | LLM API client |
| `httpx` | ≥0.27 | Ollama HTTP client |

---

## Data Format

All project data stored in `.codechat/`:

```
.codechat/
├── config.json              # Index configuration
├── embeddings.npy           # Vector matrix (N × 768 float32)
├── metadata.json            # Chunk metadata (file, lines, index)
├── file_hashes.json         # File hashes for incremental indexing
├── bm25.json                # BM25 inverted index
├── chat_history.json        # Persistent chat memory
└── memory.jsonl             # Agent long-term memory
```

---

## Supported Languages

**AST-Aware** (Tree-sitter): Python, JavaScript, TypeScript, TSX, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Lua, Bash, SQL, R, HTML, CSS

**Regex Fallback**: 40+ additional languages via pattern matching

**Auto-Skipped**: `.git`, `__pycache__`, `node_modules`, `.venv`, `dist`, `build`, `.codechat`

---

## Privacy & Security

- **Zero Cloud Dependency**: Embedding, indexing, BM25, and AST parsing all execute locally
- **LLM Optional**: System operates in retrieval-only mode without any LLM configured
- **Path Containment**: All file operations validated against project root via `resolve()` + `is_relative_to()`
- **Input Sanitization**: Regex patterns length-limited, search lines truncated, ReDoS protected
- **Backup Safety**: All write/delete operations create `.bak`/`.deleted` backups
- **No Telemetry**: No analytics, no usage tracking, no external calls beyond optional LLM API

---

## Project Statistics

| Metric | Value |
|:-------|:------|
| Total Python LOC | ~4,400 |
| Modules | 11 |
| CLI Commands | 16 |
| Agent Tools | 8 |
| Skills | 7 |
| Supported Languages | 20+ (AST) / 40+ (regex) |

---

## License

[MIT](LICENSE)
