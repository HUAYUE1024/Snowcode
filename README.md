<div align="center">

# Snowcode

**A Local RAG-Powered Code Intelligence Engine with Multimodal Agent**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Version](https://img.shields.io/badge/Version-0.4.0-blue.svg)](https://github.com/HUAYUE1024/Snowcode)
[![License: MIT](https://img.shields.io/badge/License-MIT-0DA338.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#privacy--security)

*Query, analyze, verify, and modify codebases through natural language. Snowcode combines local hybrid retrieval with autonomous agents, repository-aware planning, write safeguards, and multimodal/data readers.*

</div>

---

## Features

| Feature | Description |
|:--------|:------------|
| **Hybrid Retrieval** | Dense (vector) + Sparse (BM25) + Cross-encoder reranking |
| **Agent v2** | Enhanced ReAct agent with planning, repo map grounding, memory, and 20+ tools |
| **Multi-Agent** | Coordinator-Worker architecture for complex task decomposition |
| **Write Safety** | Runtime confirmation plus diff preview for file writes and replacements |
| **Verification Loop** | `planner -> executor -> verifier -> test` flow after edits |
| **Multimodal** | Image analysis, PDF parsing, Office docs, and structured data readers |
| **Scientific Data** | NetCDF (.nc) and MATLAB `.mat` file support for scientific workflows |
| **Interactive Chat** | Persistent session with command history (`agent-chat`) |
| **File Creation** | Agent can write files, generate reports, and modify code |
| **Multi-LLM** | DashScope / OpenAI-compatible / Ollama backends |
| **Privacy** | All indexing local; only optional LLM calls leave the machine |

---

## Quick Start

```bash
# Install
pip install -e ".[multimodal,data,scientific]"

# Configure (interactive)
snowcode config

# Index your project
snowcode ingest

# Ask questions
snowcode ask "How does authentication work?"

# Agent mode (autonomous)
snowcode agent2 "Analyze the project architecture"

# Interactive agent session
snowcode agent-chat
```

### Agent2 New In This Update

- `repo_map` for repository structure, symbol graph, and dependency-aware grounding
- diff preview before `write_file` and `search_replace`
- runtime confirmation flow that behaves better on Windows terminals
- `--auto-approve` / `-y` for non-interactive or fast approval flows
- automatic verification after edits, including inferred test commands
- planner hint normalization so architecture questions prefer `repo_map`
- richer readers via `data_reader`, `mat_reader`, and improved scientific fallbacks

---

## Commands

### Core

| Command | Description |
|:--------|:------------|
| `ingest` | Build vector index (incremental) |
| `ask` | Direct Q&A with streaming |
| `chat` | Interactive REPL with memory |
| `status` | Index statistics |
| `clean` | Delete index |
| `config` | Configure LLM settings |

### Agent

| Command | Description |
|:--------|:------------|
| `agent` | Single-turn autonomous exploration |
| `agent2` | Enhanced agent with better tools & memory |
| `agent-chat` | Interactive persistent agent session |
| `agent-help` | Detailed usage guide |

**Agent2 Options:**
- `--multi-agent` — Enable coordinator-worker mode
- `--workers N` — Number of parallel workers (default: 2)
- `--steps N` — Max steps per turn (0 = unlimited)
- `--no-plan` — Skip planning phase
- `--model MODEL` — Specify LLM model
- `--auto-approve`, `-y` — Skip runtime confirmation prompts

### Skills

| Command | Purpose |
|:--------|:--------|
| `explain` | Explain function/class/file |
| `review` | Code review (bugs, security, perf) |
| `find` | Search code patterns (regex) |
| `summary` | Architecture overview |
| `trace` | Function call chain tracing |
| `compare` | Compare two files |
| `test-suggest` | Suggest test cases |
| `tree` | Visual project structure & dependency graph |

---

## Agent Architecture

Current `agent2` runtime loop:

1. Planner decomposes the task into concrete steps.
2. Executor chooses tools and performs reads, writes, or shell actions.
3. Verifier checks whether the current step is actually complete.
4. Auto verification runs suggested tests after edits.
5. Final answer synthesis is grounded in verified execution history.

### Agent v2 (Enhanced)

```
┌─────────────────────────────────────────────────┐
│                  Agent v2 Loop                   │
│                                                  │
│  ┌─────────┐    ┌─────────┐    ┌────────────┐   │
│  │  Plan    │───▶│  Act    │───▶│  Observe   │   │
│  │ (LLM)   │    │ (Tool)  │    │ (Result)   │   │
│  └─────────┘    └─────────┘    └─────┬──────┘   │
│       ▲                               │          │
│       └───────────────────────────────┘          │
│                                                  │
│  Memory: Short-term (sliding) + Long-term (.jsonl)│
└──────────────────────────────────────────────────┘
```

### Multi-Agent Coordinator

```
┌──────────────┐     ┌──────────┐     ┌──────────┐
│ Coordinator  │────▶│ Worker 1 │     │ Worker 2 │
│  (Plan &     │     └────┬─────┘     └────┬─────┘
│   Delegate)  │          │                │
└──────────────┘          ▼                ▼
                   ┌─────────────────────────┐
                   │   Synthesize Results    │
                   └─────────────────────────┘
```

### Tool Suite (20+ tools)

**Core code tools**

| Tool | Type | Description |
|:-----|:-----|:------------|
| `search` | Read | Semantic code search |
| `read_file` | Read | Read file content with line windows |
| `find_pattern` | Read | Regex pattern search |
| `list_dir` | Read | Directory structure |
| `repo_map` | Read | Repository structure, symbols, and dependencies |
| `write_file` | Write | Create or overwrite a file with confirmation |
| `search_replace` | Write | Find-and-replace with diff preview |
| `shell` | Execute | Run terminal commands with confirmation |

**Analysis tools**

| Tool | Type | Description |
|:-----|:-----|:------------|
| `explain` | Skill | Explain a function, class, or file |
| `review` | Skill | Code review for bugs, security, and performance |
| `summary` | Skill | Architecture overview with repo-map fallback |
| `trace` | Skill | Function or method call-chain tracing |
| `compare` | Skill | Compare two files or modules |
| `test_suggest` | Skill | Suggest test cases |

**Multimodal and data tools**

| Tool | Type | Description |
|:-----|:-----|:------------|
| `image_reader` | Multimodal | Image OCR and analysis |
| `pdf_reader` | Multimodal | PDF parsing |
| `document_reader` | Multimodal | Word, Excel, CSV, TXT, Markdown reading |
| `data_reader` | Data | JSON, JSONL, YAML, TOML, INI, CSV, TSV, XML, NPY, NPZ, HDF5, Parquet, Feather |
| `mat_reader` | Data | MATLAB `.mat` reader |
| `file_browser` | Utility | Directory browsing by file type |
| `nc_reader` | Scientific | NetCDF scientific data analysis |

---

## Multimodal Capabilities

### Image Analysis
```bash
# Agent automatically uses image_reader for image files
snowcode agent2 "Analyze screenshots/ui.png and describe the layout"
```
- **OCR**: Extract text from images (requires `pytesseract`)
- **AI Analysis**: Send image to vision model (e.g., `qwen-vl-plus`) for detailed description

### Document Reading
- **PDF**: Extract text, metadata, specific pages
- **Word/Excel/CSV**: Parse structured content
- **HTML**: Extract readable content
- **Structured Data**: Inspect JSON, JSONL, YAML, TOML, INI, TSV, XML, NPY, NPZ, HDF5, Parquet, and Feather

### Scientific Data (NetCDF + MAT)
```bash
snowcode agent2 "Analyze data/ocean_temp.nc variables and statistics"
snowcode agent2 "Inspect experiments/model_output.mat"
```
- View dimensions, variables, attributes
- Extract variable data with slicing
- Compute statistics (min/max/mean/std)
- Read MATLAB `.mat` files through `mat_reader`

---

## Installation

```bash
# Basic installation
pip install -e .

# Better symbol extraction
pip install -e ".[ast]"

# With multimodal support (recommended)
pip install -e ".[multimodal]"

# With structured data readers
pip install -e ".[data]"

# With scientific data support
pip install -e ".[scientific]"

# All features
pip install -e ".[ast,multimodal,data,scientific,test]"
```

### Dependencies

| Package | Role |
|:--------|:-----|
| `click` | CLI framework |
| `numpy` | Vector storage |
| `sentence-transformers` | Embedding + reranking |
| `rich` | Terminal rendering |
| `prompt-toolkit` | Interactive REPL |
| `openai` | LLM API client |
| `Pillow` | Image processing |
| `PyMuPDF` | PDF reading |
| `h5py` | HDF5 / MAT v7.3 support |
| `pandas` + `pyarrow` | Parquet / Feather readers |
| `scipy` | MATLAB `.mat` support |

---

## Configuration

### Environment Variables

```bash
# DashScope (default)
export DASHSCOPE_API_KEY="sk-xxx"

# OpenAI-compatible
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Ollama (local)
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="codellama"
```

### Interactive Config

```bash
snowcode config
```

Stores settings in `.snowcode/config.json` per project.

---

## Data Format

All project data stored in `.snowcode/`:

```
.snowcode/
├── config.json              # Index & LLM configuration
├── embeddings.npy           # Vector matrix (N × 768)
├── metadata.json            # Chunk metadata
├── file_hashes.json         # Incremental indexing
├── bm25.json                # BM25 inverted index
├── chat_history.json        # Chat memory
├── repo_map.json            # Repository map and symbol graph cache
└── memory.jsonl             # Agent long-term memory
```

---

## Privacy & Security

- **Local Indexing**: All embedding, BM25, and parsing runs locally
- **Path Validation**: All file operations restricted to project root
- **Runtime Confirmation**: Write and shell tools can require approval
- **Diff Preview**: File edits show a preview before confirmation
- **Backup Safety**: Destructive operations create `.bak` backups
- **No Telemetry**: Zero analytics or external tracking
- **Optional LLM**: Works in retrieval-only mode without any API key

---

## License

[MIT](LICENSE)
