<div align="center">

# Snowcode

**A Local RAG-Powered Code Intelligence Engine with Multimodal Agent**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Version](https://img.shields.io/badge/Version-0.3.1-blue.svg)](https://github.com/HUAYUE1024/Snowcode)
[![License: MIT](https://img.shields.io/badge/License-MIT-0DA338.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#privacy--security)

*Query, analyze, and modify codebases through natural language. Features hybrid RAG retrieval, autonomous ReAct agents with planning & memory, and multimodal tools for images, PDFs, and scientific data.*

</div>

---

## Features

| Feature | Description |
|:--------|:------------|
| **Hybrid Retrieval** | Dense (vector) + Sparse (BM25) + Cross-encoder reranking |
| **Agent v2** | Enhanced ReAct agent with planning, memory, and 12+ tools |
| **Multi-Agent** | Coordinator-Worker architecture for complex task decomposition |
| **Multimodal** | Image analysis (OCR + AI), PDF parsing, document reading |
| **Scientific Data** | NetCDF (.nc) file support for climate/ocean data analysis |
| **Interactive Chat** | Persistent session with command history (`agent-chat`) |
| **File Creation** | Agent can write files, generate reports, and modify code |
| **Multi-LLM** | DashScope / OpenAI-compatible / Ollama backends |
| **Privacy** | All indexing local; only optional LLM calls leave the machine |

---

## Quick Start

```bash
# Install
pip install -e ".[multimodal]"

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

### Tool Suite (12 tools)

| Tool | Type | Description |
|:-----|:-----|:------------|
| `search` | Read | Semantic code search |
| `read_file` | Read | Read full file content |
| `find_pattern` | Read | Regex pattern search |
| `list_dir` | Read | Directory structure |
| `write_file` | Write | Create/overwrite file |
| `search_replace` | Write | Find-and-replace code |
| `shell` | Execute | Run terminal commands |
| `image_reader` | Multimodal | Image OCR + AI analysis |
| `pdf_reader` | Multimodal | PDF document parsing |
| `document_reader` | Multimodal | Word/Excel/CSV/TXT reading |
| `file_browser` | Multimodal | Directory file listing |
| `nc_reader` | Scientific | NetCDF data analysis |

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
- **Word/Excel/CSV**: Parse structured data
- **HTML**: Extract readable content

### Scientific Data (NetCDF)
```bash
snowcode agent2 "Analyze data/ocean_temp.nc variables and statistics"
```
- View dimensions, variables, attributes
- Extract variable data with slicing
- Compute statistics (min/max/mean/std/percentiles)

---

## Installation

```bash
# Basic installation
pip install -e .

# With multimodal support (recommended)
pip install -e ".[multimodal]"

# With scientific data support
pip install -e ".[scientific]"

# All features
pip install -e ".[multimodal,scientific]"
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
└── memory.jsonl             # Agent long-term memory
```

---

## Privacy & Security

- **Local Indexing**: All embedding, BM25, and parsing runs locally
- **Path Validation**: All file operations restricted to project root
- **Backup Safety**: Destructive operations create `.bak` backups
- **No Telemetry**: Zero analytics or external tracking
- **Optional LLM**: Works in retrieval-only mode without any API key

---

## License

[MIT](LICENSE)
