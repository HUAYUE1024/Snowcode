# LLM Configuration Guide

This document provides comprehensive instructions for configuring Large Language Model (LLM) backends in codechat.

## Overview

codechat supports three LLM backends, automatically detected in priority order:

| Priority | Backend | Protocol | Offline | Default Model |
|:---------|:--------|:---------|:--------|:--------------|
| 1 | **DashScope** (Alibaba Qwen) | OpenAI-compatible | No | `qwen-flash` |
| 2 | **OpenAI Compatible** | OpenAI API | No | `gpt-4o-mini` |
| 3 | **Ollama** (Local) | Ollama HTTP API | Yes | `codellama` |

If no LLM is configured, codechat operates in **retrieval-only mode** — returning relevant code snippets without AI-generated analysis.

---

## 1. DashScope (Alibaba Qwen) — Recommended for China

DashScope provides the Qwen series of models with full OpenAI-compatible API.

### Setup

```bash
# Windows (cmd)
set DASHSCOPE_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="sk-your-key-here"

# Linux / macOS
export DASHSCOPE_API_KEY=sk-your-key-here
```

### Permanent Configuration

```bash
# Windows — persists across terminal sessions
setx DASHSCOPE_API_KEY "sk-your-key-here"

# Linux / macOS — add to shell profile
echo 'export DASHSCOPE_API_KEY=sk-your-key-here' >> ~/.bashrc
source ~/.bashrc

# macOS (zsh)
echo 'export DASHSCOPE_API_KEY=sk-your-key-here' >> ~/.zshrc
source ~/.zshrc
```

### Available Models

| Model | ID | Speed | Quality | Cost | Thinking |
|:------|:---|:------|:--------|:-----|:---------|
| Qwen Flash | `qwen-flash` | Fast | Good | Low | No |
| Qwen Plus | `qwen-plus` | Medium | Better | Medium | No |
| Qwen Max | `qwen-max` | Slow | Best | High | No |
| Qwen Long | `qwen-long` | Medium | Good | Low | No |
| QwQ (Reasoning) | `qwq-32b` | Slow | Best | Medium | Yes |
| DeepSeek V3 | `deepseek-v3` | Medium | Good | Low | No |
| DeepSeek R1 | `deepseek-r1` | Slow | Best | Medium | Yes |

```bash
# Switch model
codechat ask "question" -m qwen-plus

# Or via environment variable
set CODECHAT_MODEL=qwen-plus
```

### Thinking / Reasoning Mode

Some models (QwQ, DeepSeek R1) support extended thinking with visible reasoning tokens.

```bash
# Enable thinking mode
set CODECHAT_THINKING=1
codechat ask "complex question" --show-thinking

# Disable (default)
set CODECHAT_THINKING=0
```

Output with thinking enabled:
```
=== Thinking ===
Let me analyze this step by step...
The user is asking about vector storage...
I need to look at store.py first...

=== Answer ===
The vector store persists data as...
```

### Custom Base URL

DashScope default endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`

Override if needed:
```bash
set DASHSCOPE_BASE_URL=https://your-proxy.example.com/v1
```

### API Key Acquisition

1. Visit [DashScope Console](https://dashscope.console.aliyun.com/)
2. Register / Login
3. Navigate to **API-KEY Management**
4. Create a new API Key
5. Copy the key (starts with `sk-`)

---

## 2. OpenAI Compatible API

Any service implementing the OpenAI Chat Completions API format works.

### Setup

```bash
# OpenAI official
set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek
set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set CODECHAT_MODEL=deepseek-chat

# Moonshot (Kimi)
set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://api.moonshot.cn/v1
set CODECHAT_MODEL=moonshot-v1-8k

# Zhipu (GLM)
set OPENAI_API_KEY=your-key
set OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set CODECHAT_MODEL=glm-4

# SiliconFlow (free tier available)
set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://api.siliconflow.cn/v1
set CODECHAT_MODEL=Qwen/Qwen2.5-72B-Instruct

# Together AI
set OPENAI_API_KEY=your-key
set OPENAI_BASE_URL=https://api.together.xyz/v1
set CODECHAT_MODEL=meta-llama/Llama-3-70b-chat-hf
```

### Provider Reference

| Provider | Base URL | Recommended Model | Free Tier |
|:---------|:---------|:------------------|:----------|
| OpenAI | `api.openai.com/v1` | `gpt-4o-mini` | No |
| DeepSeek | `api.deepseek.com/v1` | `deepseek-chat` | Limited |
| Moonshot | `api.moonshot.cn/v1` | `moonshot-v1-8k` | Yes (limited) |
| Zhipu | `open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | Yes |
| SiliconFlow | `api.siliconflow.cn/v1` | `Qwen/Qwen2.5-72B-Instruct` | Yes |
| Together AI | `api.together.xyz/v1` | `meta-llama/Llama-3-70b-chat-hf` | $5 credit |
| Groq | `api.groq.com/openai/v1` | `llama3-70b-8192` | Yes (rate limited) |
| Fireworks | `api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3-70b-instruct` | Yes |

### Environment Variables Summary

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `OPENAI_API_KEY` | Yes | — | API authentication key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | API endpoint |
| `CODECHAT_MODEL` | No | `gpt-4o-mini` | Model name |
| `CODECHAT_THINKING` | No | `0` | Enable thinking mode (`0` or `1`) |

---

## 3. Ollama (Local / Offline)

Ollama runs models entirely locally — zero network dependency, zero cost, full privacy.

### Installation

```bash
# Windows — download from https://ollama.ai/download

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### Pull a Model

```bash
# Recommended for code analysis
ollama pull qwen2.5-coder:7b        # 4.7GB — Chinese code understanding
ollama pull deepseek-coder-v2:16b    # 8.9GB — Strong reasoning
ollama pull codellama:13b            # 7.4GB — Classic, English-focused
ollama pull codegemma:7b             # 5.0GB — Lightweight

# General purpose
ollama pull llama3.1:8b              # 4.7GB — Good all-rounder
ollama pull qwen2.5:14b              # 9.0GB — Strong Chinese
ollama pull mistral:7b               # 4.1GB — Fast, multilingual
```

### Configure codechat

```bash
# Windows
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen2.5-coder

# Linux / macOS
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5-coder
```

### Verify

```bash
# Check Ollama is running
ollama list

# Test in codechat
codechat ask "hello" --show-sources
```

### Model Recommendations

| Model | Size | RAM Needed | Code Quality | Chinese | Speed |
|:------|:-----|:-----------|:-------------|:--------|:------|
| `qwen2.5-coder:7b` | 4.7GB | 8GB | Good | Excellent | Fast |
| `deepseek-coder-v2:16b` | 8.9GB | 16GB | Best | Good | Medium |
| `codellama:13b` | 7.4GB | 16GB | Good | Poor | Medium |
| `codegemma:7b` | 5.0GB | 8GB | Fair | Poor | Fast |
| `llama3.1:8b` | 4.7GB | 8GB | Fair | Fair | Fast |

### Remote Ollama

If Ollama runs on a different machine:

```bash
set OLLAMA_URL=http://192.168.1.100:11434
```

---

## 4. Embedding Model Configuration

The embedding model converts code into vectors for semantic search. This is separate from the LLM.

### Default Model

`all-mpnet-base-v2` — 768 dimensions, 420MB, best quality for English/mixed content.

### Available Models

| Model | Dimensions | Size | Quality | Speed | Multilingual |
|:------|:-----------|:-----|:--------|:------|:-------------|
| `all-mpnet-base-v2` | 768 | 420MB | Best | Medium | Partial |
| `all-MiniLM-L6-v2` | 384 | 90MB | Good | Fast | Partial |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | Good | Medium | Full |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Good | Fast | English |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Best | Medium | English |
| `shibing624/text2vec-base-chinese` | 768 | 400MB | Good | Medium | Chinese |

### Switch Embedding Model

```bash
# First ingest — must use --reset when switching
codechat ingest --reset -m all-mpnet-base-v2

# Subsequent ingests use the same model automatically
codechat ingest  # incremental, keeps same model
```

### Model Storage

Models are cached in `~/.cache/huggingface/hub/` after first download. For offline use, pre-download the model on a connected machine and copy the cache directory.

### China Network

codechat automatically uses the HuggingFace mirror `hf-mirror.com` for model downloads. No manual configuration needed. If the mirror fails, set manually:

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

---

## 5. Priority and Detection Logic

```
codechat ask "question"
  │
  ├─ DASHSCOPE_API_KEY set?  →  Use DashScope (qwen-flash)
  │
  ├─ OPENAI_API_KEY set?     →  Use OpenAI-compatible (gpt-4o-mini)
  │
  ├─ OLLAMA_URL set?         →  Use Ollama (codellama)
  │
  └─ None set                →  Retrieval-only mode (no LLM)
```

**Precedence is fixed**: DashScope > OpenAI > Ollama. To use Ollama when DashScope is configured, unset `DASHSCOPE_API_KEY`:

```bash
# Temporarily use Ollama
set DASHSCOPE_API_KEY=
codechat ask "question"
```

---

## 6. Troubleshooting

### "No LLM configured"

**Cause**: No API key or URL environment variable is set.

**Fix**:
```bash
set DASHSCOPE_API_KEY=sk-xxx
codechat ask "test"
```

### SSL Certificate Error

**Cause**: Corporate firewall or China network blocking HTTPS.

**Fix**: codechat automatically sets `HF_HUB_DISABLE_SSL_VERIFICATION=1` for model downloads. If LLM API calls fail with SSL errors:
```bash
# Windows — disable SSL system-wide (use with caution)
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
```

### Connection Refused (WinError 10061)

**Cause**: Ollama is not running or API key is not set.

**Fix**:
```bash
# Check Ollama
ollama list

# Or set API key
set DASHSCOPE_API_KEY=sk-xxx
```

### Connection Timeout (WinError 10060)

**Cause**: Network unreachable or mirror down.

**Fix**:
```bash
# Try a different mirror
set HF_ENDPOINT=https://huggingface.co

# Or download models manually and place in cache
```

### Model Not Found

**Cause**: Model name doesn't exist on the provider.

**Fix**: Check the model ID is correct for your provider:
```bash
codechat ask "test" -m qwen-flash    # DashScope
codechat ask "test" -m gpt-4o-mini   # OpenAI
codechat ask "test" -m llama3.1:8b   # Ollama
```

### Slow First Run

**Cause**: Embedding model downloading (~420MB for mpnet).

**Fix**: First run downloads the model. Subsequent runs use cached model. Pre-download if needed:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

---

## 7. Quick Reference

### Minimum Setup (DashScope)

```bash
set DASHSCOPE_API_KEY=sk-xxx
codechat ingest
codechat ask "hello"
```

### Minimum Setup (Ollama)

```bash
ollama pull qwen2.5-coder:7b
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen2.5-coder
codechat ingest
codechat ask "hello"
```

### All Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DASHSCOPE_API_KEY` | — | DashScope API key |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | DashScope endpoint |
| `OPENAI_API_KEY` | — | OpenAI-compatible API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OLLAMA_URL` | — | Ollama server URL |
| `OLLAMA_MODEL` | `codellama` | Ollama model name |
| `CODECHAT_MODEL` | `qwen-flash` / `gpt-4o-mini` | Override default model |
| `CODECHAT_THINKING` | `0` | Enable thinking mode |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror |
