import pytest
from codechat.rag import _format_context, _build_prompt

def test_format_context():
    results = [
        {
            "metadata": {
                "file_path": "src/main.py",
                "start_line": 10,
                "end_line": 15
            },
            "content": "def hello():\n    print('world')"
        }
    ]
    
    formatted = _format_context(results)
    assert "[1] `src/main.py`" in formatted
    assert "```python" in formatted
    assert "def hello():" in formatted

def test_build_prompt():
    context = "Context info"
    question = "How to run?"
    prompt = _build_prompt(context, question)
    
    assert "Context info" in prompt
    assert "How to run?" in prompt
    assert "代码上下文" in prompt
