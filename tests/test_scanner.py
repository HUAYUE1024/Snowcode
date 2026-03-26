import pytest
from pathlib import Path
from codechat.scanner import _load_ignore_patterns, scan_files

def test_load_ignore_patterns(tmp_path):
    # Create mock .codechatignore
    ignore_file = tmp_path / ".codechatignore"
    ignore_file.write_text("*.log\nnode_modules/", encoding="utf-8")
    
    spec = _load_ignore_patterns(tmp_path)
    assert spec is not None
    assert spec.match_file("test.log")
    assert spec.match_file("node_modules/index.js")
    assert not spec.match_file("src/main.py")

def test_scan_files_with_ignore(tmp_path):
    # Create test directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "src" / "test.log").write_text("error", encoding="utf-8")
    
    ignore_file = tmp_path / ".codechatignore"
    ignore_file.write_text("*.log", encoding="utf-8")
    
    files = scan_files(tmp_path)
    file_names = [f.name for f in files]
    
    assert "main.py" in file_names
    assert "test.log" not in file_names
