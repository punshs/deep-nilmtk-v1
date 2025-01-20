import sys
import pytest

def test_python_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    assert "torch" in sys.modules