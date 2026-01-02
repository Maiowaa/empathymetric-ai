"""
Basic tests for EmpathyMetric AI
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that basic imports work"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    assert True

def test_app_structure():
    """Test app.py structure"""
    assert os.path.exists('app.py')
    assert os.path.exists('requirements.txt')
    assert os.path.exists('README.md')

# Note: Full API tests require model download
# Add more comprehensive tests after model is available
