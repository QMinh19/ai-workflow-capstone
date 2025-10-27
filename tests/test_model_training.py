import os
import pytest
from solution_guidance.model import model_train
from pathlib import Path

def test_model_train_creates_models(tmp_path, monkeypatch):
    try:
        tmp_models = tmp_path / "models"
        os.chdir(str(tmp_path))
        model_train("cs-train", test=True)
        assert (Path("models")).exists()
    finally:
        os.chdir(cwd)
