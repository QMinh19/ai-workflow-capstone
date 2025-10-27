import os
import json
import tempfile
import pytest
from src.api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_missing_params(client):
    res = client.get('/predict')
    assert res.status_code == 400

def test_logfile_not_found(client, monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    res = client.get('/logfile?type=predict')
    assert res.status_code == 404
