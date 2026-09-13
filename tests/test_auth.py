import base64
import importlib

import webapp.app as appmod


def _basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_no_auth_configured_allows_all_requests(monkeypatch):
    monkeypatch.delenv("BASIC_AUTH_USER", raising=False)
    monkeypatch.delenv("BASIC_AUTH_PASS", raising=False)
    importlib.reload(appmod)
    client = appmod.app.test_client()
    assert client.get("/").status_code == 200


def test_auth_configured_rejects_missing_or_wrong_credentials(monkeypatch):
    monkeypatch.setenv("BASIC_AUTH_USER", "demo")
    monkeypatch.setenv("BASIC_AUTH_PASS", "secret123")
    importlib.reload(appmod)
    client = appmod.app.test_client()

    assert client.get("/").status_code == 401
    assert client.get("/", headers=_basic("demo", "wrong")).status_code == 401
    assert client.get("/", headers=_basic("demo", "secret123")).status_code == 200

    # restore module state for any tests that import it after this one
    monkeypatch.delenv("BASIC_AUTH_USER", raising=False)
    monkeypatch.delenv("BASIC_AUTH_PASS", raising=False)
    importlib.reload(appmod)
