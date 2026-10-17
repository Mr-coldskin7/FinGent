"""API 鉴权：判定规则（纯函数）+ HTTP 层（TestClient）。"""

import pytest
from fastapi.testclient import TestClient

import api_server
from api_auth import decide, is_public, validate_token

TOKEN = "test-token-0123456789abcdef"  # 长度 >= 16


# ── 纯函数：判定表 ──────────────────────────────────────────────────

class TestDecide:
    def test_public_read_only_paths_need_no_token(self):
        for path in ("/api/v1/market", "/api/v1/eval/runs", "/api/v1/eval/runs/20261017_112055",
                     "/api/v1/backtest-chart", "/health", "/docs"):
            d = decide(method="GET", path=path, authorization=None,
                       client_host="203.0.113.9", token=TOKEN)
            assert d.allowed, path

    def test_options_preflight_is_not_blocked(self):
        d = decide(method="OPTIONS", path="/api/v1/feedback", authorization=None,
                   client_host="203.0.113.9", token=TOKEN)
        assert d.allowed

    @pytest.mark.parametrize("method,path", [
        ("POST", "/api/v1/feedback"),
        ("POST", "/api/v1/chat"),
        ("POST", "/api/v1/chat-stream"),
        ("PUT", "/api/v1/model/config"),
        ("POST", "/api/v1/model/test"),
        ("PATCH", "/api/v1/memory/rules/3"),
        ("DELETE", "/api/v1/memory/rules/3"),
        ("PUT", "/api/v1/agents/morefit"),
        ("POST", "/api/v1/agents/reload"),
        ("POST", "/api/v1/backtest-stream"),
        ("GET", "/api/v1/agents"),
        ("GET", "/api/v1/model/config"),
        ("GET", "/api/v1/memory/rules"),
    ])
    def test_protected_without_token_is_401_when_token_configured(self, method, path):
        d = decide(method=method, path=path, authorization=None,
                   client_host="127.0.0.1", token=TOKEN)
        assert not d.allowed and d.status == 401 and d.code == "unauthorized"

    def test_wrong_token_and_wrong_scheme_are_rejected(self):
        # 错误 token、错误 scheme、空 Bearer、裸 token（没有 Bearer 前缀）都必须是 401
        for header in ("Bearer wrong-token-xxxxxxxx", "Basic dGVzdDp0ZXN0", "Bearer ", TOKEN):
            d = decide(method="GET", path="/api/v1/agents", authorization=header,
                       client_host="127.0.0.1", token=TOKEN)
            assert d.status == 401, header

    def test_correct_bearer_token_is_allowed_from_any_host(self):
        d = decide(method="POST", path="/api/v1/feedback", authorization=f"Bearer {TOKEN}",
                   client_host="203.0.113.9", token=TOKEN)
        assert d.allowed

    def test_no_token_configured_allows_loopback_only(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            assert decide(method="POST", path="/api/v1/feedback", authorization=None,
                          client_host=host, token="").allowed, host
        d = decide(method="POST", path="/api/v1/feedback", authorization=None,
                   client_host="192.168.1.20", token="")
        assert not d.allowed and d.status == 403 and d.code == "forbidden"

    def test_is_public_only_for_known_paths(self):
        assert is_public("GET", "/api/v1/market")
        assert not is_public("GET", "/api/v1/market/extra")
        assert not is_public("GET", "/api/v1/memory/accuracy")


class TestValidateToken:
    def test_empty_means_disabled(self):
        validate_token("")  # 不抛异常

    def test_short_token_is_rejected(self):
        with pytest.raises(ValueError, match="至少 16 位"):
            validate_token("short")

    def test_long_enough_token_passes(self):
        validate_token("x" * 16)


# ── HTTP 层 ─────────────────────────────────────────────────────────

@pytest.fixture
def token_on(monkeypatch):
    monkeypatch.setattr(api_server.settings, "api_token", TOKEN)
    yield


@pytest.fixture
def token_off(monkeypatch):
    monkeypatch.setattr(api_server.settings, "api_token", "")
    yield


def _client(host="127.0.0.1"):
    return TestClient(api_server.app, client=(host, 50000))


def test_http_protected_endpoint_401_with_www_authenticate(token_on):
    r = _client().get("/api/v1/agents")
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"
    body = r.json()
    assert body["success"] is False and body["code"] == "unauthorized"


def test_http_protected_endpoint_ok_with_token(token_on):
    r = _client().get("/api/v1/agents", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_http_public_endpoint_needs_no_token(token_on):
    r = _client("203.0.113.9").get("/api/v1/eval/runs")
    assert r.status_code == 200


def test_http_write_without_token_does_not_touch_db(token_on):
    r = _client().post("/api/v1/feedback", json={
        "session_id": "s", "stock_symbol": "NVDA", "feedback": "agree", "user_id": "u-auth",
    })
    assert r.status_code == 401


def test_http_remote_host_without_token_configured_is_403(token_off):
    r = _client("203.0.113.9").get("/api/v1/agents")
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden"


def test_http_loopback_without_token_configured_is_allowed(token_off):
    r = _client("127.0.0.1").get("/api/v1/agents")
    assert r.status_code == 200


def test_http_cors_preflight_passes_without_credentials(token_on):
    r = _client("203.0.113.9").options("/api/v1/feedback", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    # 不断言具体来源：实际 CORS 来源取决于本机 .env（CORS_ORIGINS），这里只验证预检本身不被鉴权拦截
    assert r.status_code == 200
    assert "access-control-allow-origin" in r.headers
