from frontend import approval_history


class _Response:
    status_code = 200

    @staticmethod
    def json():
        return {"pending": [{"id": 7}]}


def test_approval_history_uses_authenticated_http_wrapper(monkeypatch):
    captured = {}

    def fake_get_pending_list(backend_url: str, timeout: int, status: str):
        captured.update(backend_url=backend_url, timeout=timeout, status=status)
        return _Response()

    monkeypatch.setattr(approval_history, "get_pending_list", fake_get_pending_list)

    assert approval_history._load_approval_items("http://backend:8000", "pending") == [{"id": 7}]
    assert captured == {
        "backend_url": "http://backend:8000",
        "timeout": 10,
        "status": "pending",
    }
