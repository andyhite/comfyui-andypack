import pytest
from aiohttp import web

from andypack import server


class _FakeRequest:
    """Minimal stand-in for an aiohttp request: just the query mapping the route
    helpers read (the routes themselves only register inside ComfyUI)."""

    def __init__(self, query):
        self.query = query


def test_manifest_from_request_defaults_and_wraps_errors(tmp_path, monkeypatch):
    # An empty/missing manifest param must fall back to default.json (not resolve to
    # the manifests directory), and an unreadable manifest must surface as a JSON
    # 400 rather than an unhandled 500 from open()-ing a directory.
    seen = {}

    def fake_safe(name):
        seen["name"] = name
        return str(tmp_path)  # a directory -> open() raises IsADirectoryError

    monkeypatch.setattr(server.api, "safe_manifest_path", fake_safe)
    with pytest.raises(web.HTTPBadRequest):
        server._manifest_from_request(_FakeRequest({}))
    assert seen["name"] == "default.json"  # empty param falls back to the default


# NOTE: the /anim_coord/thumb route is defined inside the `if _routes is not None:`
# block in server.py, so it is only registered when running inside ComfyUI
# (where PromptServer is importable). In tests _routes is None and the route
# function is never defined, so it cannot be unit-tested here. The route's two
# helpers — api.thumb_path and images.thumbnail_data_uri — are tested thoroughly
# in tests/test_api.py and tests/test_images.py.
