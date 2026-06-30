"""ComfyUI HTTP routes for the Animation Coordinator. Registered on import."""

from __future__ import annotations

from aiohttp import web

from andypack import api
from andypack.manifest import load_manifest

try:
    from server import PromptServer  # provided by ComfyUI
    _routes = PromptServer.instance.routes
except Exception:  # pragma: no cover - import-time guard outside ComfyUI
    _routes = None


def _manifest_from_request(request):
    path = request.query.get("manifest", "")
    return load_manifest(api.resolve_manifest_path(path))


if _routes is not None:

    @_routes.get("/anim_coord/characters")
    async def _characters(request):
        # The character list is always the pack's own <output>/characters dir,
        # resolved server-side — the client never points this at an arbitrary
        # filesystem path, so there's nothing to traverse out of.
        return web.json_response(api.list_characters(api.characters_dir() or ""))

    @_routes.get("/anim_coord/ping")
    async def _ping(request):
        # Lets the frontend confirm the pack's routes are live before enabling
        # the selector inputs.
        return web.json_response({"ok": True})

    @_routes.get("/anim_coord/manifest_options")
    async def _manifest_options(request):
        manifest = _manifest_from_request(request)
        return web.json_response(api.manifest_options(manifest))

    def _root_and_char(request):
        return api.character_root_and_name(
            request.query.get("character_dir", ""), request.query.get("character", "")
        )

    @_routes.get("/anim_coord/options")
    async def _options(request):
        root, character = _root_and_char(request)
        manifest = _manifest_from_request(request)
        return web.json_response(api.list_options(manifest, root, character))
