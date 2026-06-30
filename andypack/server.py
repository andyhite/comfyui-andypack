"""ComfyUI HTTP routes for the Animation Coordinator. Registered on import."""

from __future__ import annotations

import json

from aiohttp import web

from andypack import api
from andypack.manifest import ManifestError, load_manifest

try:
    from server import PromptServer  # provided by ComfyUI
    _routes = PromptServer.instance.routes
except Exception:  # pragma: no cover - import-time guard outside ComfyUI
    _routes = None


def _manifest_from_request(request):
    # An empty/missing `manifest` falls back to the conventional default.json (the
    # same name the loader node defaults to) rather than resolving to the manifests
    # directory itself. A bad/missing manifest returns a JSON 400, not an unhandled
    # 500 from open()-ing a directory or a nonexistent file.
    name = request.query.get("manifest") or "default.json"
    try:
        return load_manifest(api.resolve_manifest_path(name))
    except (OSError, ManifestError) as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(exc)}), content_type="application/json"
        ) from exc


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

    # --- sidebar GUI: manifest + character management ----------------------- #
    # Write routes are JSON-in / JSON-out and take NO client filesystem path: a
    # manifest is a bare basename validated server-side and resolved under the
    # pack's manifests dir; a character is a name snake-cased to one path segment
    # under the pack's characters dir. Nothing the client sends can escape those
    # trees. Edits are validated before they touch disk, so a bad payload returns
    # a 400 instead of corrupting a working file.

    @_routes.get("/anim_coord/manifests")
    async def _manifests(request):
        return web.json_response({"manifests": api.list_manifest_names()})

    @_routes.get("/anim_coord/manifest")
    async def _manifest(request):
        name = request.query.get("name") or "default.json"
        text = api.read_manifest_text(name)
        if text is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": f"manifest {name!r} not found"}),
                content_type="application/json",
            )
        return web.json_response({"name": name, "text": text})

    async def _json_body(request):
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "expected a JSON object body"}),
                content_type="application/json",
            )
        return body

    @_routes.post("/anim_coord/manifest/save")
    async def _manifest_save(request):
        body = await _json_body(request)
        result = api.save_manifest_text(
            str(body.get("name") or ""), str(body.get("content") or "")
        )
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)

    @_routes.get("/anim_coord/character")
    async def _character_get(request):
        root = api.characters_dir() or ""
        name = request.query.get("character", "")
        return web.json_response({"name": name, "layer": api.read_character_layer(root, name)})

    @_routes.post("/anim_coord/character/create")
    async def _character_create(request):
        body = await _json_body(request)
        root = api.characters_dir()
        if root is None:
            return web.json_response(
                {"ok": False, "error": "characters dir unavailable"}, status=400
            )
        result = api.create_character(root, str(body.get("character") or ""))
        return web.json_response(result, status=200 if result.get("ok") else 400)

    @_routes.post("/anim_coord/character/save")
    async def _character_save(request):
        body = await _json_body(request)
        root = api.characters_dir()
        if root is None:
            return web.json_response(
                {"ok": False, "error": "characters dir unavailable"}, status=400
            )
        result = api.save_character_layer(
            root, str(body.get("character") or ""),
            str(body.get("positive_prompt") or ""),
            str(body.get("negative_prompt") or ""),
        )
        return web.json_response(result, status=200 if result.get("ok") else 400)
