"""Manifest loading, validation, and ref classification (pure stdlib)."""

from __future__ import annotations

import json
import warnings
from typing import Any

Manifest = dict[str, Any]


class ManifestError(Exception):
    """Raised when a manifest is structurally invalid or has a dependency cycle."""


def node_kind(manifest: Manifest, ref: str) -> str:
    """Classify a ref as 'concept', 'pose', or 'animation'."""
    if ref == "concept":
        return "concept"
    if ref in manifest.get("poses", {}):
        return "pose"
    if ref in manifest.get("animations", {}):
        return "animation"
    raise ManifestError(f"unknown ref: {ref!r}")


def _validate_refs(manifest: Manifest) -> None:
    for pid, pose in manifest.get("poses", {}).items():
        frm = pose.get("from")
        if not isinstance(frm, dict) or "ref" not in frm:
            raise ManifestError(f"pose {pid!r} missing 'from.ref'")
        if node_kind(manifest, frm["ref"]) == "animation":
            raise ManifestError(f"pose {pid!r} 'from' must reference concept or a pose")
        if not isinstance(pose.get("directions"), dict):
            raise ManifestError(f"pose {pid!r} missing 'directions' map")
    default_start = manifest.get("defaults", {}).get("start_from")
    if default_start is not None:
        if not isinstance(default_start, dict) or "ref" not in default_start:
            raise ManifestError("defaults.start_from missing 'ref'")
        node_kind(manifest, default_start["ref"])  # raises on unknown
    for aid, anim in manifest.get("animations", {}).items():
        for slot in ("start_from", "end_at"):
            dep = anim.get(slot)
            if dep is not None:
                if not isinstance(dep, dict) or "ref" not in dep:
                    raise ManifestError(f"animation {aid!r} {slot} missing 'ref'")
                node_kind(manifest, dep["ref"])  # raises on unknown
        # Every animation needs a start image for I2V: explicit start_from or the
        # manifest-level defaults.start_from.
        if anim.get("start_from") is None and default_start is None:
            raise ManifestError(
                f"animation {aid!r} has no 'start_from' and no defaults.start_from "
                "(I2V needs a start image)"
            )
        if not isinstance(anim.get("directions"), dict):
            raise ManifestError(f"animation {aid!r} missing 'directions' map")


def _dependency_edges(manifest: Manifest) -> dict[str, list[str]]:
    """Adjacency `node -> [dependency ids]` over poses + animations. The `concept`
    seed is a leaf (it depends on nothing) and is omitted from edge targets."""
    edges: dict[str, list[str]] = {}

    def add(node: str, ref: str | None) -> None:
        edges.setdefault(node, [])
        if ref and ref != "concept":
            edges[node].append(ref)

    default_start = manifest.get("defaults", {}).get("start_from")
    for pid, pose in manifest.get("poses", {}).items():
        add(pid, pose.get("from", {}).get("ref"))
    for aid, anim in manifest.get("animations", {}).items():
        edges.setdefault(aid, [])
        # An animation with no explicit start_from depends on defaults.start_from
        # (its I2V seed), so fold that into the graph for ordering + cycle checks.
        start = anim.get("start_from") or default_start
        for dep in (start, anim.get("end_at")):
            if dep:
                add(aid, dep.get("ref"))
    return edges


def _detect_cycles(manifest: Manifest) -> None:
    edges = _dependency_edges(manifest)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in edges}

    def dfs(node: str) -> None:
        color[node] = GRAY
        for nxt in edges.get(node, []):
            if color.get(nxt, BLACK) == GRAY:
                raise ManifestError(f"dependency cycle: {node} -> {nxt}")
            if color.get(nxt, WHITE) == WHITE:
                dfs(nxt)
        color[node] = BLACK

    for node in list(edges):
        if color[node] == WHITE:
            dfs(node)


def collect_warnings(manifest: Manifest) -> list[str]:
    """Non-fatal lint findings: Wan-unfriendly lengths and directions declared on
    an entity but absent from the canonical top-level `directions` list. Pure
    (returns strings); `validate_manifest` emits these via `warnings.warn`, and
    the ManifestLint node surfaces them in the graph."""
    out: list[str] = []
    default_len = manifest.get("defaults", {}).get("length")
    for aid, anim in manifest.get("animations", {}).items():
        length = anim.get("length", default_len)
        if isinstance(length, int) and (length - 1) % 4 != 0:
            out.append(f"animation {aid!r} length {length} is not 4n+1 (Wan-unfriendly)")

    canonical = manifest.get("directions")
    if isinstance(canonical, list) and canonical:
        known = set(canonical)
        for kind, collection in (("pose", "poses"), ("animation", "animations")):
            for eid, entity in manifest.get(collection, {}).items():
                for direction in entity.get("directions", {}) or {}:
                    if direction not in known:
                        out.append(
                            f"{kind} {eid!r} direction {direction!r} is not in the "
                            "canonical 'directions' list"
                        )
    return out


def topo_order(manifest: Manifest) -> list[str]:
    """Pose + animation ids in dependency order (a node appears after every node
    it depends on). Assumes the manifest is acyclic (validate_manifest enforces)."""
    edges = _dependency_edges(manifest)
    order: list[str] = []
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        for dep in edges.get(node, []):
            visit(dep)
        order.append(node)

    for node in edges:
        visit(node)
    return order


def validate_manifest(manifest: Manifest) -> None:
    """Structural validation + cycle detection. Raises ManifestError on failure."""
    if not isinstance(manifest.get("version"), int):
        raise ManifestError("manifest missing integer 'version'")
    for key in ("poses", "animations"):
        if not isinstance(manifest.get(key), dict):
            raise ManifestError(f"manifest missing '{key}' object")
    _validate_refs(manifest)
    _detect_cycles(manifest)
    for message in collect_warnings(manifest):
        warnings.warn(message)


def load_manifest(path: str) -> Manifest:
    """Load, validate, and return the manifest dict."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    validate_manifest(data)
    return data
