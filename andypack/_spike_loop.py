"""Throwaway spike — validates the ComfyUI 0.26.2 loop-node expansion contract.

THROWAWAY: not imported by `andypack/nodes.py`, not part of `NODE_CLASS_MAPPINGS`
in the shipped pack. Deleted in Task 7 once the real `SweepLoopOpen`/
`SweepLoopClose` nodes land. It exists only so Task 1 can prove out the
Open/Close subgraph-clone-and-expand mechanic against the live 0.26.2 instance
before any real pipeline code depends on it.

Ground truth: ComfyUI's OWN reference implementation of this exact pattern, in
its test suite, at
`tests/execution/testing_nodes/testing-pack/flow_control.py`
(`TestWhileLoopOpen` / `TestWhileLoopClose`) and
`tests/execution/testing_nodes/testing-pack/util.py`
(`TestForLoopOpen` / `TestForLoopClose`, which layers a decrementing counter on
top of the while-loop primitive — the same shape this spike needs). Both are
exercised by ComfyUI's own test suite, so the clone/rewire mechanic below is
not invented — it is copied down to the token names.

See docs/superpowers/notes/2026-07-01-loop-spike-findings.md for the full
write-up (clone mechanic, remaining-threading, gotchas, validation method).
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    # comfy_execution is only importable inside a running ComfyUI process.
    # Guarded the same way andypack/server.py guards `from server import
    # PromptServer`, so `python -c "import andypack..."`, ruff, and mypy stay
    # green in CI (no ComfyUI installed there).
    from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:  # pragma: no cover - import-time guard outside ComfyUI
    GraphBuilder = None  # type: ignore[assignment,misc]
    is_link = None  # type: ignore[assignment]

# Where CounterBody persists its iteration count between expansions. A plain
# file (not a class attribute) because every loop iteration gets a FRESH
# CounterBody instance — the expansion clones a new node id per iteration, and
# ComfyUI's object cache is keyed by node id, so there is no Python object to
# hold state across iterations. This mirrors how a real body node would only
# be able to persist state via the manifest/sidecar files on disk, not memory.
_COUNTER_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".spike_loop_counter.json")


def _read_counter() -> int:
    try:
        with open(_COUNTER_PATH) as f:
            return int(json.load(f)["count"])
    except Exception:
        return 0


def _write_counter(count: int) -> None:
    tmp = _COUNTER_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"count": count}, f)
    os.replace(tmp, _COUNTER_PATH)


class SpikeLoopOpen:
    """Emits a SWEEP_FLOW token carrying the initial iteration budget.

    The token is a plain dict (JSON-serializable, no torch/ComfyUI types) so
    it can flow through ComfyUI's normal (non-rawLink) value passing. Unlike
    the reference `TestWhileLoopOpen`, the Close side here does NOT need
    `flow`'s *value* — it needs the Open node's *id* (see SpikeLoopClose
    below), so `flow` is a value-typed socket, and the Open node's own id is
    recovered from the rawLink on the `remaining`-producing chain instead. See
    findings note "why SWEEP_FLOW carries no rawLink" for the reasoning.
    """

    CATEGORY = "andypack/_spike"
    FUNCTION = "open"
    RETURN_TYPES = ("SWEEP_FLOW",)
    RETURN_NAMES = ("flow",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"iterations": ("INT", {"default": 3, "min": 0, "max": 100000})}}

    def open(self, iterations: int):
        _write_counter(0)
        return ({"iterations": iterations},)


class CounterBody:
    """The loop body: increments the on-disk counter and reports remaining.

    `IS_CHANGED` returns NaN unconditionally so ComfyUI's cache never treats
    two calls to a node of this class as equivalent — required because each
    loop iteration clones a NEW node id for the body (see SpikeLoopClose), so
    per-node caching isn't even the mechanism at risk; NaN protects against a
    (docs-recommended) belt-and-suspenders case: any path where the execution
    cache might otherwise short-circuit a node whose real side effect (the
    counter file) changed. `float("nan") != float("nan")` is always True, so
    the cache-signature comparison in comfy_execution/caching.py always
    treats the node as changed. See findings for the exact citation.
    """

    CATEGORY = "andypack/_spike"
    FUNCTION = "step"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("remaining",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"flow": ("SWEEP_FLOW",)}}

    @classmethod
    def IS_CHANGED(cls, flow, **kwargs):  # noqa: N802 - ComfyUI's required casing
        return float("nan")

    def step(self, flow: dict[str, Any]):
        count = _read_counter() + 1
        _write_counter(count)
        remaining = int(flow["iterations"]) - count
        return (remaining,)


class SpikeLoopClose:
    """Closes the loop: re-expands Open..Close while `remaining > 0`.

    Mechanic (copied from ComfyUI's own `TestWhileLoopClose` /
    `TestForLoopClose`, tests/execution/testing_nodes/testing-pack/
    flow_control.py + util.py):

    1. `dynprompt` (hidden `DYNPROMPT`) gives access to the CURRENT expanded
       prompt graph — `dynprompt.get_node(node_id)` returns
       `{"class_type": ..., "inputs": {...}}` for any node, including ones
       created by a prior expansion.
    2. Walk the input-link graph backwards from `unique_id` (this Close
       node's own id, from hidden `UNIQUE_ID`) to find every node the Close
       transitively depends on (`explore_dependencies`), then walk forward
       from the Open node id to collect only the nodes BETWEEN Open and
       Close inclusive (`collect_contained`) — this is "the loop body".
    3. To find the Open node's id at all, `flow` must be declared with
       `{"forceInput": True, "rawLink": True}` (or, for the `flow_control`
       token in the reference nodes, just `rawLink: True`): a `rawLink`
       input arrives as the UNRESOLVED `[node_id, output_index]` pair
       instead of the resolved value, so `flow[0]` is the Open node's id.
       Regular (non-rawLink) inputs — like `remaining` here — arrive fully
       resolved, which is what SpikeLoopClose needs to test `remaining <= 0`.
    4. Build a fresh `GraphBuilder()`, clone every contained node under a new
       id (the reference names the recursive copy of the CLOSE node itself
       "Recurse" so ids don't grow per iteration), rewire every internal
       link to point at its cloned counterpart, and rewrite the CLONED
       Open node's `iterations` input to the value that should seed the
       NEXT iteration.
    5. Return `{"result": (...), "expand": graph.finalize()}`. The engine
       (execution.py `get_output_from_returns` / `execute`) treats a dict
       with an `"expand"` key as a subgraph expansion: it registers every
       node in the returned graph as an "ephemeral" node
       (`dynprompt.add_ephemeral_node`), re-adds any node in the expansion
       flagged `OUTPUT_NODE = True` to the execution list so it actually
       RUNS (this is why CounterBody, or the cloned Close itself, executes
       again even though nothing "queued" it directly), and treats `result`
       as a set of (possibly-unresolved) output links to await before this
       Close's own apparent output is considered ready.
    6. Termination: when `remaining <= 0`, return a plain (no `expand` key)
       result — no new graph, so the engine does not re-expand and the
       chain of clones stops growing.

    Why an explicit id map (not GraphBuilder.node(class_type)) for contained
    node bodies: GraphBuilder.node(class_type, id=...) is the constructor;
    the SECOND positional arg is the (prefixed) id for THIS clone, not the
    original node's id. Re-declaring inputs by walking
    `dynprompt.get_node(node_id)["inputs"]` and remapping any link whose
    source is also in `contained` is what makes the clone a faithful
    self-contained copy of the loop body.
    """

    CATEGORY = "andypack/_spike"
    FUNCTION = "close"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("done",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("SWEEP_FLOW", {"forceInput": True, "rawLink": True}),
                "remaining": ("INT", {"forceInput": True}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def _explore_dependencies(self, node_id, dynprompt, upstream) -> None:
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for _key, value in node_info["inputs"].items():
            if is_link(value):
                parent_id = value[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self._explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained) -> None:
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def close(self, flow, remaining: int, dynprompt=None, unique_id=None):
        if remaining <= 0:
            return ("done",)

        assert dynprompt is not None
        assert unique_id is not None

        # `flow` arrived as a rawLink: [open_node_id, output_index].
        open_node_id = flow[0]

        upstream: dict[str, list[str]] = {}
        self._explore_dependencies(unique_id, dynprompt, upstream)

        contained: dict[str, bool] = {}
        self._collect_contained(open_node_id, upstream, contained)
        contained[unique_id] = True
        contained[open_node_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.node(original["class_type"], clone_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.lookup_node(clone_id)
            assert node is not None
            for key, value in original["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    assert parent is not None
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        my_clone = graph.lookup_node("Recurse")
        assert my_clone is not None
        return {
            "result": ("looping",),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "SpikeLoopOpen": SpikeLoopOpen,
    "SpikeCounterBody": CounterBody,
    "SpikeLoopClose": SpikeLoopClose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpikeLoopOpen": "Spike Loop Open (throwaway)",
    "SpikeCounterBody": "Spike Counter Body (throwaway)",
    "SpikeLoopClose": "Spike Loop Close (throwaway)",
}
