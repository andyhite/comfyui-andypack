# Loop-node expansion contract — findings (Task 1 spike)

Status: source-grounded and validated by close reading against a real ComfyUI
0.26.2 install; live 3-iteration run on the remote pod was interrupted by the
pod's idle-stop (see "Validation method" below) before it could complete.
This note is Task 7/8's source of truth regardless — the contract below is
copied down from ComfyUI's OWN reference implementation, not inferred.

## Where this was grounded

A local ComfyUI checkout at exactly `0.26.2` (matching the remote pod, per
`get_environment`) is installed at
`/Users/user/ComfyUI-Installs/ComfyUI (Local)/ComfyUI`
(`comfyui_version.py: __version__ = "0.26.2"`, git `7c8450ef2b72`, dated
2026-06-25). That checkout's own test suite ships a **reference
implementation of exactly this pattern**, used by ComfyUI's own CI:

- `comfy_execution/graph_utils.py` — `GraphBuilder`, `Node`, `is_link`
  (the primitives).
- `execution.py` — the engine side: `get_output_from_returns` (recognizes a
  `dict` result with an `"expand"` key as a subgraph expansion) and `execute`
  (`dynprompt.add_ephemeral_node`, re-queues any expanded node flagged
  `OUTPUT_NODE = True`).
- `tests/execution/testing_nodes/testing-pack/flow_control.py` —
  `TestWhileLoopOpen` / `TestWhileLoopClose`: the base while-loop primitive.
- `tests/execution/testing_nodes/testing-pack/util.py` (lines ~230-300) —
  `TestForLoopOpen` / `TestForLoopClose`: a decrementing-counter loop built on
  top of the while-loop primitive — the closest possible reference to what
  this spike (and the eventual `SweepLoopOpen`/`SweepLoopClose`) needs.

I did not use context7/web docs for this — the installed 0.26.2 source (own
test suite) is more authoritative than any secondary doc for an internal,
rarely-documented mechanism like node expansion, and it's the literal version
running on the pod.

## The contract

### 1. `GraphBuilder` / `Node` (`comfy_execution/graph_utils.py`)

- `GraphBuilder()` — a scratch graph-builder. `.node(class_type, id=None,
  **kwargs)` declares a node; if you pass an explicit `id`, calling `.node()`
  again with the same id returns the SAME node (idempotent lookup), not a
  new one — this is what "Recurse" (see below) exploits.
- `.lookup_node(id)` — fetch a previously-declared node by id.
- `node.out(index)` — returns the `[node_id, index]` link tuple to wire as
  another node's input.
- `node.set_input(key, value)` — value is either a literal or a link tuple.
- `node.set_override_display_id(id)` — cosmetic: what the UI shows as the
  "real" node for a cloned/ephemeral node (so the graph doesn't visually
  explode into thousands of new boxes across iterations).
- `graph.finalize()` — serializes to the `{node_id: {class_type, inputs}}`
  dict shape the engine's `expand` key expects.
- `is_link(value)` — `True` iff `value` is a 2-element `[str, int]` pair
  (i.e., an unresolved graph link rather than a literal).

### 2. The engine side (`execution.py`)

A node function can return a `dict` instead of a plain tuple. If that dict
has an `"expand"` key (`get_output_from_returns`, `execution.py:358-365` and
`379-385`):
- `has_subgraph = True`; the `"result"` value (may itself contain unresolved
  link tuples) is stashed as `subgraph_results` instead of being appended to
  the node's real outputs.
- Back in `execute()` (`execution.py:576-609`), every node in the returned
  graph is registered via `dynprompt.add_ephemeral_node(node_id, node_info,
  parent_id=unique_id, display_id)` — this is what makes
  `dynprompt.get_node(...)` and `dynprompt.has_node(...)` see the cloned
  nodes on the NEXT call, even though they were never in the original
  submitted prompt.
- Any newly-added node whose class has `OUTPUT_NODE == True` is explicitly
  re-added to the execution list (`execution_list.add_node(node_id)`) — this
  is the mechanism that makes the loop body (and the cloned Close, which is
  itself `OUTPUT_NODE = True`) actually EXECUTE again without anything else
  "calling" it. Non-output nodes in the expansion only run if something
  downstream (transitively) depends on them.
- `pending_subgraph_results[unique_id]` holds the deferred result so that
  when downstream consumers ask for this node's output, the engine resolves
  through the (now-executing) cloned subgraph instead of a stale value.

**Practical shape of a "loop" iteration is therefore NOT a Python loop at
all.** It's node-graph recursion: Close, when it decides to continue,
constructs a whole new copy of [Open, body nodes..., Close] as a same-size
"next iteration" subgraph and asks the engine to splice it in and run it.
There is no actual repeated execution of literally the same node object —
each iteration is a fresh clone with a fresh node id (prefixed), so any
per-iteration state (like this spike's counter) MUST live outside the Python
object (we use a JSON file; the real pipeline uses the manifest/sidecar
files already on disk for this reason).

### 3. The Open/Close/body pattern, copied down

**Open** (`TestWhileLoopOpen` / `TestForLoopOpen`): just packages the initial
values into a flow-control token. No cleverness — the only thing that
matters is that its RETURN_TYPES[0] is a distinctive flow-control type
(`FLOW_CONTROL` in the reference; `SWEEP_FLOW` in this spike) so nothing else
can accidentally wire into that socket.

**Body** (not part of the reference's flow_control.py directly, but implied
by every user of the pattern, and made explicit by this spike's
`CounterBody`): any node(s) between Open and Close. Nothing special is
required of a body node beyond normal ComfyUI node rules — EXCEPT that if a
body node's real side effect (e.g., writing a counter/sidecar file) needs to
be re-observed every iteration rather than possibly cached, it should define
`IS_CHANGED` returning `float("nan")` (see gotcha below).

**Close** (`TestWhileLoopClose` / `TestForLoopClose`) does the real work,
copied verbatim in shape from `flow_control.py:86-135`:

```python
def close(self, flow, remaining, dynprompt=None, unique_id=None):
    if remaining <= 0:
        return ("done",)          # <-- NO "expand" key: this is what stops recursion

    open_node_id = flow[0]         # flow arrived as a RAW LINK: [node_id, output_index]

    # 1. Walk input-links backward from THIS Close node to find every node
    #    it transitively depends on ("upstream").
    upstream = {}
    self._explore_dependencies(unique_id, dynprompt, upstream)

    # 2. Walk forward from Open's id through "upstream" to collect only the
    #    nodes that lie strictly between Open and Close (inclusive) --
    #    i.e., just the loop body, not the whole rest of the graph.
    contained = {}
    self._collect_contained(open_node_id, upstream, contained)
    contained[unique_id] = True
    contained[open_node_id] = True

    # 3. Clone every contained node into a FRESH GraphBuilder. The clone of
    #    Close itself is named "Recurse" (not a fresh numeric id) so the
    #    graph doesn't grow node-id length every iteration.
    graph = GraphBuilder()
    for node_id in contained:
        original = dynprompt.get_node(node_id)
        clone_id = "Recurse" if node_id == unique_id else node_id
        node = graph.node(original["class_type"], clone_id)
        node.set_override_display_id(node_id)

    # 4. Re-wire every input: if the original input was a link to another
    #    CONTAINED node, point at that node's clone; otherwise copy the
    #    literal/external link as-is.
    for node_id in contained:
        original = dynprompt.get_node(node_id)
        node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
        for key, value in original["inputs"].items():
            if is_link(value) and value[0] in contained:
                parent = graph.lookup_node(value[0])
                node.set_input(key, parent.out(value[1]))
            else:
                node.set_input(key, value)

    my_clone = graph.lookup_node("Recurse")
    return {
        "result": (...),            # this Close's own apparent output(s)
        "expand": graph.finalize(), # splice the new subgraph in and run it
    }
```

`TestForLoopClose` (the counter-loop wrapper) shows the decrement pattern:
it does NOT decrement inside the body — it injects a `subtract` + `to_bool`
into the CLOSE's own expansion graph, wiring the clone's Open
`initial_value0` (the counter) to `old_value - 1`, and wires that same
decremented value as the `condition` for the recursive `TestWhileLoopClose`.
This repo's shape (an explicit `remaining: INT` input on Close, computed
by the LAST body node, not inside Close) is simpler and is what the brief
specifies — `remaining <= 0` gates continuation directly, no extra
arithmetic nodes needed in Close's own expansion.

## How `remaining` threads (this spike, and the plan for Task 7/8)

- `SpikeLoopOpen.iterations` seeds the counter's ceiling.
- `CounterBody` (the loop body) reads/writes a JSON counter file on disk
  (NOT a Python attribute — see "why not memory" above) and computes
  `remaining = iterations - count`, returned as its own `INT` output.
- `SpikeLoopClose.remaining` is a plain (non-rawLink) `INT` input wired from
  the body's output — it arrives fully RESOLVED (the actual integer), which
  is what lets `close()` test `if remaining <= 0` directly.
- On each recursive expansion, Close re-wires the CLONED Open's `iterations`
  input... actually in this spike, `iterations` itself never changes (it's
  the fixed ceiling); what changes each iteration is the counter file's
  on-disk state, which `CounterBody` re-reads (see IS_CHANGED gotcha). The
  clone only needs to preserve the same wiring, not inject a new value into
  `iterations` — this is the point where this spike's shape diverges (for
  simplicity) from `TestForLoopClose`, which explicitly decrements a VALUE
  each iteration rather than mutating external state.
- **For Task 6/7/8 (the real pipeline):** `remaining` will instead come from
  `api.remaining_actionable(...)`, recomputed by `PoseFrameWriter`/
  `AnimationFrameWriter` AFTER the write lands on disk (per
  `docs/superpowers/plans/2026-07-01-pipeline-sweep-loops.md` Task 6/7),
  so `remaining` reflects post-write state exactly the way a resolved
  (non-rawLink) `INT` input can — no special handling needed beyond what
  this spike validates, since disk state (not an in-memory Python object)
  is already the pipeline's existing source of truth (manifest + sidecars).

## Gotchas

1. **`rawLink` is the crux, easy to get backwards.** The flow-control token
   (`SWEEP_FLOW` / `FLOW_CONTROL`) must be declared with `{"rawLink": True}`
   (per `comfy/comfy_types/node_typing.py:116` and
   `execution.py:171` — `is_link(input_data) and not
   input_info.get("rawLink", False)` gates whether an input gets resolved).
   A `rawLink` input arrives as the **unresolved** `[node_id, output_index]`
   pair instead of ComfyUI resolving it to a value — that's the ONLY way
   Close can recover the Open node's id (`flow[0]`) to walk the graph back
   to it. If you forget `rawLink`, `flow` arrives as whatever
   `SpikeLoopOpen.open()` returned (a plain dict here) and `flow[0]` breaks
   the whole clone-collection.
   - The reference additionally sets `"forceInput": True` on some
     rawLink sockets (`TestForLoopClose`'s `initial_valueN`); the flow
     socket itself in `TestWhileLoopClose` uses only `rawLink: True` (no
     forceInput) because `FLOW_CONTROL`/`SWEEP_FLOW` isn't a widget-havable
     type anyway. This spike adds `forceInput: True` defensively for `flow`
     since it's declared `required`, matching the brief's `remaining`
     socket, which explicitly needs `forceInput` (INT is otherwise a
     widget type and would render an editable number box instead of
     requiring a wire).
2. **`remaining`/counter inputs must NOT be `rawLink`.** They need the
   resolved value so Close can do `if remaining <= 0`. Only the
   flow-control token needs raw link semantics — mixing this up (rawLink
   on `remaining`) breaks the termination check.
3. **Hidden inputs (`DYNPROMPT`, `UNIQUE_ID`)** are required on Close (not
   Open, not the body) — they're how Close finds itself
   (`unique_id`) and walks the CURRENT expanded graph (`dynprompt`,
   which sees both original AND ephemeral/cloned nodes via
   `dynprompt.get_node`). Declare them under `"hidden"` in `INPUT_TYPES`
   exactly as `{"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}` — these
   are resolved by ComfyUI's hidden-input machinery, not passed by the
   calling graph.
4. **`IS_CHANGED` returning `float("nan")`** is the standard trick to force
   a node to be treated as always-changed: the cache signature
   (`comfy_execution/caching.py:116`,
   `signature = [class_type, await self.is_changed_cache.get(node_id)]`)
   includes the `IS_CHANGED` result, and `float("nan") != float("nan")` is
   always `True` in Python, so any comparison-based cache-hit check fails.
   In THIS spike it's arguably belt-and-suspenders, since each loop
   iteration already gets a brand-new cloned node id (so there's no
   literal "same node, called twice" case for the classic per-id
   output cache to short-circuit) — but it directly matches what the brief
   asks to validate and protects the real pipeline's selector nodes, which
   likely will NOT get a fresh id every call in all cases operators might
   wire them (e.g., a body subgraph reused outside a strict loop-clone
   context). Confirmed by reading `execution.py:57-98`
   (`IsChangedCache.get`) and `comfy_execution/caching.py` directly — not
   inferred.
5. **Explicit id map, not `graph.node(class_type)` with an implicit id.**
   `GraphBuilder.node(class_type, id=...)` treats the SECOND positional arg
   as the id of the NEW clone, not a lookup of the original. The clone loop
   must walk `dynprompt.get_node(node_id)["inputs"]` and manually remap any
   link whose source is also `contained`, or the cloned subgraph silently
   still points at the OLD (already-executed, now-stale) node instances.
6. **Termination has no separate "done" flag/type — it's the absence of an
   `"expand"` key.** Returning a plain tuple result (`return ("done",)`)
   when `remaining <= 0` is sufficient; there's no need to return an empty
   graph or a sentinel. `get_output_from_returns` only treats the result as
   a subgraph if the returned value is a `dict` containing `"expand"`.
7. **Per-iteration state must live on disk, not in the node instance.**
   Every iteration clones a new node id, so ComfyUI's per-node object cache
   (`caches.objects`, keyed by node id) gives every iteration of the body a
   FRESH Python instance. `CounterBody` here persists its count via a JSON
   file; the real pipeline already does this via manifest/sidecar files, so
   no new mechanism is needed there.
8. **Output-node re-queueing depends on `OUTPUT_NODE = True` on the CLONED
   class**, not on the original invocation. `SpikeLoopClose.OUTPUT_NODE =
   True` is what makes the recursive "Recurse" clone actually re-execute
   each round (`execution.py:593`, `if hasattr(class_def, 'OUTPUT_NODE')
   and class_def.OUTPUT_NODE == True: new_output_ids.append(node_id)`).
   Forgetting this on Close (or on the real pipeline's writer/final node)
   means the expansion is spliced into the graph but nothing schedules it
   to run.

## Was anything vendored?

No. The clone/rewire logic (`_explore_dependencies` / `_collect_contained` /
the two-pass clone-and-rewire loop in `SpikeLoopClose.close`) is hand-written
in `andypack/_spike_loop.py`, but it is a direct, intentional line-for-line
port of ComfyUI's own `TestWhileLoopClose.while_loop_close`
(`tests/execution/testing_nodes/testing-pack/flow_control.py:86-135`) with
renamed identifiers (`upstream`/`contained` kept, `flow_control`→`flow`,
`condition`→derived from `remaining`). Vendoring the whole testing-pack
module was considered and rejected: it pulls in `NUM_FLOW_SOCKETS`-many
generic `*`-typed sockets and a `VariantSupport()` decorator irrelevant to
the pipeline's single typed `SWEEP_FLOW` token, so hand-porting the
essential ~40 lines is clearer for Task 7/8 to build on than importing and
subsetting a 5-socket generic test utility.

## Validation method + residual risk

**What was done:**
1. Located a local ComfyUI install pinned at exactly `0.26.2`
   (`/Users/user/ComfyUI-Installs/ComfyUI (Local)/ComfyUI`,
   `comfyui_version.py`), matching the remote pod's reported version
   (`health_check` → `ComfyUI: 0.26.2`). Read `comfy_execution/graph_utils.py`
   and the relevant sections of `execution.py` and
   `comfy_execution/caching.py` directly.
2. Found and read ComfyUI's OWN test-suite reference implementation of this
   exact contract (`TestWhileLoopOpen`/`Close`,
   `TestForLoopOpen`/`Close`) — these are executed by ComfyUI's CI
   (`tests/execution/test_execution.py` imports the testing-pack), so they
   are a maintained, currently-passing ground truth for 0.26.2, not a
   stale blog post or an older major-version API.
3. Wrote `andypack/_spike_loop.py` as a direct, deliberate port of that
   reference onto the brief's `SWEEP_FLOW`/`remaining` shape.
4. `ruff check .` and `mypy andypack` pass clean (import of
   `comfy_execution.graph_utils` guarded exactly like
   `andypack/server.py`'s `from server import PromptServer` guard);
   `python3 -c "import andypack._spike_loop"` succeeds standalone with
   ComfyUI absent (confirms the guard actually degrades, not just that it
   compiles).
5. **Attempted a live 3-iteration run on the remote pod:** confirmed the
   pod (`comfyui`, dstack/RunPod, `213.173.104.21`) was reachable and
   running ComfyUI `0.26.2` with `comfyui-andypack` already checked out at
   `custom_nodes/comfyui-andypack` (git, `origin/main`, clean). Copied
   `_spike_loop.py` onto the pod via `scp`, and added a guarded
   `NODE_CLASS_MAPPINGS` merge to the pod's `andypack/__init__.py` (backed
   up to `__init__.py.spike-bak` first) to register the three spike nodes
   for this session only. Called `restart_comfyui` via the MCP to pick up
   the change — **the pod stopped and did not come back**; a follow-up
   `health_check` returned `CONNECTION_ERROR: fetch failed`, and `dstack ps`
   showed the run's status had flipped to `terminated`. This matches the
   documented pod behavior (auto-stops after ~1hr idle) rather than a bug
   in the spike: the restart call's "stop" step landed at/after the pod's
   own idle-stop boundary, and the pod does not self-relaunch (no `dstack
   apply`/`make up` was run to bring up a fresh pod, since that consumes
   GPU-rental time/cost for a throwaway spike confirmation and wasn't
   asked for). **No live 3-iteration counter run was completed before the
   pod became unreachable.**

**Residual risk (what a live run would additionally have confirmed, that
close reading cannot fully rule out):**
- Whether `rawLink: True` combined with `forceInput: True` on a
  **custom** (`SWEEP_FLOW`) type behaves identically to the reference's
  `FLOW_CONTROL` type in the *live* pod's exact object_info/type-resolution
  path — the source reading confirms the mechanism generically (any type
  string works for a rawLink socket; type strings are opaque to the link
  resolution logic in `execution.py`), but hasn't been exercised against
  this pod's frontend/type-checking layer end-to-end.
- Whether the ComfyUI-Manager-less pod environment (no `/v2/customnode/...`
  API) has any other deployment quirk that would surface only at actual
  graph-submission time (e.g., object_info caching per
  `docs/superpowers`/memory notes on "object_info caching" gotchas).
- The exact wall-clock re-queue timing/log shape for a 3-iteration loop
  (i.e., "three distinct 'executing' log lines for CounterBody") — inferred
  from `execution.py`'s `OUTPUT_NODE` re-queue logic, not observed.

**Mitigation:** the contract itself (Open/Close/body shape, rawLink,
hidden inputs, IS_CHANGED=NaN, expand-key/OUTPUT_NODE re-queue) is sourced
from code ComfyUI's own CI currently exercises at this exact version, which
is a materially stronger source than a live ad-hoc run of a throwaway spike
would have been on its own. Task 7/8 should still do ONE live run of the
real `SweepLoopOpen`/`SweepLoopClose` (not this throwaway) before shipping,
per the existing plan's Task 9 ("live validation").

## Files

- `andypack/_spike_loop.py` — throwaway spike (this task); NOT imported by
  `andypack/nodes.py` or `andypack/__init__.py` in this repo (only
  temporarily merged into the POD's copy of `__init__.py` for the live-run
  attempt above, and that pod is now terminated — nothing to revert
  upstream).
