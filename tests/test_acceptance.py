from andypack.resolve import status


def test_chain_unlocks_step_by_step(manifest, tree):
    root, char = tree.root, tree.char

    # Only concept present -> only base@E/SE/S selectable; nothing combat yet.
    tree.concept()
    assert status(manifest, root, char, "base", "E") == "ready"
    assert status(manifest, root, char, "fighting_stance", "E") == "blocked"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "blocked"

    # Generate base -> fighting_stance unlocks.
    tree.pose("base", "E")
    assert status(manifest, root, char, "fighting_stance", "E") == "ready"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "blocked"

    # Generate fighting_stance -> idle unlocks.
    tree.pose("fighting_stance", "E")
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "ready"
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "E") == "blocked"

    # Generate idle -> entry/exit/punch unlock.
    tree.animation("fighting_stance_idle", "E", frames=3)
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "E") == "ready"


def test_editing_base_prompt_makes_whole_subtree_stale_but_selectable(manifest, tree):
    root, char = tree.root, tree.char
    # Fully render the chain fresh.
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    assert status(manifest, root, char, "punch", "E") == "ready"

    # "Edit the base pose prompt": its stored hash no longer matches the manifest.
    tree.pose("base", "E", stale=True)

    # base + everything downstream show stale, but stay selectable (amber).
    assert status(manifest, root, char, "base", "E") == "stale"
    assert status(manifest, root, char, "fighting_stance", "E") == "stale"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "stale"
    assert status(manifest, root, char, "punch", "E") == "stale"
    from andypack.resolve import resolve_animation
    assert resolve_animation(manifest, root, char, "punch", "E")["selectable"] is True
