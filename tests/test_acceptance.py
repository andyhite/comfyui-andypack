from andypack.resolve import status


def test_chain_unlocks_step_by_step(manifest, tree):
    root, char = tree.root, tree.char

    # Only concept present -> only base@EAST/SOUTH_EAST/SOUTH selectable; nothing combat yet.
    tree.character()
    assert status(manifest, root, char, "base", "EAST") == "ready"
    assert status(manifest, root, char, "fighting_stance", "EAST") == "blocked"
    assert status(manifest, root, char, "fighting_stance_idle", "EAST") == "blocked"

    # Generate base -> fighting_stance unlocks.
    tree.pose("base", "EAST")
    assert status(manifest, root, char, "fighting_stance", "EAST") == "ready"
    assert status(manifest, root, char, "fighting_stance_idle", "EAST") == "blocked"

    # Generate fighting_stance -> idle unlocks.
    tree.pose("fighting_stance", "EAST")
    assert status(manifest, root, char, "fighting_stance_idle", "EAST") == "ready"
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "EAST") == "blocked"

    # Generate idle -> entry/exit/punch unlock.
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "EAST") == "ready"


def test_editing_base_prompt_makes_whole_subtree_stale_but_selectable(manifest, tree):
    root, char = tree.root, tree.char
    # Fully render the chain fresh.
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    assert status(manifest, root, char, "punch", "EAST") == "ready"

    # "Edit the base pose prompt": its stored hash no longer matches the manifest.
    tree.pose("base", "EAST", stale=True)

    # base + everything downstream show stale, but stay selectable (amber).
    assert status(manifest, root, char, "base", "EAST") == "stale"
    assert status(manifest, root, char, "fighting_stance", "EAST") == "stale"
    assert status(manifest, root, char, "fighting_stance_idle", "EAST") == "stale"
    assert status(manifest, root, char, "punch", "EAST") == "stale"
    from andypack.resolve import resolve_animation
    assert resolve_animation(manifest, root, char, "punch", "EAST")["selectable"] is True
