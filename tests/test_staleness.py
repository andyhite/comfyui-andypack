from andypack.resolve import outdated


def _full_stance_tree(tree):
    return (
        tree.concept()
        .pose("base", "EAST")
        .pose("fighting_stance", "EAST")
        .animation("fighting_stance_idle", "EAST", frames=3)
    )


def test_concept_never_outdated(manifest, tree):
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "concept", "EAST") is False


def test_incomplete_node_is_not_outdated(manifest, tree):
    # nothing rendered -> base is incomplete -> not "stale" (that's blocked territory)
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "base", "EAST") is False


def test_fresh_chain_is_not_outdated(manifest, tree):
    _full_stance_tree(tree)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance_idle", "EAST") is False


def test_own_hash_drift_marks_outdated(manifest, tree):
    _full_stance_tree(tree)
    # re-render fighting_stance with a bogus stored hash
    tree.pose("fighting_stance", "EAST", stale=True)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True


def test_staleness_is_transitive(manifest, tree):
    # base rendered with a stale hash; idle/punch are otherwise fresh.
    # `outdated` is the staleness predicate for a COMPLETE node (spec §6), so
    # punch must be rendered for its transitive staleness to be observable here
    # (an unrendered punch is the `blocked`/`ready` axis, never `outdated`).
    tree.concept().pose("base", "EAST", stale=True).pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    ).animation("punch", "EAST", frames=3)
    # fighting_stance's own hash is fine, but its ancestor (base) is outdated
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True
    # ripples all the way to punch (start_from idle -> fighting_stance -> base)
    assert outdated(manifest, tree.root, tree.char, "punch", "EAST") is True
