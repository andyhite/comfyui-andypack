from andypack.resolve import outdated


def _full_stance_tree(tree):
    return (
        tree.concept()
        .pose("base", "E")
        .pose("fighting_stance", "E")
        .animation("fighting_stance_idle", "E", frames=3)
    )


def test_concept_never_outdated(manifest, tree):
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "concept", "E") is False


def test_incomplete_node_is_not_outdated(manifest, tree):
    # nothing rendered -> base is incomplete -> not "stale" (that's blocked territory)
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "base", "E") is False


def test_fresh_chain_is_not_outdated(manifest, tree):
    _full_stance_tree(tree)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance_idle", "E") is False


def test_own_hash_drift_marks_outdated(manifest, tree):
    _full_stance_tree(tree)
    # re-render fighting_stance with a bogus stored hash
    tree.pose("fighting_stance", "E", stale=True)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "E") is True


def test_staleness_is_transitive(manifest, tree):
    # base rendered with a stale hash; idle/punch are otherwise fresh.
    # `outdated` is the staleness predicate for a COMPLETE node (spec §6), so
    # punch must be rendered for its transitive staleness to be observable here
    # (an unrendered punch is the `blocked`/`ready` axis, never `outdated`).
    tree.concept().pose("base", "E", stale=True).pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    ).animation("punch", "E", frames=3)
    # fighting_stance's own hash is fine, but its ancestor (base) is outdated
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "E") is True
    # ripples all the way to punch (start_from idle -> fighting_stance -> base)
    assert outdated(manifest, tree.root, tree.char, "punch", "E") is True
