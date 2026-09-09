"""The coupled composite must declare ``environment`` as an emit path.

Regression test from a coupled bioreactor campaign: the declared-emit switch
replaced each runner's hand-curated allow-list with the composite's own
declaration, and
``environment`` was not in it -- so ``environment.exchange.*`` (the per-agent
exchange fluxes the coupled analyses read) stopped being persisted. The run
completed normally and the columns were simply absent, which is why nothing
failed at dispatch time.

``environment`` is AGENT-RELATIVE: the coupler reads
``agents.*.environment.exchange``. It therefore belongs in the declared paths
and must NOT be added to COUPLED_DOCUMENT_EMIT_ROOTS, which is for the stores
wired upward out of the agent frame (reactor / population / lineage).
"""
from v2ecoli.composites.reactor_bird_coupled import (
    COUPLED_DOCUMENT_EMIT_ROOTS,
    reactor_bird_coupled,
)


def _declared_paths():
    emitters = reactor_bird_coupled._composite_generator_entry.emitters
    assert len(emitters) == 1, f"expected one declared emitter, got {len(emitters)}"
    return emitters[0]["paths"]


def test_environment_is_declared():
    assert "environment" in _declared_paths()


def test_environment_is_agent_relative_not_a_document_root():
    # If someone "fixes" a missing environment column by adding it to the
    # document roots instead, it is resolved in the wrong frame and the
    # per-agent exchange leaves stay absent.
    assert "environment" not in COUPLED_DOCUMENT_EMIT_ROOTS


def test_the_previously_declared_paths_are_all_still_present():
    # The bug was a path silently dropping out of this list; guard the rest of
    # it the same way rather than only the one we noticed.
    declared = _declared_paths()
    for path in (
        "global_time", "bulk", "listeners", "boundary",
        "reactor", "population", "lineage",
    ):
        assert path in declared, f"{path} dropped from the declared emit paths"


def test_environment_binds_the_agent_store_not_the_document_one():
    """``declared_emit_set`` classifies a declared root by WHERE IT IS FOUND in
    the built state, not by ``COUPLED_DOCUMENT_EMIT_ROOTS``:

        root in agent_state -> agent leaf
        root in document    -> ROOT leaf
        neither             -> agent leaf (catch-all)

    This composite creates a top-level ``environment`` store unconditionally,
    so if an agent is built WITHOUT its own ``environment`` the declared root
    binds the document store instead -- the wrong one -- and the per-agent
    ``exchange`` leaves stay absent while the run still looks fixed.
    """
    from v2ecoli.library.parquet_run import declared_emit_set

    class _Fake:
        def __init__(self, state):
            self.state = state

    agent_with = _Fake({
        "environment": {"external_concentrations": {}},   # document-level
        "agents": {"0": {"bulk": [1], "environment": {"exchange": {}}}},
    })
    agent_leaves, root_leaves = declared_emit_set(agent_with, reactor_bird_coupled)
    assert ("environment",) in agent_leaves
    assert ("environment",) not in root_leaves

    # The hazard, stated as a test so it cannot be forgotten: no agent-side
    # environment store => the declared root binds the document one.
    agent_without = _Fake({
        "environment": {"external_concentrations": {}},
        "agents": {"0": {"bulk": [1]}},
    })
    _, root_leaves_bad = declared_emit_set(agent_without, reactor_bird_coupled)
    assert ("environment",) in root_leaves_bad
