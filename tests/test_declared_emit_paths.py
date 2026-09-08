"""Config-declared EXTRA emit store paths — the general, domain-agnostic
capability that lets a composite/config persist stores beyond the baseline
parquet set (global_time / bulk / listeners) by declaring ``emit_paths`` on the
emitter config. Framework-level: no subsystem-specific schema is baked in; the
declaring config owns which store paths matter.
"""
from v2ecoli.composites._helpers import _merge_emit_paths


def test_merge_single_two_segment_path():
    emit_schema = {"global_time": "float"}
    topo = {"global_time": ("global_time",)}
    _merge_emit_paths(emit_schema, topo, [["some_store", "sub_key"]])
    assert emit_schema["some_store"] == {"sub_key": "node"}
    assert topo["some_store"] == ("some_store",)


def test_merge_deep_path():
    emit_schema, topo = {}, {}
    _merge_emit_paths(emit_schema, topo, [["compartment", "global", "volume"]])
    assert emit_schema == {"compartment": {"global": {"volume": "node"}}}
    assert topo == {"compartment": ("compartment",)}


def test_paths_sharing_a_prefix_merge():
    emit_schema, topo = {}, {}
    _merge_emit_paths(emit_schema, topo, [
        ["compartment", "global", "volume"],
        ["compartment", "global", "area"],
    ])
    assert emit_schema == {
        "compartment": {"global": {"volume": "node", "area": "node"}}}
    assert topo == {"compartment": ("compartment",)}


def test_multiple_independent_paths():
    emit_schema, topo = {}, {}
    _merge_emit_paths(emit_schema, topo, [["store_a", "x"], ["store_b", "y"]])
    assert emit_schema == {"store_a": {"x": "node"}, "store_b": {"y": "node"}}
    assert set(topo) == {"store_a", "store_b"}


def test_none_and_empty_are_noops():
    emit_schema, topo = {"listeners": {}}, {"listeners": ("listeners",)}
    _merge_emit_paths(emit_schema, topo, None)
    _merge_emit_paths(emit_schema, topo, [])
    _merge_emit_paths(emit_schema, topo, [[]])  # empty path skipped
    assert emit_schema == {"listeners": {}}
    assert topo == {"listeners": ("listeners",)}


def test_does_not_clobber_an_existing_dict_leaf():
    emit_schema = {"store_a": {"x": "node"}}
    topo = {}
    _merge_emit_paths(emit_schema, topo, [["store_a"]])  # single-seg over a dict
    # the existing nested dict is preserved, not overwritten with "node"
    assert emit_schema["store_a"] == {"x": "node"}
