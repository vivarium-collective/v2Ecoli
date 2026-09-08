"""build_lineage_ray_batch_document -- item 115: automatic per-generation
checkpointing and per-seed overrides (resume + variant-specific caching).

No prior test file exercised this function's own document-construction logic
directly (only the composite-registration wrapper, tests/composites/
test_lineage_ray_batch.py, and n_workers's own pool-sizing fix) -- these tests
cover it directly.
"""
from v2ecoli.workflow.batch_lineage_ray import build_lineage_ray_batch_document


def test_every_lineage_gets_a_real_checkpoint_dir_by_default():
    """Item 115: the resume seam must be usable with ZERO extra caller effort
    -- a plain n_seeds/n_generations call, no seed_overrides, still gets a
    real, working per-generation checkpoint destination for every lineage."""
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=3, experiment_id="exp1", out_dir="s3://bucket/out")
    cfg0 = doc["state"]["lineage_0000"]["config"]
    cfg1 = doc["state"]["lineage_0001"]["config"]
    assert cfg0["checkpoint_dir"] == "s3://bucket/out/checkpoints/exp1/seed_0000"
    assert cfg1["checkpoint_dir"] == "s3://bucket/out/checkpoints/exp1/seed_0001"
    # No resume/cache override requested -> every lineage starts fresh, exactly
    # today's behavior (byte-identical for the fields seed_overrides can touch).
    assert "initial_carry_state_path" not in cfg0
    assert "initial_generation_index" not in cfg0


def test_out_dir_trailing_slash_does_not_double_up():
    doc = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out/")
    cfg = doc["state"]["lineage_0000"]["config"]
    assert cfg["checkpoint_dir"] == "s3://bucket/out/checkpoints/exp1/seed_0000"


def test_seed_overrides_resumes_one_seed_from_a_specific_checkpoint():
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=5, base_seed=10, experiment_id="exp1",
        out_dir="s3://bucket/out",
        seed_overrides={
            10: {
                "initial_carry_state_path": "s3://bucket/out/checkpoints/exp1/seed_0010/gen_0003.pkl",
                "initial_generation_index": 4,
            },
        },
    )
    resumed = doc["state"]["lineage_0010"]["config"]
    fresh = doc["state"]["lineage_0011"]["config"]
    assert resumed["initial_carry_state_path"] == (
        "s3://bucket/out/checkpoints/exp1/seed_0010/gen_0003.pkl")
    assert resumed["initial_generation_index"] == 4
    # Only the named seed resumes -- the other lineage in the same batch is
    # completely unaffected.
    assert "initial_carry_state_path" not in fresh
    assert "initial_generation_index" not in fresh
    # Resuming does not disable the SAME lineage's own future checkpointing.
    assert resumed["checkpoint_dir"] == "s3://bucket/out/checkpoints/exp1/seed_0010"


def test_seed_overrides_accepts_string_keys_from_a_json_round_trip():
    """A caller building the request in Python has int seeds; one passing it
    through JSON (--params over HTTP) has string keys after decoding. Both
    must resolve to the same lineage -- neither caller shape may silently
    miss its own override."""
    doc = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=2, base_seed=7, experiment_id="exp1",
        out_dir="s3://bucket/out",
        seed_overrides={"7": {"initial_generation_index": 1,
                               "initial_carry_state_path": "s3://bucket/carry.pkl"}},
    )
    cfg = doc["state"]["lineage_0007"]["config"]
    assert cfg["initial_generation_index"] == 1
    assert cfg["initial_carry_state_path"] == "s3://bucket/carry.pkl"


def test_seed_overrides_gives_one_seed_a_variant_specific_cache():
    """Item 115's other real gap: a strain-specific ParCa cache for ONE seed,
    distinct from the batch-wide default -- what Run1/Run2 (K4/J3) need."""
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=1, cache_dir="s3://bucket/cache/baseline",
        experiment_id="exp1", out_dir="s3://bucket/out",
        seed_overrides={0: {"cache_dir": "s3://bucket/cache/k4-strain"}},
    )
    k4 = doc["state"]["lineage_0000"]["config"]
    baseline = doc["state"]["lineage_0001"]["config"]
    assert k4["cache_dir"] == "s3://bucket/cache/k4-strain"
    assert baseline["cache_dir"] == "s3://bucket/cache/baseline"


def test_seed_overrides_for_an_absent_seed_is_a_no_op():
    """An override dict that doesn't mention a given batch's seeds must not
    raise or otherwise affect it -- e.g. a caller reusing the same
    seed_overrides shape across differently-sized batches."""
    doc = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, base_seed=0, experiment_id="exp1",
        out_dir="s3://bucket/out",
        seed_overrides={99: {"cache_dir": "s3://bucket/cache/unrelated"}},
    )
    cfg = doc["state"]["lineage_0000"]["config"]
    assert "cache_dir" in cfg
    assert cfg["cache_dir"] != "s3://bucket/cache/unrelated"


def test_seed_overrides_none_is_byte_identical_to_omitted():
    doc_omitted = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out")
    doc_none = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out",
        seed_overrides=None)
    assert doc_omitted == doc_none


def test_exchange_fluxes_threaded_onto_every_lineage():
    """Item 106: a caller-supplied exchange-flux map must reach every lineage's
    own config, unmodified -- needed for a real product-exchange-flux column
    (e.g. CD2 Run 2's violacein_exchange) rather than only raw state."""
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out",
        exchange_fluxes={"violacein_exchange": "VIOLACEIN", "glucose_exchange": "GLC"},
        exchange_flux_basis="gdcw",
    )
    for node in ("lineage_0000", "lineage_0001"):
        cfg = doc["state"][node]["config"]
        assert cfg["exchange_fluxes"] == {"violacein_exchange": "VIOLACEIN", "glucose_exchange": "GLC"}
        assert cfg["exchange_flux_basis"] == "gdcw"


def test_exchange_fluxes_omitted_is_byte_identical_to_today():
    doc_omitted = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out")
    doc_none = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out",
        exchange_fluxes=None, exchange_flux_basis=None)
    assert doc_omitted == doc_none
    cfg = doc_omitted["state"]["lineage_0000"]["config"]
    assert "exchange_fluxes" not in cfg
    assert "exchange_flux_basis" not in cfg


def test_exchange_flux_basis_without_fluxes_is_a_no_op():
    """exchange_flux_basis alone (no exchange_fluxes map) has nothing to apply
    a basis to -- must not raise or add a dangling field."""
    doc = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, experiment_id="exp1", out_dir="s3://bucket/out",
        exchange_flux_basis="gdcw",
    )
    cfg = doc["state"]["lineage_0000"]["config"]
    assert "exchange_fluxes" not in cfg
    assert cfg["exchange_flux_basis"] == "gdcw"
