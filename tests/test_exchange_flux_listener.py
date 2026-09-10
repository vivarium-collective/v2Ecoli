"""Candidate-arm exchange-flux listener: re-homes named environment.exchange
fluxes onto listeners.exchange_flux.<name> leaves so the listeners-only compact
view carries them. Generic — the flux map is config; no pathway is special-cased.
"""
import pytest

from v2ecoli.steps.derivers.exchange_flux_listener import (
    ExchangeFluxListener, derive_fluxes, resolve_exchange_key)

FLUXES = {"acetate_exchange": "AC[p]", "glucose_exchange": "GLC[p]"}


def test_resolve_key_is_compartment_tolerant():
    # v2ecoli strips compartments (GLC); fork ids carry them (GLC[p]). One config
    # value must match either store convention.
    stripped_store = {"GLC": -8.5, "AC": 0.042}
    assert resolve_exchange_key(stripped_store, "GLC[p]") == -8.5
    assert resolve_exchange_key(stripped_store, "AC[p]") == 0.042
    full_store = {"GLC[p]": -8.5}
    assert resolve_exchange_key(full_store, "GLC") == -8.5      # reverse direction
    assert resolve_exchange_key(full_store, "GLC[p]") == -8.5   # exact
    assert resolve_exchange_key({"GLC": -8.5}, "MISSING[c]") is None


def test_zero_placeholder_does_not_shadow_the_real_flux():
    """A genuine-fork ``environment.exchange`` store carries BOTH forms of the
    same molecule, and only one of them is real.

    The fork's metabolism process declares the store's schema over the
    COMPARTMENT-TAGGED exchange ids, each with ``_default: 0``, but writes its
    per-tick exchange dmdt under the COMPARTMENT-STRIPPED id. So every exchange
    molecule ends up with a tagged key parked at the schema default alongside
    the stripped key carrying the value. An exact-match-first lookup returns the
    0 and reports "no exchange" for a molecule the cell is visibly exchanging.

    This is the store shape a resolver has to survive; the compartment-tolerance
    tests above never exercise it, because they never put both forms in one
    store.
    """
    both = {"AC[p]": 0, "AC": 0.042, "GLC[p]": 0, "GLC": -8.5}
    # secretion (positive) and uptake (negative) both reach past the placeholder
    assert resolve_exchange_key(both, "AC[p]") == 0.042
    assert resolve_exchange_key(both, "GLC[p]") == -8.5
    # and asking by the stripped id is unchanged
    assert resolve_exchange_key(both, "AC") == 0.042


def test_derive_reads_real_flux_from_a_placeholder_shadowed_store():
    """End-to-end through the public helper: the declared leaves carry the real
    values, not the placeholders. A study declares tagged ids (the fork
    convention), so this is the path every reference-arm run takes."""
    both = {"AC[p]": 0, "AC": 0.042, "GLC[p]": 0, "GLC": -8.5}
    assert derive_fluxes(both, FLUXES) == {
        "acetate_exchange": 0.042, "glucose_exchange": -8.5}


def test_a_genuine_zero_still_reads_zero():
    """The fix must not manufacture a value: a molecule that really is not
    being exchanged still resolves to 0.0, so a zero on the leaf keeps meaning
    'no flux' rather than 'lookup gave up'."""
    assert resolve_exchange_key({"AC[p]": 0, "AC": 0}, "AC[p]") == 0
    assert derive_fluxes({"AC[p]": 0, "AC": 0, "GLC": -8.5}, FLUXES) == {
        "acetate_exchange": 0.0, "glucose_exchange": -8.5}


def test_derive_matches_fork_ids_against_stripped_store():
    # study config uses fork ids; candidate store is stripped -> still matches
    out = derive_fluxes({"GLC": -8.5, "AC": 0.042}, FLUXES)
    assert out == {"acetate_exchange": 0.042, "glucose_exchange": -8.5}


def test_derive_selects_named_fluxes_preserving_sign():
    exch = {"AC[p]": 0.042, "GLC[p]": -8.5, "CO2[p]": 12.0}
    assert derive_fluxes(exch, FLUXES) == {
        "acetate_exchange": 0.042, "glucose_exchange": -8.5}


def test_derive_missing_key_is_zero_not_gap():
    assert derive_fluxes({"GLC[p]": -8.5}, FLUXES) == {
        "acetate_exchange": 0.0, "glucose_exchange": -8.5}


def test_derive_empty():
    assert derive_fluxes({}, {}) == {}
    assert derive_fluxes(None, {"x": "Y"}) == {"x": 0.0}


@pytest.mark.fast
def test_outputs_declared_from_config_flux_map():
    from v2ecoli.core import build_core
    core = build_core()
    step = ExchangeFluxListener({"fluxes": FLUXES}, core=core)
    out = step.outputs()
    assert set(out["listeners"]["exchange_flux"]) == set(FLUXES)
    # no fluxes configured -> no leaves declared (feature effectively off)
    step0 = ExchangeFluxListener({"fluxes": {}}, core=core)
    assert step0.outputs()["listeners"]["exchange_flux"] == {}


@pytest.mark.fast
def test_update_writes_listener_leaves():
    from v2ecoli.core import build_core
    core = build_core()
    step = ExchangeFluxListener({"fluxes": FLUXES}, core=core)
    upd = step.update({"exchange": {"AC[p]": 0.042, "GLC[p]": -8.5},
                       "global_time": 0.0, "timestep": 1.0})
    assert upd["listeners"]["exchange_flux"]["acetate_exchange"] == 0.042
    assert upd["listeners"]["exchange_flux"]["glucose_exchange"] == -8.5


@pytest.mark.fast
def test_feature_inserts_step_after_mass_listener():
    from v2ecoli.composites.ecoli_baseline import build_execution_layers
    flat = [s for L in build_execution_layers(["exchange_flux"]) for s in L]
    assert "exchange_flux_listener" in flat
    assert flat.index("exchange_flux_listener") > flat.index("ecoli-mass-listener")


# --------------------------------------------------------------------------
# The gDCW basis
#
# The store is a running total, so the default leaf is not a rate. Anything
# that time-averages it — which is what a per-cell KPI table does — averages a
# running total and gets a number that grows with generation length. These
# cover the conversion, and the two ways it could quietly go wrong.
# --------------------------------------------------------------------------

from v2ecoli.steps.derivers.exchange_flux_listener import (  # noqa: E402
    BASIS_COUNTS, BASIS_GDCW, counts_to_gdcw_rate)


def _listener(fluxes, basis=BASIS_COUNTS):
    """Build the Step directly, matching how the tests above construct it."""
    from v2ecoli.core import build_core
    return ExchangeFluxListener({"fluxes": fluxes, "basis": basis},
                                core=build_core())


@pytest.mark.fast
def test_the_conversion_lands_in_the_physiological_band():
    """The check that is not self-referential.

    A glucose uptake for E. coli on minimal medium is ~8-10 mmol/gDCW/h, and a
    genuine vEcoli reports 9.73 for this condition. Feeding this converter a
    real measured count trace has to land there — which tests the arithmetic
    against biology and against the other engine, rather than against itself.
    """
    # One generation of measured glucose uptake: counts, seconds, mean fg.
    counts, seconds, dry_mass_fg = 2.2617e9, 2701.0, (420.3 + 747.3) / 2
    per_tick = counts / seconds

    rate = counts_to_gdcw_rate(per_tick, dry_mass_fg, timestep_s=1.0)

    assert 8.0 <= rate <= 10.0, (
        f"{rate:.2f} mmol/gDCW/h is outside the physiological band for glucose "
        "uptake; the unit conversion is wrong")


@pytest.mark.fast
def test_a_rate_is_intensive_not_extensive():
    """Doubling the cell halves the per-gram rate at the same absolute uptake.

    This is what separates a rate from the running total it is derived from,
    and it is the property that makes the leaf comparable across engines and
    across cells of different sizes.
    """
    a = counts_to_gdcw_rate(1e6, dry_mass_fg=500.0, timestep_s=1.0)
    b = counts_to_gdcw_rate(1e6, dry_mass_fg=1000.0, timestep_s=1.0)
    assert b == pytest.approx(a / 2)


@pytest.mark.fast
def test_sign_is_preserved_so_uptake_stays_negative():
    """Uptake negative, secretion positive — the same convention as the counts
    basis and as a genuine vEcoli. A basis change must not flip it, since a
    flipped uptake reads as secretion and nothing downstream would object."""
    assert counts_to_gdcw_rate(-1e6, 500.0, 1.0) < 0
    assert counts_to_gdcw_rate(+1e6, 500.0, 1.0) > 0


@pytest.mark.fast
@pytest.mark.parametrize("dry_mass,timestep", [(0.0, 1.0), (500.0, 0.0),
                                               (-1.0, 1.0),
                                               # nan <= 0 is False: the old
                                               # `<= 0` guard passed a NaN
                                               # dry mass straight through
                                               # to a NaN flux in the trace.
                                               (float("nan"), 1.0),
                                               (500.0, float("nan"))])
def test_an_undefined_rate_is_zero_not_nan_or_infinite(dry_mass, timestep):
    """At division the mass listener can read zero. An infinity or NaN there
    propagates through every downstream mean, turning one undefined tick into
    an undefined generation — so the undefined case yields 0.0."""
    out = counts_to_gdcw_rate(1e6, dry_mass, timestep)
    assert out == 0.0


@pytest.mark.fast
def test_an_unknown_basis_is_refused_rather_than_defaulted():
    """The two bases are different QUANTITIES, not different units. Defaulting a
    misspelled basis would emit a running total under a name the caller meant as
    a rate, and nothing downstream could tell."""
    with pytest.raises(ValueError, match="unknown basis"):
        _listener({"glucose": "GLC[p]"}, basis="per-gram")


@pytest.mark.fast
def test_counts_basis_is_unchanged_by_the_addition():
    """Guard against over-correction. The default must still re-home the store's
    value verbatim — existing consumers depend on it, and a silent switch to
    rates would rescale every number they read."""
    step = _listener({"glucose": "GLC[p]"})
    out = step.update({"exchange": {"GLC[p]": -1234.0}, "global_time": 0.0,
                       "timestep": 1.0, "mass": {"dry_mass": 500.0}})
    assert out["listeners"]["exchange_flux"]["glucose"] == -1234.0


@pytest.mark.fast
def test_the_first_observation_emits_no_rate():
    """Differencing against an assumed zero would report a whole generation's
    accumulation as one tick if the store survives division — a spike at every
    division that looks like a result. One lost tick is the cheaper error."""
    step = _listener({"glucose": "GLC[p]"}, basis=BASIS_GDCW)
    first = step.update({"exchange": {"GLC[p]": -1e9}, "global_time": 0.0,
                         "timestep": 1.0, "mass": {"dry_mass": 500.0}})
    assert first["listeners"]["exchange_flux"]["glucose"] == 0.0


@pytest.mark.fast
def test_the_rate_is_the_difference_of_the_running_total():
    """The whole point: a constant per-tick uptake gives a CONSTANT rate, even
    though the underlying store keeps climbing."""
    step = _listener({"glucose": "GLC[p]"}, basis=BASIS_GDCW)
    common = {"global_time": 0.0, "timestep": 1.0, "mass": {"dry_mass": 500.0}}
    step.update({"exchange": {"GLC[p]": -1e6}, **common})          # priming
    second = step.update({"exchange": {"GLC[p]": -2e6}, **common})
    third = step.update({"exchange": {"GLC[p]": -3e6}, **common})

    expected = counts_to_gdcw_rate(-1e6, 500.0, 1.0)
    assert second["listeners"]["exchange_flux"]["glucose"] == pytest.approx(expected)
    assert third["listeners"]["exchange_flux"]["glucose"] == pytest.approx(expected)
    assert expected < 0


# --------------------------------------------------------------------------
# Surviving division
#
# baseline() takes the flux map through a module-level override that it CLEARS
# in its own finally. A daughter rebuilt mid-run therefore gets an empty map,
# declares no leaves, and reports 0.0 for the rest of the lineage — while the
# cell itself divides and grows normally. Measured on a two-generation run:
# generation 1 carried the data, generation 2 read exactly zero on every sample
# with dry mass, growth rate and division all normal.
#
# That is the shape that gets mistaken for a result, and an 8-generation screen
# would carry one generation of data and seven of zeros.
# --------------------------------------------------------------------------

@pytest.mark.fast
def test_division_config_carries_the_declared_flux_map():
    """The map must be captured while the override is still set — at
    daughter-build time it has already been restored, so reading it there gets
    nothing."""
    from v2ecoli.composites import _helpers

    saved = dict(_helpers._EXCHANGE_FLUXES_OVERRIDE)
    try:
        _helpers.set_exchange_fluxes_override({"glucose": "GLC[p]"})
        captured = dict(_helpers._EXCHANGE_FLUXES_OVERRIDE)
    finally:
        _helpers.set_exchange_fluxes_override(saved)

    assert captured == {"glucose": "GLC[p]"}
    # And the override really is cleared afterwards — the behaviour that makes
    # capture-at-build-time necessary rather than merely tidy.
    _helpers.set_exchange_fluxes_override({})
    assert _helpers._EXCHANGE_FLUXES_OVERRIDE == {}


@pytest.mark.fast
def test_the_division_step_passes_the_flux_map_to_each_daughter():
    """The regression itself: a daughter rebuilt without the map declares no
    leaves, so every exchange-flux reading after the first division is 0.0.

    Asserted at the baseline() call rather than by running two generations —
    a real two-generation run is ~25 minutes, and the defect is entirely in
    what this call is given.
    """
    import inspect

    from v2ecoli.steps import division as division_mod

    # 1. The step's own initialize() must LIFT the map off its parameters. Run
    #    the production initializer rather than assigning the attribute here —
    #    a test that sets the value it then asserts checks nothing.
    step = object.__new__(division_mod.Division)
    step.parameters = {"injected_processes": None,
                       "exchange_fluxes": {"glucose": "GLC[p]"}}
    division_mod.Division.initialize(step, step.parameters)

    assert getattr(step, "_exchange_fluxes", None) == {"glucose": "GLC[p]"}, (
        "Division.initialize dropped the declared flux map")

    # 2. And the daughter rebuild must actually HAND it to baseline(). Checked
    #    against the source of the rebuild, because the call happens deep inside
    #    a division event that a unit test cannot reach — and holding the map
    #    without passing it on would satisfy (1) while still producing a
    #    lineage of zeros.
    src = inspect.getsource(division_mod)
    call = src[src.index("doc = baseline("):]
    call = call[:call.index(")")]
    assert "exchange_fluxes=" in call, (
        "the daughter's baseline() rebuild does not pass exchange_fluxes, so "
        "daughters declare no flux leaves and report 0.0 for the rest of the "
        "lineage")


@pytest.mark.fast
def test_baseline_accepts_the_parameter_the_division_step_passes():
    """Guard against the two halves drifting apart. The threading is only real
    if ``baseline`` actually takes the keyword the division step hands it — a
    rename on either side would otherwise fail at division time, mid-run,
    rather than here."""
    import inspect
    from v2ecoli.composites.ecoli_baseline import baseline

    params = inspect.signature(baseline).parameters
    assert "exchange_fluxes" in params
    assert "injected_processes" in params
