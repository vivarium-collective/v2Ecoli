"""A numpy-typed variant param must be EXPANDED, never baked raw (#330).

The regression (#330, generalized here): a config declared an expression axis as
``logspace: {start: 6.07, stop: 6.07, num: 1}``. The correct value is
``np.logspace(start=6.07, ...) == 10**6.07 == 1_174_897.55``; the bug baked the
raw ``start`` (``6.07``) instead — a ~194,000x under-induction that ran green and
looked like a legitimate (tiny) expression level.

``parse_variant_params`` resolves any non-``value``/``nested`` param type through
``getattr(np, ptype)(**pvals).tolist()`` — the same path vEcoli's
``runscripts/create_variants.parse_variants`` uses. These tests pin that ALL of
the numpy generator types (``logspace``/``linspace``/``geomspace``/``arange``)
resolve through numpy rather than echoing a declared bound, so a future
numpy-typed param cannot silently regress the same way ``logspace`` did.
"""
import numpy as np
import pytest

from v2ecoli.workflow.variants import parse_variant_params


def _values(params):
    """The single override value from each expanded one-param arm, in order."""
    return [next(iter(p.values())) for p in params]


def test_logspace_start_is_expanded_not_baked_raw():
    """The exact #330 shape: a one-point logspace must yield 10**start, never
    the raw start bound."""
    params = parse_variant_params({
        "expr": {"target": "new-gene.exp", "logspace": {"start": 6.07,
                                                         "stop": 6.07, "num": 1}},
    })
    (value,) = _values(params)
    assert value == pytest.approx(10 ** 6.07)          # 1_174_897.55...
    assert value == pytest.approx(1_174_897.5549395303)
    # The bug: baking the declared bound. Guard against a >100x collapse back to
    # the raw start (any regression to "echo the bound" trips this).
    assert value != pytest.approx(6.07)
    assert value > 1e5


@pytest.mark.parametrize(
    "ptype,pvals",
    [
        ("logspace", {"start": 6.07, "stop": 8.0, "num": 4}),
        ("linspace", {"start": 0.0, "stop": 1.0, "num": 5}),
        ("geomspace", {"start": 1.0, "stop": 1000.0, "num": 4}),
        ("arange", {"start": 0.0, "stop": 2.0, "step": 0.5}),
    ],
)
def test_numpy_typed_param_matches_numpy_generator(ptype, pvals):
    """Every numpy generator type resolves via getattr(np, type)(**vals), the
    exact vEcoli parse_variants contract — not a raw/echoed declaration."""
    params = parse_variant_params({"p": {"target": "proc.k", **{ptype: pvals}}})
    expected = getattr(np, ptype)(**pvals).tolist()
    assert _values(params) == expected


@pytest.mark.parametrize(
    "ptype,pvals,declared_bounds",
    [
        ("logspace", {"start": 6.07, "stop": 8.0, "num": 4}, (6.07, 8.0)),
        ("geomspace", {"start": 2.0, "stop": 2000.0, "num": 5}, (2.0, 2000.0)),
    ],
)
def test_logspace_geomspace_do_not_echo_declared_bounds(ptype, pvals,
                                                        declared_bounds):
    """For log/geom axes the raw start/stop are NOT the expanded first/last
    values, so an arm that echoed a declared bound (the #330 failure mode) is
    detectable: log10-expanded endpoints differ from the raw bounds."""
    values = _values(parse_variant_params(
        {"p": {"target": "proc.k", **{ptype: pvals}}}))
    lo, hi = declared_bounds
    if ptype == "logspace":
        # first == 10**start, last == 10**stop — never the raw bounds.
        assert values[0] == pytest.approx(10 ** lo)
        assert values[-1] == pytest.approx(10 ** hi)
        assert values[0] != pytest.approx(lo)
        assert values[-1] != pytest.approx(hi)
    else:  # geomspace endpoints ARE the bounds, but interior points are geometric
        assert values[0] == pytest.approx(lo)
        assert values[-1] == pytest.approx(hi)
        # a linear (baked/echoed) reading would put the midpoint at the
        # arithmetic mean; geometric spacing puts it far below that.
        mid = values[len(values) // 2]
        assert mid < (lo + hi) / 2


def test_unknown_numpy_type_is_refused_not_silently_baked():
    """An unrecognized generator type must raise, never fall through to baking
    the declaration as-is (which is how a mis-typed axis would regress silent)."""
    with pytest.raises(ValueError, match="unknown value source"):
        parse_variant_params({"p": {"target": "proc.k",
                                    "logspce": {"start": 1, "stop": 2, "num": 2}}})
