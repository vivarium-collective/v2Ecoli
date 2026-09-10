"""
=======================
Exchange-Flux Listener
=======================

Generic derived-emit Step: each tick it reads named keys out of the cell's
``environment.exchange`` store and writes them as plain scalar leaves under
``listeners.exchange_flux.<name>``.

WHY. The compact comparison view (and the XArray emitter view generally) only
carries ``listeners.*`` leaves, so a metabolic exchange flux — which lives in
``environment.exchange`` — never reaches the emitted store otherwise. This
listener bridges that gap by re-homing named fluxes onto listener leaves. It
also sidesteps v2ecoli#547 (dict-valued listener leaves drop on the injection
path): it READS the exchange dict (which populates) and WRITES plain scalars.

DELIBERATELY GENERIC. It knows nothing about any particular pathway: the caller
supplies a ``fluxes`` map ``{leaf_name: exchange_key}`` (e.g.
``{"acetate_exchange": "AC[p]", "glucose_exchange": "GLC[p]"}``). Enable
it via the ``ecoli_baseline`` generator's ``exchange_fluxes`` param. Sign is
preserved verbatim (uptake negative, secretion positive).

⚠ THE STORE IS CUMULATIVE, AND THE DEFAULT LEAF INHERITS THAT
--------------------------------------------------------------
``environment.exchange`` is a ``map[float]``, and a bare float leaf ACCUMULATES
(``state + update``) while metabolism writes a per-step molecule-count DELTA. So
the store holds a running total, not a rate — a property of the store's type
rather than of any writer, so it holds on the native and injected paths alike.

Measured over one generation: the glucose leaf grows 8.7x while dry mass grows
1.8x. A per-cell rate tracks mass; a running total does not.

That makes the default ``counts`` leaf the wrong input for anything that
time-averages, and time-averaging is exactly what a per-cell KPI table does:
the mean of a running total is not a rate, and it grows with how long the
generation ran. Hence ``basis``.

``basis="gdcw"`` emits a RATE
-----------------------------
Per tick: difference the running total, then normalise by time and dry mass —

    flux [mmol/gDCW/h] = d(counts) / (N_A * m_dry * dt)

which is the same quantity, in the same units and the same sign convention, as
a genuine vEcoli metabolism reports for its own exchanges. That matters because
it is what makes the two engines' flux leaves comparable at all; on the default
basis they are different quantities wearing the same name.

⚠ The first observation of a leaf emits 0.0 rather than its first difference.
Whether the store resets at division is NOT established here, and if it does not,
a first difference taken against an assumed zero would report an entire
generation's accumulation as one tick's rate — a spike at every division. Losing
one tick per lineage is the cheaper error, and it is the one that cannot be
mistaken for a measurement.
"""
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "exchange_flux_listener"
TOPOLOGY = {
    "exchange": ("environment", "exchange"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    # Read-only, and only consulted on the ``gdcw`` basis. Same source
    # ``metabolism`` uses for its own gDCW-basis reporting coefficient.
    "mass": ("listeners", "mass"),
}

#: Avogadro, mol^-1. Local rather than imported so the pure helper below stays
#: dependency-free and testable without the unit stack.
_N_AVOGADRO = 6.02214076e23

#: Recognised values of ``basis``.
BASIS_COUNTS = "counts"
BASIS_GDCW = "gdcw"


def _strip_compartment(mid: str) -> str:
    """``AC[p]`` -> ``AC``; unchanged if no ``[..]`` suffix."""
    return mid[:-3] if len(mid) > 3 and mid.endswith("]") and mid[-3] == "[" else mid


def resolve_exchange_key(exchange: dict, key: str):
    """Compartment-tolerant lookup: exchange stores may key by full metabolite id
    (``AC[p]``, the fork convention) or compartment-stripped (``AC``,
    v2ecoli's convention), and a genuine-vEcoli store can carry BOTH forms for the
    same molecule — a zero-valued compartment-tagged placeholder alongside the
    real flux on the stripped key (e.g. ``{"GLC[p]": 0, "GLC": -1.28e7}``). So an
    exact-first match would return the 0 placeholder and miss the real value.
    Instead gather every entry whose compartment-stripped form matches the
    request and return the one with the largest magnitude (placeholders are 0;
    the real flux is not). Returns None if no key matches."""
    exchange = exchange or {}
    stripped = _strip_compartment(key)
    matches = [v for k, v in exchange.items()
               if k == key or _strip_compartment(k) == stripped]
    if not matches:
        return None
    return max(matches, key=lambda v: abs(float(v)) if v is not None else 0.0)


def derive_fluxes(exchange: dict, fluxes: dict) -> dict:
    """Pure core: pull each configured exchange key into its leaf name. A key
    absent this tick yields 0.0 so the leaf stays a continuous trace."""
    exchange = exchange or {}
    out = {}
    for leaf, key in (fluxes or {}).items():
        v = resolve_exchange_key(exchange, key)
        out[leaf] = float(v) if v is not None else 0.0
    return out


def _as_float(value) -> float:
    """Coerce a DIMENSIONLESS reading (e.g. ``timestep``, in seconds) to a float,
    tolerating a pint Quantity by taking its magnitude as-is.

    Deliberately separate from ``_as_float_fg``: that one converts to femtograms,
    which is meaningless for a time. ``timestep`` is declared ``float`` on this
    step's ports, so a Quantity is not expected here — the tolerance exists so an
    unexpected type yields "no rate is defined" rather than taking a run down.
    """
    if value is None:
        return 0.0
    magnitude = getattr(value, "magnitude", None)
    if magnitude is not None:
        value = magnitude
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_float_fg(value) -> float:
    """Coerce a dry-mass reading to a plain float IN FEMTOGRAMS.

    ⚠ ``listeners.mass.dry_mass`` arrives as a pint Quantity in femtograms on the
    real composite, so a bare ``float()`` raises
    ``DimensionalityError: Cannot convert from 'femtogram' to 'dimensionless'``
    and takes the whole run down on the first tick of the gdcw basis.

    This path shipped unreachable — no study could set a basis until the setting
    was threaded — so it had never been executed against a real composite, only
    against unit tests that pass plain floats. Unreachable and untested are the
    same fact here: the first real run found it immediately.

    ⚠ CONVERTED, not read off as-is. ``counts_to_gdcw_rate`` expects femtograms,
    and this step's own port declares ``dry_mass`` a bare ``float`` while the
    composite supplies a ``quantity[float,fg]`` — so nothing in the type contract
    here guarantees the unit a Quantity arrives in. Taking ``.magnitude`` verbatim
    off a Quantity in picograms yields 0.4 where 400.0 is meant, and the rate is
    then 1000x too large with no error: a silently wrong QUANTITY under a correct
    name, which is the exact failure this basis exists to remove. Delegated to
    ``quantity_helpers.fg_magnitude``, the accessor the units-on-ports migration
    provides for precisely this (it is a pass-through for bare floats, so nothing
    changes while the port is still a plain number).

    A value that is neither a number nor a Quantity — or a Quantity whose units
    are not a mass — yields 0.0, which ``counts_to_gdcw_rate`` already treats as
    "no rate is defined" rather than an infinity.
    """
    if value is None:
        return 0.0
    try:
        from v2ecoli.library.quantity_helpers import fg_magnitude
        return float(fg_magnitude(value))
    except Exception:  # noqa: BLE001 — an uncoercible mass is "no rate", not a crash
        return _as_float(value)


def counts_to_gdcw_rate(delta_counts: float, dry_mass_fg: float,
                        timestep_s: float) -> float:
    """Molecule-count delta -> mmol/gDCW/h.

    The same conversion a genuine vEcoli metabolism applies to its own
    exchanges, so a leaf on this basis is comparable across engines rather than
    merely similarly named.

    Args:
        delta_counts: molecules exchanged this tick (signed; uptake negative).
        dry_mass_fg: cell dry mass in femtograms, as the mass listener reports it.
        timestep_s: tick duration in seconds.

    Returns:
        The rate, or ``0.0`` when it is not defined — a non-positive dry mass or
        timestep yields no rate rather than an infinity. ⚠ Returned as 0.0 and
        not NaN deliberately: these leaves are emitted as a continuous trace and
        a NaN would propagate through every downstream mean, turning one
        undefined tick into an undefined generation.
    """
    # `not (x > 0)`, NOT `x <= 0`: `nan <= 0` is False, so a NaN dry mass
    # (a freshly divided daughter whose mass listener seed missed) would
    # sail through the old guard and emit a NaN flux into the KPI trace —
    # exactly the undefined-tick-poisons-the-generation case the 0.0 return
    # exists to prevent.
    if not (dry_mass_fg > 0) or not (timestep_s > 0):
        return 0.0
    mmol = (delta_counts / _N_AVOGADRO) * 1e3
    grams = dry_mass_fg * 1e-15
    hours = timestep_s / 3600.0
    return mmol / grams / hours


class ExchangeFluxListener(Step):
    """Re-home named exchange fluxes onto ``listeners.exchange_flux.<name>``."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # {leaf_name: exchange_key} — the fluxes to surface.
        "fluxes": "map[string]",
        # "counts" (default) re-homes the store's running total verbatim;
        # "gdcw" differences it and normalises to mmol/gDCW/h. See the module
        # docstring — these are different quantities, not different units.
        "basis": "string{counts}",
        "time_step": "float{1.0}",
    }

    def initialize(self, config):
        self.fluxes = dict(self.parameters.get("fluxes") or {})
        self.basis = str(self.parameters.get("basis") or BASIS_COUNTS)
        if self.basis not in (BASIS_COUNTS, BASIS_GDCW):
            raise ValueError(
                f"{NAME}: unknown basis {self.basis!r}; expected "
                f"{BASIS_COUNTS!r} or {BASIS_GDCW!r}. Refused rather than "
                "defaulted: the two bases are different quantities, so a "
                "silently-defaulted basis would emit a running total under a "
                "name the caller meant as a rate.")
        # Previous running totals, per leaf. Absent = not yet observed, which
        # is NOT the same as zero — see the docstring on the first-tick 0.0.
        self._previous: dict[str, float] = {}

    def inputs(self):
        return {
            "exchange": {"_type": "map[float]", "_default": {}},
            "global_time": {"_type": "float", "_default": 0.0},
            "timestep": {"_type": "float", "_default": 1.0},
            "mass": {"dry_mass": {"_type": "float", "_default": 0.0}},
        }

    def outputs(self):
        # Recomputed-absolute each tick → 'set'/overwrite semantics. Leaves are
        # declared from the configured flux map (no fluxes → nothing declared).
        return {"listeners": {"exchange_flux": {
            leaf: {"_type": "overwrite[float]", "_default": 0.0}
            for leaf in self.fluxes}}}

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        if not self.fluxes:
            return {}
        totals = derive_fluxes(states.get("exchange"), self.fluxes)
        if self.basis == BASIS_COUNTS:
            return {"listeners": {"exchange_flux": totals}}

        dry_mass = _as_float_fg((states.get("mass") or {}).get("dry_mass"))
        timestep = _as_float(states.get("timestep"))
        out = {}
        for leaf, total in totals.items():
            previous = self._previous.get(leaf)
            # First observation: emit no rate rather than differencing against
            # an assumed zero. If the store carries across division that
            # assumption would report a whole generation's accumulation as one
            # tick, i.e. a spike at every division that looks like a result.
            out[leaf] = (0.0 if previous is None else
                         counts_to_gdcw_rate(total - previous, dry_mass, timestep))
            self._previous[leaf] = total
        return {"listeners": {"exchange_flux": out}}
