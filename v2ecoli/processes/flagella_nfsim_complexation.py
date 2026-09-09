"""Flagellar complexation via NFsim -- real bulk-coupled wrapper.

Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation,
NFSIM_WCM_WIRING_PLAN.md step 3.

Wraps ``pbg_nfsim.processes.NFSimProcess`` (scaffold-persistence-fixed,
vivarium-collective/viva-nfsim#2) so it can run as a real v2ecoli Step:
reads real WCM bulk counts each firing, feeds them to NFsim as observables,
runs NFsim's own chunked ``update()``, writes deltas back to the real bulk
store, and carries ``scaffold_species`` forward via its own persistent
state port -- exactly the mechanism proven working in
``diagnostic_real_bulk_seeding.py`` (step 2), now two-way-coupled instead
of read-only.

Two kinds of state persist across firings, both needed for correctness:

1. ``scaffold_species`` -- partial/Growing_X counter-state complexes (the
   scaffold-persistence fix's own port).
2. ``internal_observables`` -- the model's 3 species with NO real v2ecoli
   bulk-molecule correspondent (see generate_flagella_bngl.py's own
   docstring: 'flagellar export apparatus subunit', 'flagellar hook',
   'flagella'). These are "simple" (complete, not counter-state) NFsim
   species, but since they have no real bulk-store home, THIS Step must
   track their cumulative counts itself -- critically, 'flagellar hook' is
   CONSUMED downstream by the 'flagellum reaction', so an unconsumed hook
   completed in one firing must still be available to feed a later firing;
   without this port, every firing would tell NFsim "0 hooks exist yet"
   regardless of history, silently discarding real completed material.

Bridges NFsim's 'flagella' (hook-basal-body complete) completions into real
nascent_flagellum unique-molecule creation -- the same trigger point
flagella_filament_nucleation.py uses today. flagella_filament_elongation.py
is UNCHANGED and continues to consume whatever nascent_flagellum entries
exist, regardless of which mechanism created them.

Fires on a realistic cadence (NOT every tick) via the same
next_update_time-based rate-limiting pattern flagella_filament_nucleation.py
uses -- spawning a BioNetGen subprocess every 2s tick would be far too slow
wall-clock. Default interval matches the complexation_interval used
elsewhere in this investigation's standalone NFsim runs (1200s).

NOT yet wired into ecoli_baseline.py's flagella_regulation feature (that is
the LAST step in NFSIM_WCM_WIRING_PLAN.md's recommended rollout, behind its
own sub-flag so the existing custom-Steps pipeline stays selectable too) --
this file is tested standalone against a real composite first.
"""
import importlib.util
import os

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


_LOCAL_MODEL_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workspace", "investigations", "flagella-cascade", "studies",
    "flagella-04-complexation-nfsim", "models", "generate_flagella_bngl.py",
)


def _import_local_model():
    """Load v2ecoli's own generate_flagella_bngl.py by file path (it lives
    in an investigation folder, not an installed package) -- same pattern
    as flagella_nfsim_assembly.py's _import_local_model()."""
    spec = importlib.util.spec_from_file_location(
        "flagella_cascade_nfsim_model", _LOCAL_MODEL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_nfsim_process():
    # bionetgen 0.8.6 does `from pkg_resources import packaging` (removed on
    # py3.12); shim it from the standalone `packaging` before importing
    # pbg_nfsim -- same shim used throughout this investigation.
    import pkg_resources
    import packaging as _packaging
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = _packaging
    from pbg_nfsim.processes import NFSimProcess
    return NFSimProcess


NAME = "ecoli-flagella-nfsim-complexation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "scaffold_species": ("nfsim_scaffold_species",),
    "internal_observables": ("nfsim_internal_observables",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_nfsim_complexation"),
    "global_time": ("global_time",),
}

# The species in generate_flagella_bngl.py's model with no real v2ecoli
# bulk molecule ID (see that module's docstring) -- their cumulative counts
# have nowhere to live except this Step's own internal_observables port.
# Extended 2026-08-27 (HOOK DEPENDENCY FIX): added the two new in-between
# items on the way to a finished base -- rod alone, then rod with the
# P-ring -- same reasoning as the original 3 (each can complete in one
# firing without being consumed by the next stage in that same firing, so
# it must persist to the next).
_INTERNAL_ONLY_OBSERVABLES = (
    "flagellar_export_apparatus_subunit", "flagellar_hook", "flagella",
    "flagellar_rod", "flagellar_rod_with_p_ring",
)

# Added 2026-08-27 (Maya's request): C-ring, export apparatus, and motor
# complex are real bulk species, but their LIVE standing count is usually 0
# or 1 at any sampled instant -- each is a fast-flowing intermediate,
# produced by a rare nucleation event and then (once it happens) consumed by
# the very next reaction. Plotting the live count alone makes it look like
# "nothing is happening" even when real throughput exists. Track a
# cumulative "total ever formed" counter for each too (same pattern already
# used for the 3 no-real-bulk-ID species above), piggybacked onto the same
# internal_observables port under a distinct key so it doesn't collide with
# the real bulk count. Only counts POSITIVE deltas (new copies formed) --
# negative deltas (this species being consumed by the next stage) don't
# subtract from the running total, since the question this answers is "how
# many were ever built," not "how many exist right now" (that's still the
# real bulk count, unchanged).
_CUMULATIVE_TRACKED_REAL_IDS = (
    "CPLX0-7450[i]",          # C-ring
    "CPLX0-7451[j]",          # export apparatus
    "FLAGELLAR-MOTOR-COMPLEX[j]",  # motor complex
)


class FlagellaNFsimComplexation(Step):
    """Real-bulk-coupled NFsim complexation Step (motor-switch through
    hook-basal-body-complete), replacing the custom deterministic assembly
    Steps 1-4 per NFSIM_WCM_WIRING_PLAN.md."""

    description = (
        "FlagellaNFsimComplexation -- runs the NFsim rule-based reaction "
        "network against real WCM bulk counts.\n\n"
        "    Reads real bulk monomer/complex counts -> feeds NFsim -> writes "
        "deltas back to bulk.\n"
        "  Carries scaffold_species (partial assemblies) and "
        "internal_observables (the 3 species\n"
        "  with no real bulk-ID correspondent) forward across firings. "
        "'flagella' completions become\n"
        "  new nascent_flagellum unique molecules for "
        "flagella_filament_elongation.py to grow."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Matches this investigation's standalone NFsim runs
        # (run_nfsim_assembly.py's default --sample).
        "interval": {"_type": "float", "_default": 1200.0},
        # Raised 50 -> 500 (2026-09-01): controls NFsim's own reporting
        # resolution within each 1200s chunk (24s -> 2.4s), not how much it
        # actually simulates -- NFsim is a continuous-time event-driven
        # engine regardless of this value, so this is mostly extra I/O, not
        # extra compute. Tried to shrink the real blind-spot window in
        # gross_positive_by_name (see _CUMULATIVE_TRACKED_REAL_IDS above):
        # rod+P-ring, hook, and the export-apparatus-subunit intermediate
        # were still reading confirmed-false 0s even after the
        # gross-positive-delta fix, because their whole produce-then-
        # consume cycle can complete within a single 24s sub-step. Old
        # value kept per standing preserve-old-code rule:
        # "n_steps": {"_type": "integer", "_default": 50},
        "n_steps": {"_type": "integer", "_default": 500},
        # Threaded through to NFSimProcess (added 2026-09-01, pbg_nfsim
        # sibling package) -- NFsim used to self-seed on every call, so
        # runs sharing the same outer v2ecoli --seed still produced
        # different NFsim-driven timing. This "seed" key is picked up
        # automatically by ecoli_baseline.py's generic per-process seed
        # derivation (_derive_process_seed(master_seed, base_name)) the
        # same way every other stochastic process's config already is --
        # no extra plumbing needed on this side beyond declaring the key.
        "seed": {"_type": "integer", "_default": 0},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "nascent_flagellum": {"_type": NASCENT_FLAGELLUM_ARRAY, "_default": []},
            "scaffold_species": {"_type": "map[float]", "_default": {}},
            "internal_observables": {"_type": "map[float]", "_default": {}},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "nascent_flagellum": NASCENT_FLAGELLUM_ARRAY,
            "scaffold_species": "overwrite[map[float]]",
            "internal_observables": "overwrite[map[float]]",
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        model = _import_local_model()
        NFSimProcess = _import_nfsim_process()
        from process_bigraph import allocate_core

        self.interval = self.parameters["interval"]
        # real_bulk_id -> NFsim observable name (e.g.
        # 'FLIF-FLAGELLAR-MS-RING[i]' -> 'Free_FLIF_FLAGELLAR_MS_RING_i').
        self.id_to_obs = model.bulk_id_to_observable_name()
        self._real_ids = list(self.id_to_obs.keys())

        nfsim_core = allocate_core()
        self.nfsim = NFSimProcess(
            config={
                "model_file": model.get_model_path(),
                "n_steps": self.parameters["n_steps"],
                "seed": self.parameters["seed"],
            },
            core=nfsim_core,
        )

        self.reactant_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.reactant_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.reactant_idx = bulk_name_to_idx(self._real_ids, bulk_ids)

        update = {"next_update_time": states["global_time"] + self.interval}

        # Build NFsim's observables dict: real bulk counts for the 30
        # real-bulk-ID species, plus the 3 internal-only species carried
        # forward from the last firing (see module docstring).
        current_real_counts = counts(states["bulk"], self.reactant_idx)
        nfsim_observables = {name: 0.0 for name in self.nfsim.observable_names}
        for real_id, count in zip(self._real_ids, current_real_counts):
            nfsim_observables[self.id_to_obs[real_id]] = float(count)

        incoming_internal = dict(states.get("internal_observables") or {})
        for name in _INTERNAL_ONLY_OBSERVABLES:
            if name in nfsim_observables:
                nfsim_observables[name] = float(incoming_internal.get(name, 0.0))

        nfsim_state = {
            "observables": nfsim_observables,
            "scaffold_species": dict(states.get("scaffold_species") or {}),
        }
        result = self.nfsim.update(nfsim_state, self.interval)
        deltas_by_name = result["observables"]
        # Net delta over the whole chunk -- correct for real bulk counts,
        # but structurally blind to a species produced AND consumed within
        # the same chunk (e.g. C-ring formed then immediately consumed by
        # export-apparatus rxn 1). Use gross_positive_deltas (2026-09-01)
        # for anything that needs a true "how much was ever produced"
        # count instead -- see _CUMULATIVE_TRACKED_REAL_IDS below.
        gross_positive_by_name = result.get("gross_positive_deltas", {})

        # Real bulk deltas.
        bulk_deltas = np.zeros(len(self._real_ids), dtype=np.int64)
        for i, real_id in enumerate(self._real_ids):
            obs_name = self.id_to_obs[real_id]
            bulk_deltas[i] = int(round(deltas_by_name.get(obs_name, 0.0)))
        update["bulk"] = [(self.reactant_idx, bulk_deltas)]

        # Persist scaffold state (the fix this Step exists to exploit).
        update["scaffold_species"] = result["scaffold_species"]

        # Persist the 3 internal-only species' new cumulative counts.
        new_internal = {}
        for name in _INTERNAL_ONLY_OBSERVABLES:
            prev = incoming_internal.get(name, 0.0)
            delta = deltas_by_name.get(name, 0.0)
            new_internal[name] = prev + delta

        # Cumulative "total ever formed" for C-ring/export apparatus/motor
        # complex (2026-08-27, see _CUMULATIVE_TRACKED_REAL_IDS above).
        # FIXED 2026-09-01: was reading deltas_by_name (net delta over the
        # whole chunk), which is 0 whenever a species is produced AND
        # consumed within the same chunk -- confirmed real for these three
        # fast intermediate stages, which read flat 0 in every chart this
        # investigation has produced despite real completions happening.
        # Now uses gross_positive_deltas (sum of positive jumps across the
        # chunk's own ~50 NFsim sub-steps), which catches production even
        # when it's fully consumed later in the same chunk. Old line kept
        # per standing preserve-old-code rule:
        # delta = deltas_by_name.get(obs_name, 0.0)
        for real_id in _CUMULATIVE_TRACKED_REAL_IDS:
            key = f"{real_id}__cumulative"
            obs_name = self.id_to_obs[real_id]
            prev = incoming_internal.get(key, 0.0)
            delta = gross_positive_by_name.get(obs_name, 0.0)
            new_internal[key] = prev + max(0.0, delta)

        # Same fix, same reasoning, for the internal-only (no real bulk ID)
        # stages that also get consumed by the very next reaction within
        # the same chunk (added 2026-09-01): rod, rod+P-ring, export
        # apparatus subunit, hook. Deliberately does NOT touch
        # new_internal[name] above for these same names -- that value is
        # real, live, net-delta state fed back into NFsim's next firing as
        # a seed count (see nfsim_observables loop earlier in this
        # method); switching IT to gross-positive-only would make it only
        # ever increase, double-counting on every subsequent firing. This
        # is a separate, additional "ever formed" key, same pattern as
        # _CUMULATIVE_TRACKED_REAL_IDS above -- not a replacement.
        # 'flagella' excluded: already effectively monotonic (nothing in
        # the live model ever consumes it), so its existing net-delta
        # accumulation ("flagella_internal_cumulative" in reports) is
        # already correct.
        for name in ("flagellar_rod", "flagellar_rod_with_p_ring",
                     "flagellar_export_apparatus_subunit", "flagellar_hook"):
            key = f"{name}__cumulative"
            prev = incoming_internal.get(key, 0.0)
            delta = gross_positive_by_name.get(name, 0.0)
            new_internal[key] = prev + max(0.0, delta)

        update["internal_observables"] = new_internal

        # 'flagella' completions this firing -> new nascent_flagellum
        # unique molecules, same trigger point flagella_filament_
        # nucleation.py uses today. flagella_filament_elongation.py grows
        # them from here on, unchanged.
        n_new = int(round(deltas_by_name.get("flagella", 0.0)))
        if n_new > 0:
            update["nascent_flagellum"] = {
                "add": {"filament_length": np.zeros(n_new, dtype=np.int64)}
            }

        return update
