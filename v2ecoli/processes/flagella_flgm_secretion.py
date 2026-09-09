"""Flagella FlgM secretion — the Class II -> Class III timing gate.

Ported to process-bigraph / v2ecoli from Maya Abdalla's vEcoli ``biofilm`` branch
(``ecoli/processes/flagella_flgm_secretion.py``). Biology preserved verbatim; only
the framework scaffolding is adapted to ``EcoliStep``.

Mechanism
---------
Once the hook-basal body is complete, the type-III secretion channel pumps
cytoplasmic FlgM (G369-MONOMER[c]) out of the cell. As FlgM drops, the
FLGM-FLIA-CPLX equilibrium shifts toward releasing free FliA (EG11355-MONOMER[c],
sigma-28), which activates Class III promoters (fliC, motAB, cheAW, ...) via
:mod:`v2ecoli.processes.flagella_transcription_regulation`. This is the Class
II -> Class III gate (Kalir et al. 2001): without it, FlgM accumulates
indefinitely and permanently sequesters FliA.

Trigger: ``count(nascent_flagellum)``, confirmed to BE the hook-basal-body-
complete stage exactly -- created the instant flagella_nfsim_complexation.py's
internal 'flagella' species forms, before any filament growth -- matching the
real substrate-specificity-switch timing (Hughes et al. 1993; Karlinsey et al.
2000). NOT ``CPLX0-7452`` (fully complete flagellum, filament included), which
can take multiple division cycles and made the gate almost never engage
(confirmed: "complete flagella" grew only 4->5->6 over 90 min while free FliA
ran unchecked 250->17,000). Two earlier trigger designs (CPLX0-7452 alone;
CPLX0-7452 + nascent_flagellum, reverted 2026-08-27 -- an uncited additive
assumption that also caused a division-crash regression) are kept as comments
in ``update()`` below per the standing preserve-old-code rule; full history in
MASTER_DOCUMENT.md's History Appendix.

Rate: first-order in the current FlgM pool (fixed 2026-09-02), gated on
hbb_count > 0 -- see config_schema's ``turnover_rate_per_s`` for the derivation.
Replaces an earlier zero-order, per-HBB placeholder (uncited, scaled export
with completed-flagella count, an assumption Karlinsey's data never supported),
kept as a comment below.

Refs: Hughes KT et al. (1993) Science 262:1277; Karlinsey JE, Tanaka S,
Bettenworth V, Yamaguchi S, Boos W, Aizawa SI & Hughes KT (2000) Mol Microbiol
37:1220-1231 ("Completion of the hook-basal body complex..." -- MASTER_DOCUMENT.md's
bibliography previously mis-cited this title/authors under 1998/J Bacteriol,
now corrected); Karlinsey JE, Tsui HC, Winkler ME & Hughes KT (1998) J Bacteriol
180:5384-5397 (FlgM turnover half-life, strain TH2592); Kalir S et al. (2001)
Science 292:2080.

Ordered in the composite flow:
    ecoli-complexation -> ecoli-flagella-flgm-secretion -> ecoli-transcript-initiation
"""


from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


NAME = "ecoli-flagella-flgm-secretion"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flgm_secretion"),
    "global_time": ("global_time",),
}


class FlagellaFlgMSecretion(Step):
    """Secrete cytoplasmic FlgM through completed hook-basal bodies, releasing
    free FliA. See module docstring for why the trigger is
    count(nascent_flagellum) alone -- confirmed to BE the hook-basal-body-
    complete stage, not an approximation of it."""

    description = (
        "FlagellaFlgMSecretion — type-III secretion of the anti-sigma FlgM.\n\n"
        "    hbb_count = count(nascent_flagellum)\n"
        "    exported = min(FlgM, round(FlgM * turnover_rate_per_s * timestep))"
        " if hbb_count > 0 else 0\n"
        "  First-order FlgM turnover, gated on hook-basal-body completion;\n"
        "  shifts the FlgM-FliA equilibrium to release sigma-28 and open Class III."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flgM_id": {"_type": "string", "_default": "G369-MONOMER[c]"},
        # Unused since 2026-08-27 (trigger is nascent_flagellum, see module
        # docstring) -- kept so the parameter isn't silently orphaned.
        "hbb_id": {"_type": "string", "_default": "CPLX0-7452[j]"},
        # Old zero-order placeholder (2026-08-27 - 2026-09-01), uncited, scaled
        # export with n_hbb -- an assumption Karlinsey's data never supported.
        # Superseded 2026-09-02, kept per standing preserve-old-code rule:
        # "secretion_rate": {"_type": "float", "_default": 0.1},
        # First-order turnover rate constant, k = ln(2) / t_half, t_half=7.3min
        # (438s): Karlinsey JE, Tsui HC, Winkler ME & Hughes KT (1998) J
        # Bacteriol 180:5384-5397, pulse-chase FlgM turnover in Fla+
        # (HBB-complete) strain TH2592. HBB-incomplete ring mutants (flgB,
        # DeltaflgHI) showed no detectable turnover -- captured by gating this
        # rate on hbb_count > 0 in update(), not by a further per-HBB
        # multiplier (see module docstring for why the old n_hbb scaling was
        # dropped, not just retuned).
        "turnover_rate_per_s": {"_type": "float", "_default": 0.0015823},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "nascent_flagellum": {"_type": NASCENT_FLAGELLUM_ARRAY, "_default": []},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        # self.secretion_rate = self.parameters["secretion_rate"]  # old zero-order rate, superseded 2026-09-02
        self.turnover_rate = self.parameters["turnover_rate_per_s"]
        # Bulk indices resolved lazily against the live bulk array.
        self.flgM_idx = None
        # hbb_idx no longer resolved/used as of 2026-08-27 (see config_schema
        # note on hbb_id) -- kept at None, not deleted, per standing
        # preserve-old-code rule.
        self.hbb_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flgM_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flgM_idx = bulk_name_to_idx(self.parameters["flgM_id"], bulk_ids)
            # self.hbb_idx = bulk_name_to_idx(self.parameters["hbb_id"], bulk_ids)
            # ^ no longer resolved as of 2026-08-27 -- CPLX0-7452 is not part
            # of the trigger anymore, see module docstring.

        # Trigger history (old code kept per standing preserve-old-code rule;
        # full account in MASTER_DOCUMENT.md's History Appendix): pre-2026-08-25
        # used CPLX0-7452 alone (wrong -- gates on filament completion, which
        # can take multiple division cycles); 2026-08-25 tried adding
        # CPLX0-7452 + nascent_flagellum, reverted 2026-08-27 (uncited additive
        # assumption, also caused a division-crash regression).
        # hbb_count = counts(states["bulk"], self.hbb_idx)
        # (nascent_lengths,) = attrs(states["nascent_flagellum"], ["filament_length"])
        # hbb_count = counts(states["bulk"], self.hbb_idx) + nascent_lengths.size

        # Current (2026-08-27): nascent_flagellum alone -- IS the hook-basal-
        # body-complete stage exactly (see module docstring).
        (nascent_lengths,) = attrs(states["nascent_flagellum"], ["filament_length"])
        hbb_count = nascent_lengths.size
        flgM_count = counts(states["bulk"], self.flgM_idx)

        exported = 0
        if hbb_count > 0 and flgM_count > 0:
            # Old zero-order formula, superseded 2026-09-02 (see config_schema):
            # exported = min(int(flgM_count), int(round(hbb_count * self.secretion_rate * states["timestep"])))
            # First-order: rate proportional to the current FlgM pool, gated
            # on hbb_count > 0 -- no further per-HBB multiplier (see module
            # docstring). Clamped to available FlgM so counts never go negative.
            exported = min(
                int(flgM_count),
                int(round(flgM_count * self.turnover_rate * states["timestep"])),
            )

        return {
            "bulk": [(self.flgM_idx, -exported)],
            "next_update_time": states["global_time"] + states["timestep"],
        }
