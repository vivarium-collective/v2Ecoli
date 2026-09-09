"""Flagella transcription regulation — the Kalir & Alon (Cell 2004) SUM-gate.

Ported to process-bigraph / v2ecoli from Maya Abdalla's vEcoli ``biofilm`` branch
(``ecoli/processes/flagella_transcription_regulation.py``). The biology and the
gate math are preserved verbatim; only the framework scaffolding (vivarium-core
``Step`` -> ``EcoliStep`` with ``inputs``/``outputs``/``update``) is adapted.

Mechanism
---------
Each timestep the gate computes two normalized activity signals via
Michaelis-Menten saturation of the master regulators:

    X = [FlhDC] / (K_flhDC + [FlhDC])      (CPLX0-3930[c], the class-II activator)
    Y = [FliA]  / (K_fliA  + [FliA])       (EG11355-MONOMER[c], free sigma-28)

and writes a per-promoter ``init_prob_override`` onto the promoters unique
molecule so that :mod:`v2ecoli.processes.transcript_initiation` uses the K&A
value directly instead of the default ``basal_prob + delta_prob * bound_TF``.

* **Class II** (7 transcription units): the bilinear SUM gate
  ``p_i = (beta*X + beta'*Y) / (beta + beta')`` normalized by ``p_i_ref`` (its
  value at the t=0 reference state, ``X=X_ref, Y=0``) and scaled by the gene's
  ParCa ``basal_prob`` so the override equals ``basal_prob`` exactly at reference
  conditions and rises as free FliA accumulates.
* **Class III** (fliC, fliD, flgK/L, motAB, cheAW, flgM): ``override = Y * basal_prob``,
  rising from ~0 when FliA is sequestered by FlgM to ``basal_prob`` at full FliA
  activity. The release of FliA is driven downstream by
  :mod:`v2ecoli.processes.flagella_flgm_secretion`. Note: fliS (EG11388) is
  NOT listed here separately, but is NOT unregulated either -- it shares
  fliD's exact transcription unit (fliD-fliS-fliT is one real operon,
  TU0-14278) so it already rides on fliD's Class III entry above. See
  get_flagella_transcription_regulation_config's 2026-08-21 note for the
  investigation that confirmed this (and ruled out translation efficiency
  as a fliD/fliS disparity too).

Writing ``init_prob_override`` directly (rather than touching ``bound_TF``)
bypasses the TF-binding pathway for flagella genes, eliminating the
double-counting that otherwise drives FliA ~5x above its calibrated level.

EcoCyc IDs: CPLX0-3930 (FlhDC), EG11355-MONOMER (FliA / sigma-28).
Class II TUs (genes, verified against get_flagella_transcription_regulation_
config's classII_cistron_ids -- an earlier version of this comment wrongly
listed flhD as EG10322, which is actually fliL): fliL EG10322, fliE EG11346,
fliF EG11347, flgB G358, flgA G357, flhB G7028, fliA EG11355. flhD (EG10320)
and flhC (EG10319) are NOT Class II genes in this gate -- FlhD4C2 (CPLX0-3930)
is the gate's INPUT (X), assembled via the standard, unmodified
CPLX0-3930_RXN complexation reaction (4 FlhD + 2 FlhC), and its own
transcription runs on plain ParCa basal probability, untouched by this Step.

Ordered in the composite flow:
    ecoli-tf-binding -> ecoli-flagella-transcription-regulation -> ecoli-transcript-initiation
"""

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import PROMOTER_ARRAY


NAME = "ecoli-flagella-transcription-regulation"
TOPOLOGY = {
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_transcription_regulation"),
    "global_time": ("global_time",),
}


class FlagellaTranscriptionRegulation(Step):
    """Kalir & Alon (Cell 2004) bilinear SUM-gate for flagella transcription."""

    description = (
        "FlagellaTranscriptionRegulation — Kalir & Alon (Cell 2004) SUM-gate.\n\n"
        "    X = [FlhDC] / (K_flhDC + [FlhDC]);  Y = [FliA] / (K_fliA + [FliA])\n"
        "  Class II:  p_i = (beta*X + beta'*Y)/(beta+beta'),  override = p_i/p_i_ref * basal_prob\n"
        "  Class III: override = Y * basal_prob\n"
        "  Writes init_prob_override onto promoters; transcript_initiation substitutes it where > 0."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # flhDC activation coefficients (one per Class II TU)
        "beta": {"_type": "list[float]", "_default": [1200, 450, 350, 350, 150, 100, 50]},
        # FliA activation coefficients (one per Class II TU)
        "beta_prime": {"_type": "list[float]", "_default": [250, 350, 300, 450, 300, 350, 300]},
        # Class II RNA IDs. Defaults are bare cistron IDs (used in unit tests);
        # sim_data wiring overrides with the TU-level [c] IDs present in rna_ids.
        "flg_classII_rnaids": {
            "_type": "list[string]",
            "_default": [
                "EG10322_RNA", "EG11346_RNA", "EG11347_RNA",
                "G358_RNA", "G357_RNA", "G7028_RNA", "EG11355_RNA",
            ],
        },
        # Class III RNA IDs. Empty default is safe for tests (loop skipped);
        # sim_data wiring populates with resolved TU IDs.
        "flg_classIII_rnaids": {"_type": "list[string]", "_default": []},
        "fliA": {"_type": "string", "_default": "EG11355-MONOMER[c]"},
        "flhDC": {"_type": "string", "_default": "CPLX0-3930[c]"},
        # Full ordered list of TU RNA IDs (matches transcript_initiation rna_data).
        "rna_ids": {"_type": "list[string]", "_default": []},
        "K_flhDC": {"_type": "float", "_default": 10.0},
        "K_fliA": {"_type": "float", "_default": 10.0},
        # basal_prob indexed by TU_index (same ordering as rna_ids / rna_data).
        # Populated by sim_data at wiring; empty -> fall back to 1.0 scaling.
        "basal_prob": {"_type": "list[float]", "_default": []},
        "seed": {"_type": "integer", "_default": 0},
    }

    def inputs(self):
        return {
            "promoters": {"_type": PROMOTER_ARRAY, "_default": []},
            "bulk": {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "promoters": PROMOTER_ARRAY,
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        self.beta = np.asarray(self.parameters["beta"], dtype=float)
        self.beta_prime = np.asarray(self.parameters["beta_prime"], dtype=float)

        rna_ids = list(self.parameters["rna_ids"])
        self.flg_TU_ids = np.array(
            [rna_ids.index(rna_id) for rna_id in self.parameters["flg_classII_rnaids"]]
        )
        self.flg_classIII_TU_ids = np.array(
            [rna_ids.index(rna_id) for rna_id in self.parameters["flg_classIII_rnaids"]],
            dtype=int,
        )

        # Per-gene basal_prob anchors the override to vEcoli's normalization scale.
        # After X_ref normalization the effective formula is (p_i / p_i_ref) * basal_prob,
        # so at reference conditions (Y=0, X=X_ref) the override equals basal_prob exactly.
        basal_prob = self.parameters["basal_prob"]
        if len(basal_prob) > 0:
            basal_prob = np.asarray(basal_prob, dtype=float)
            self.flg_classII_basal_probs = np.array(
                [basal_prob[i] for i in self.flg_TU_ids]
            )
            self.flg_classIII_basal_probs = np.array(
                [basal_prob[i] for i in self.flg_classIII_TU_ids]
            )
        else:
            self.flg_classII_basal_probs = np.ones(len(self.flg_TU_ids))
            self.flg_classIII_basal_probs = np.ones(len(self.flg_classIII_TU_ids))

        # Bulk indices resolved lazily on the first update against the live bulk
        # array ordering (matches the tf_binding idiom).
        self.flhDC_idx = None
        self.fliA_idx = None

        # X_ref is FlhDC activity at t=0 (the ParCa reference). Computed lazily on
        # the first update so we don't have to dig the initial FlhDC count out of
        # sim_data. Used to normalize p_i so override == basal_prob at reference.
        self.X_ref = None
        self.p_i_ref = None

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        # Resolve bulk indices against the live bulk array on first run.
        if self.flhDC_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flhDC_idx = bulk_name_to_idx(self.parameters["flhDC"], bulk_ids)
            self.fliA_idx = bulk_name_to_idx(self.parameters["fliA"], bulk_ids)

        # No promoters yet — nothing to write; just advance the clock.
        if states["promoters"]["_entryState"].sum() == 0:
            return {
                "promoters": {},
                "next_update_time": states["global_time"] + states["timestep"],
            }

        TU_index, init_prob_override = attrs(
            states["promoters"], ["TU_index", "init_prob_override"]
        )

        flhDC_count = counts(states["bulk"], self.flhDC_idx)
        fliA_count = counts(states["bulk"], self.fliA_idx)

        X = flhDC_count / (self.parameters["K_flhDC"] + flhDC_count)
        Y = fliA_count / (self.parameters["K_fliA"] + fliA_count)

        # Capture X at t=0 as the reference. At reference (X=X_ref, Y=0),
        # p_i/p_i_ref == 1 so init_prob_override == basal_prob, matching ParCa.
        if self.X_ref is None:
            self.X_ref = X
            self.p_i_ref = self.beta * self.X_ref / (self.beta + self.beta_prime)

        # K&A SUM gate: p_i in [0,1] is the normalized expression level per Class II gene.
        p_i = (self.beta * X + self.beta_prime * Y) / (self.beta + self.beta_prime)

        # Guard p_i_ref=0 (only if FlhDC=0 at t=0): treat as fully basal-driven.
        safe_p_i_ref = np.where(self.p_i_ref > 0, self.p_i_ref, 1.0)

        init_prob_override_new = init_prob_override.copy()
        for i, tu_idx in enumerate(self.flg_TU_ids):
            rows = np.where(TU_index == tu_idx)[0]
            if len(rows) == 0:
                continue
            init_prob_override_new[rows] = (
                p_i[i] / safe_p_i_ref[i] * self.flg_classII_basal_probs[i]
            )

        # Class III driven by Y only: 0 when FliA sequestered, basal_prob at full activity.
        for j, tu_idx in enumerate(self.flg_classIII_TU_ids):
            rows = np.where(TU_index == tu_idx)[0]
            if len(rows) == 0:
                continue
            init_prob_override_new[rows] = Y * self.flg_classIII_basal_probs[j]

        return {
            "promoters": {"set": {"init_prob_override": init_prob_override_new}},
            "next_update_time": states["global_time"] + states["timestep"],
        }
