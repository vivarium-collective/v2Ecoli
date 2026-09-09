"""Voronoi plot of flagellar proteins by compartment label.

Ported from vivarium-collective/vEcoli-spatial-voronoi
(ecoli/analysis/single/flagella_compartment_voronoi.py), which itself uses the
Nocaj & Brandes (2012) Voronoi-treemap algorithm (wholecell/utils voronoi_plot
library, ported verbatim to v2ecoli/library/voronoi_treemap.py).

Sections all flagellar-assembly proteins by their EcoCyc compartment label
([c] cytosol, [i] inner membrane, [j] flagellar-projection / periplasm-facing,
[o] outer membrane, [m] membrane, [p] periplasm, [e] extracellular) and
overlays each compartment's TOTAL biomass (from the mass listener) as the
top-level Voronoi region, with each protein's own mass as a sub-polygon inside
its compartment — the compartment mass sets the region size, the proteins
inside it show how much of that mass this investigation's flagellar proteins
account for.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-03-voronoi-compartments/run_flagella_voronoi.py \
        --seconds 1200 --snapshot-at 600 --cache-dir out/cache_full
"""
import argparse
import os
import pickle

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.library.voronoi_treemap import VoronoiMaster, COMPARTMENT_COLOR_MAP
from wholecell.utils import units

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Flagellar assembly monomers, grouped by the compartment listener they fall
# under (same set + grouping as vEcoli-spatial-voronoi's
# flagella_compartment_voronoi.py, confirmed present in v2ecoli's monomer_data).
FLAGELLA_MONOMERS_BY_COMPARTMENT = {
    "extracellular": {
        "FlgK": "EG11967-MONOMER[e]",
        "FlgL": "EG11545-MONOMER[e]",
        "FliC": "EG10321-MONOMER[e]",
        "FliD": "EG10841-MONOMER[e]",
    },
    "periplasm": {
        "FliE": "EG11346-MONOMER[p]",
    },
    "cytosol": {
        "FliJ": "G378-MONOMER[c]",
        "FliI": "G377-MONOMER[c]",
        "FliH": "EG11656-MONOMER[c]",
        "FlgE": "G361-MONOMER[c]",
    },
    "outer_membrane": {
        "FLGG": "FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]",
    },
    "projection": {
        "FLGB": "FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGC": "FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGF": "FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGH": "FLGH-FLAGELLAR-L-RING[j]",
        "FLGI": "FLGI-FLAGELLAR-P-RING[j]",
        "FliQ": "EG11976-MONOMER[j]",
        "FliO": "EG11224-MONOMER[j]",
        "FliL": "EG10322-MONOMER[j]",
    },
    "membrane": {
        "FLIN": "FLIN-FLAGELLAR-C-RING-SWITCH[m]",
    },
    "inner_membrane": {
        "FLIF": "FLIF-FLAGELLAR-MS-RING[i]",
        "FLIG": "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]",
        "FLIM": "FLIM-FLAGELLAR-C-RING-SWITCH[i]",
        "FlhB": "G7028-MONOMER[i]",
        "FlhA": "G370-MONOMER[i]",
        "FliR": "EG11977-MONOMER[i]",
        "FliP": "EG11975-MONOMER[i]",
        "MotA": "MOTA-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]",
        "MotB": "MOTB-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]",
    },
}

# Compartment -> mass-listener field name (v2ecoli/v2ecoli/steps/derivers/mass_deriver.py).
COMPARTMENT_LISTENER_FIELDS = {
    "extracellular": "extracellular_mass",
    "periplasm": "periplasm_mass",
    "cytosol": "cytosol_mass",
    "outer_membrane": "outer_membrane_mass",
    "projection": "projection_mass",
    "membrane": "membrane_mass",
    "inner_membrane": "inner_membrane_mass",
    "flagellum": "flagellum_mass",
}

COMPARTMENT_COLORS = {
    "extracellular": [1.0, 0.506, 0.016],    # orange
    "periplasm": [0.463, 0.361, 0.620],      # purple
    "cytosol": [0.498, 0.725, 0.357],        # green
    "outer_membrane": [0.729, 0.459, 0.341], # brown
    "projection": [0.937, 0.616, 0.851],     # pink
    "membrane": [0.863, 0.255, 0.282],       # red
    "inner_membrane": [0.314, 0.655, 0.769], # light blue
    "flagellum": [0.20, 0.45, 0.85],         # dark blue
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _mfloat(x, default=0.0):
    """Coerce a listener value to a plain float, stripping pint units."""
    if hasattr(x, "magnitude"):
        try:
            return float(x.magnitude)
        except Exception:
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def run(seconds, snapshot_at, seed, cache_dir):
    with open(os.path.join(cache_dir, "simData.cPickle"), "rb") as f:
        sim_data = pickle.load(f)
    monomer_data = sim_data.process.translation.monomer_data
    monomer_mw = dict(zip(monomer_data["id"], monomer_data["mw"]))
    n_avogadro = sim_data.constants.n_avogadro

    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    cell = comp.state["agents"]["0"]
    bulk = _arr(cell["bulk"])
    bids = bulk["id"]
    monomer_idx = {
        gene: bulk_name_to_idx(mono_id, bids)
        for group in FLAGELLA_MONOMERS_BY_COMPARTMENT.values()
        for gene, mono_id in group.items()
    }
    mono_id_by_gene = {
        gene: mono_id
        for group in FLAGELLA_MONOMERS_BY_COMPARTMENT.values()
        for gene, mono_id in group.items()
    }

    def protein_mass_fg(gene):
        idx = monomer_idx[gene]
        count = int(_arr(cell["bulk"])["count"][idx])
        mw = monomer_mw[mono_id_by_gene[gene]]
        return (units.multiply(count, mw) / n_avogadro).asNumber(units.fg)

    def snapshot():
        mass = cell.get("listeners", {}).get("mass", {})
        compartments = {}
        for comp_name, genes in FLAGELLA_MONOMERS_BY_COMPARTMENT.items():
            total = _mfloat(mass.get(COMPARTMENT_LISTENER_FIELDS[comp_name], 0))
            sub = {gene: protein_mass_fg(gene) for gene in genes}
            sub = {k: v for k, v in sub.items() if v > 0}
            remaining = total - sum(sub.values())
            if remaining > 0:
                sub[comp_name] = remaining
            if sub:
                compartments[comp_name] = sub
        # flagellum compartment: no individually tracked sub-proteins here
        # (the complete-flagella complex itself, CPLX0-7452, lives in `unique`,
        # not `bulk`) -- show the whole compartment mass as one region.
        flagellum_total = _mfloat(mass.get(COMPARTMENT_LISTENER_FIELDS["flagellum"], 0))
        if flagellum_total > 0:
            compartments["flagellum"] = {"flagellum": flagellum_total}
        return compartments

    t0_snapshot = snapshot()

    total = 0.0
    snap_at_checkpoint = None
    while total < seconds:
        chunk = min(snapshot_at, seconds - total) if total < snapshot_at else min(60.0, seconds - total)
        comp.run(chunk)
        total += chunk
        if snap_at_checkpoint is None and total >= snapshot_at:
            snap_at_checkpoint = snapshot()

    final_snapshot = snapshot()
    return t0_snapshot, snap_at_checkpoint, final_snapshot


def figure(t0_dict, initial_dict, final_dict, snapshot_at, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COMPARTMENT_COLOR_MAP.clear()
    COMPARTMENT_COLOR_MAP.update(COMPARTMENT_COLORS)

    # The underlying Nocaj & Brandes treemap solve is a stochastic iterative
    # layout (random initial site placement) and occasionally fails to
    # converge for a given draw -- when it does, it can leave an internal
    # polygon as None and crash downstream in _compute_boundaries. Retrying
    # re-seeds the layout and reliably succeeds within a few attempts.
    last_exc = None
    for attempt in range(5):
        try:
            vm = VoronoiMaster()
            vm.plot(
                [[t0_dict, initial_dict, final_dict]],
                title=[[f"Flagellar proteins by compartment (t=0s)",
                        f"Flagellar proteins by compartment (t={snapshot_at:.0f}s)",
                        f"Flagellar proteins by compartment (t={seconds:.0f}s)"]],
                ax_shape=(1, 3),
                chained=False,
                font_size=3,
            )
            last_exc = None
            break
        except AttributeError as e:
            last_exc = e
            plt.close("all")
            print(f"  (voronoi layout attempt {attempt + 1} failed to converge, retrying: {e})")
    if last_exc is not None:
        raise last_exc

    out = f"{STUDY_DIR}/charts/01_flagella_protein_compartments.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close("all")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=1200)
    ap.add_argument("--snapshot-at", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full")
    args = ap.parse_args()

    t0_dict, initial_dict, final_dict = run(args.seconds, args.snapshot_at, args.seed, args.cache_dir)
    figure(t0_dict, initial_dict, final_dict, args.snapshot_at, args.seconds)

    print("\nCompartments at t=0s:")
    for comp_name, proteins in t0_dict.items():
        print(f"  {comp_name}: total={sum(proteins.values()):.2f} fg, n_proteins={len(proteins)}")
    print("\nCompartments at t=%ds:" % args.snapshot_at)
    for comp_name, proteins in initial_dict.items():
        print(f"  {comp_name}: total={sum(proteins.values()):.2f} fg, n_proteins={len(proteins)}")


if __name__ == "__main__":
    main()
