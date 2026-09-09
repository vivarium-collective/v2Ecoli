"""Maya's Aim 2B — rule-based flagellar assembly via NFsim (pbg-nfsim engine,
v2ecoli-owned model).

Runs pbg-nfsim (https://github.com/vivarium-collective/pbg-nfsim, a process-bigraph
wrapper for BioNetGen/NFsim) against THIS investigation's own BNGL model --
models/generate_flagella_bngl.py, models/flagella_complexation.bngl -- not
pbg-nfsim's bundled example (moved here 2026-08-12 so the model can evolve
alongside v2ecoli's own flagella reaction network and eventually couple to it
directly; see flagella_nfsim_assembly.py's module docstring). The BNGL model
encodes hierarchical complexation of ~30 flagellar proteins through 7 sequential
reactions (588 rules, real cryo-EM-cited stoichiometry -- see the model's own
docstring for full provenance): free monomers -> export apparatus -> motor/basal
body -> hook -> hook-basal-body complete. This is the ordered, conditional
assembly that the stochastic Gillespie complexation cannot capture (Aim 2B
rationale).

NOTE (2026-08-12): FliC/filament elongation is NOT modeled here -- excluded from
the BNGL rule network entirely (see the model's FLIC REMOVAL docstring note) to
avoid the same combinatorial/file-size explosion v2ecoli itself hit and already
solved by moving filament growth to an incremental process outside the
combinatorial engine (flagella_filament_elongation.py). The "flagella" observable
below tracks assembly complete through the HOOK-BASAL-BODY stage, not a
filament-bearing organelle -- label reflects this.

This driver runs the composed production+complexation workflow (MonomerProduction
feeds monomers; NFSimProcess assembles them) and renders the staged appearance of
the assembly intermediates.

Note: bionetgen 0.8.6 imports the removed `pkg_resources.packaging` on Python 3.12;
we shim it from the standalone `packaging` package before importing pbg_nfsim.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/run_nfsim_assembly.py \
        --seconds 1800 --sample 100
"""
import argparse
import os

# --- shim: bionetgen 0.8.6 does `from pkg_resources import packaging` (removed) ---
import pkg_resources
import packaging as _packaging
if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = _packaging
# ----------------------------------------------------------------------------------

import numpy as np
import pbg_nfsim
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered assembly chain (coarse stages, in assembly order).
# RENAMED 2026-08-12 (NFSIM_WCM_WIRING_PLAN.md step 1): observable names now
# match the real v2ecoli bulk molecule IDs the model was renamed to use
# (CPLX0-7451[j] -> CPLX0_7451_j, etc. -- see generate_flagella_bngl.py's
# _safe_name()). Old placeholder names kept as comments per standing
# preserve-old-code rule.
CHAIN = [
    # ("flagellar_export_apparatus", "export apparatus", "#1f77b4"),
    ("CPLX0_7451_j", "export apparatus", "#1f77b4"),
    # ("flagellar_motor", "motor / basal body", "#2ca02c"),
    ("FLAGELLAR_MOTOR_COMPLEX_j", "motor / basal body", "#2ca02c"),
    ("flagellar_hook", "hook", "#ff7f0e"),  # no real bulk ID -- see model docstring
    ("flagella", "hook-basal-body complete (filament not modeled -- see module docstring)", "#d62728"),
]
MONOMERS = [
    # ("Free_fliF", "FliF (MS-ring)", "#9467bd"),
    ("Free_FLIF_FLAGELLAR_MS_RING_i", "FliF (MS-ring)", "#9467bd"),
    # ("Free_flgE", "FlgE (hook)", "#8c564b"),
    ("Free_G361_MONOMER_c", "FlgE (hook)", "#8c564b"),
    # ("Free_fliC", "FliC (flagellin)", "#17becf"),  REMOVED 2026-08-12 -- fliC
    # is no longer a species in this model at all (see FLIC REMOVAL note in
    # generate_flagella_bngl.py); this observable no longer exists.
]


def _observables(sim):
    for key in ("species", "observables"):
        d = sim.state.get(key)
        if isinstance(d, dict) and d:
            return d
    # fall back: search nested
    return sim.state.get("species", {}) or {}


def run(seconds, sample, n_steps):
    # Dispatch via the REGISTERED composite generator
    # (v2ecoli.composites.flagella_nfsim_assembly) instead of building an ad-hoc
    # in-code document — so this run is a real registered composite in the
    # Simulations DB and opens in the Composite Explorer.
    from v2ecoli.composites.flagella_nfsim_assembly import (
        flagella_nfsim_assembly, _register_nfsim_links,
    )
    core = allocate_core()
    _register_nfsim_links(core)

    doc = flagella_nfsim_assembly(
        core=core, n_steps=n_steps, complexation_interval=float(sample),
        production_interval=1.0, production_rate_scale=1.0,
    )
    sim = Composite({"state": doc}, core=core)

    names = [c[0] for c in CHAIN] + [m[0] for m in MONOMERS]
    rec = {"t": [], **{n: [] for n in names}}

    def snap(t):
        obs = _observables(sim)
        rec["t"].append(t)
        for n in names:
            rec[n].append(float(obs.get(n, 0.0)))

    snap(0.0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        sim.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    t = rec["t"] / 60.0
    for key, label, color in CHAIN:
        a.plot(t, rec[key], "-o", ms=3, color=color, label=label)
    a.set_title("NFsim rule-based flagellar assembly — staged intermediates")
    a.set_xlabel("time (min)"); a.set_ylabel("count"); a.legend(fontsize=8)

    for key, label, color in MONOMERS:
        b.plot(t, rec[key], "-", color=color, label=label)
    b.plot(t, rec["flagella"], "-o", ms=4, color="#d62728", lw=2, label="complete flagella")
    # Free monomer pools (thousands of FlgE etc.) dwarf the handful of assembled
    # flagella on a linear axis — Maya's "scale seems unfair, can't see the complete
    # flagella." A symlog y-axis keeps the low-count assembled structures legible
    # alongside the large free-monomer pools.
    b.set_yscale("symlog", linthresh=10)
    b.set_title("Free monomer pools vs assembled flagella  (symlog y — pools ≫ flagella)")
    b.set_xlabel("time (min)"); b.set_ylabel("count (symlog)"); b.legend(fontsize=8)
    fig.tight_layout()

    out = f"{STUDY_DIR}/charts/01_nfsim_assembly.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # Stacked-area view: cumulative assembly hierarchy over time.
    fig2, ax = plt.subplots(figsize=(9, 4.8))
    labels = [c[1] for c in CHAIN]
    colors = [c[2] for c in CHAIN]
    series = [rec[c[0]] for c in CHAIN]
    ax.stackplot(t, *series, labels=labels, colors=colors, alpha=0.85)
    ax.set_title("NFsim flagellar assembly — cumulative hierarchy (stacked)")
    ax.set_xlabel("time (min)"); ax.set_ylabel("count (stacked)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = f"{STUDY_DIR}/charts/02_nfsim_assembly_stacked.svg"
    fig2.savefig(out2, format="svg", bbox_inches="tight")
    plt.close(fig2)
    print("wrote", out2)

    # Small multiples: one panel per stage, each with its OWN y-axis scale.
    # Added 2026-08-12 -- the stacked/overlaid views above put hook (a fast,
    # cheap-to-finish intermediate that piles up waiting on the much slower
    # motor supply -- motor needs 9x FlhA plus the full export-apparatus
    # chain first) on the same visual scale as motor/flagella, so hook's
    # tall curve/band visually swallows the low-count stages entirely. This
    # view lets each stage's own shape/rise be seen regardless of its
    # absolute scale relative to the others.
    fig3, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (key, label, color) in zip(axes.flat, CHAIN):
        ax.plot(t, rec[key], "-o", ms=3, color=color)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("count")
        final_n = int(rec[key][-1])
        ax.text(0.97, 0.05, f"final: {final_n}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=color)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (min)")
    fig3.suptitle("NFsim flagellar assembly — each stage on its own scale", y=1.02)
    fig3.tight_layout()
    out3 = f"{STUDY_DIR}/charts/03_nfsim_assembly_small_multiples.svg"
    fig3.savefig(out3, format="svg", bbox_inches="tight")
    plt.close(fig3)
    print("wrote", out3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=50)
    args = ap.parse_args()

    rec = run(args.seconds, args.sample, args.n_steps)
    figure(rec)
    final = {k: int(rec[k][-1]) for k, _l, _c in CHAIN}
    print("final assembly state:", final)


if __name__ == "__main__":
    main()
