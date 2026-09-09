"""Real, fully-wired test of flagella_nfsim_complexation inside the real
ecoli_baseline composite (NFSIM_WCM_WIRING_PLAN.md step 3, final piece).

Added 2026-08-16, part of Maya Abdalla's flagella-cascade investigation.

Runs the FULL 55-process WCM composite (not an isolated Step diagnostic)
with the flagella_nfsim_complexation feature enabled -- ecoli-flagella-
nfsim-complexation, ecoli-flagella-filament-elongation, ecoli-flagella-
flgm-secretion, ecoli-flagella-transcription-regulation, exactly the same
set of downstream Steps flagella_regulation uses, just with NFsim replacing
the deterministic motor-switch/export-apparatus/motor-complex/nucleation
Steps. Tracks real bulk counts for the assembly intermediates plus FliA/
FlhD/FlgM (the regulatory loop) and the real nascent_flagellum count.

Division is disabled for this diagnostic (same rationale as other scripts
in this study -- division replaces agent "0" with two daughters mid-run,
which would complicate simple single-agent tracking; this is about
characterizing NFsim-in-the-WCM dynamics, not division/inheritance).

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/run_nfsim_wired_test.py \
        --seconds 7200 --sample 120 --seed 0 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.processes.flagella_filament_elongation import TARGET_LENGTH
from v2ecoli.steps.division import Division

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIG_NEXT_UPDATE = Division.next_update

# Standard starting condition used throughout this study's flagella-02
# diagnostics (e.g. run_diagnostic_no_division.py) -- NOT the same as
# artificially forcing assembly-intermediate species (which this script
# does not and should not do). This is a real, defined starting point for
# the Class II -> III regulatory cascade to evolve from, applied
# consistently across every diagnostic in this investigation for
# comparability, rather than an arbitrary/cold-start default.
INIT = {
    "CPLX0-7452[j]": 4,                 # complete flagella
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,          # free FliA
    "G369-MONOMER[c]": 800,             # FlgM
}

TRACK_IDS = {
    "CPLX0-7450[i]": "C-ring",
    "CPLX0-7451[j]": "export apparatus",
    "FLAGELLAR-MOTOR-COMPLEX[j]": "motor complex",
    "CPLX0-7452[j]": "complete flagella",
    "EG11355-MONOMER[c]": "free FliA",
    "G369-MONOMER[c]": "FlgM",
    "EG10320-MONOMER[c]": "free FlhD",
    # Added 2026-08-18 -- flagella_filament_elongation.py's own default
    # fliC_id; tracked so the FliC panel can show whether elongation is
    # FliC-availability-limited vs. just not-enough-elapsed-time (see the
    # "flagella stuck at 4" discussion this same day).
    "EG10321-MONOMER[e]": "free FliC",
    # Added 2026-08-18 per panel-redesign discussion -- previously only
    # free FlhD was tracked; FlhC and the FlhDC complex itself (the actual
    # master-regulator species that drives Class II transcription) were
    # missing entirely. Real bulk IDs confirmed via generate_flagella_bngl.py
    # (FlhC is MONOMER0-2488[c], NOT EG10319-MONOMER -- see that module's
    # own note) and COMPLEXATION_STOICHIOMETRY's 'flhDC' reaction.
    "MONOMER0-2488[c]": "free FlhC",
    "CPLX0-3930[c]": "FlhDC complex",
    # FlgM:FliA sequestration complex (equilibrium_reactions.tsv's
    # FLGM-FLIA-CPLX_RXN, ported from Maya's vEcoli biofilm branch) -- the
    # actual species mediating the Class II->III gate; real bulk ID
    # confirmed empirically (FLGM-FLIA-CPLX[c]) against a built composite.
    "FLGM-FLIA-CPLX[c]": "FlgM:FliA complex",
    # Added 2026-08-21 for the FliD double-consumption smoke test (see
    # generate_flagella_bngl.py's "FLID DOUBLE-CONSUMPTION FIX" note) --
    # want to directly confirm free FliD only drops by 5 per completed
    # flagellum now, not 10.
    "EG10841-MONOMER[e]": "free FliD",
    # Added 2026-08-21 for the FliS chaperone recycling smoke test (see
    # flagella_filament_elongation.py's own docstring note) -- want to
    # directly confirm elongation now draws from the protected complex
    # first and releases FliS back to the free pool as it does.
    "EG11388-MONOMER[c]": "free FliS",
    "FLIS-FLIC-CPLX[e]": "FLIS-FLIC-CPLX",
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir, feature="flagella_nfsim_complexation", nfsim_interval=None):
    Division.next_update = lambda self, timestep, states: {}
    try:
        enable_features(feature)
        comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
        enable_features()

        # DIAGNOSTIC ONLY (2026-08-18) -- the NFsim Step's own "interval"
        # config (default 1200s, see flagella_nfsim_complexation.py) is a
        # software/performance choice (avoid spawning a BioNetGen subprocess
        # every 2s tick), NOT a biological timescale. Each firing simulates
        # the full interval internally and only writes the NET delta back to
        # the real bulk store, so intermediates (C-ring, export apparatus)
        # that nucleate AND get fully consumed within one interval are
        # invisible to us. Overriding it here (after build, on the live
        # instance -- NOT touching the class default used by real runs) lets
        # this diagnostic see transient dynamics at finer resolution.
        if nfsim_interval is not None and feature == "flagella_nfsim_complexation":
            found = False
            for path, subtree in comp.step_paths.items():
                if path and path[-1] == "ecoli-flagella-nfsim-complexation":
                    instance = subtree.get("instance") if isinstance(subtree, dict) else None
                    if instance is not None:
                        instance.interval = float(nfsim_interval)
                        found = True
                        break
            if not found:
                raise RuntimeError("Could not find flagella-nfsim-complexation step instance to override interval")

        cell = comp.state["agents"]["0"]
        bulk = _arr(cell["bulk"])
        for name, val in INIT.items():
            bulk["count"][bulk_name_to_idx(name, bulk["id"])] = val
        idx = {name: bulk_name_to_idx(name, bulk["id"]) for name in TRACK_IDS}

        # nfsim_internal_observables (hook count, cumulative internal
        # 'flagella' count) only exists as a state key when the NFsim Step
        # is actually wired in -- added 2026-08-18 after this exact gap
        # caused real confusion (complete flagella looked "stuck" while
        # hook was actually progressing invisibly the whole time, only
        # found via a separate debug script). Tracked here now so this
        # script's own output/plot shows the full picture directly.
        track_internal = feature == "flagella_nfsim_complexation"

        rec = {"t": [], "n_nascent_flagellum": []}
        for name in TRACK_IDS:
            rec[name] = []
        if track_internal:
            rec["hook_internal"] = []
            rec["flagella_internal_cumulative"] = []
            # Added 2026-08-18 -- the model has THREE internal-only species
            # with no real bulk ID (see generate_flagella_bngl.py's
            # docstring / _INTERNAL_ONLY_OBSERVABLES), but only 2 were being
            # tracked -- 'flagellar_export_apparatus_subunit' (the export
            # apparatus's own precursor, between C-ring consumption and
            # export-apparatus formation) was silently missing.
            rec["export_apparatus_subunit_internal"] = []
        # Per-filament length trajectory, added 2026-08-18 -- flat/long-form
        # table (one row per active filament per snapshot, keyed by its
        # stable unique_index) since the number of simultaneously-active
        # filaments varies over time as new ones nucleate and others
        # complete. figure() groups rows by uid to plot each filament's own
        # growth curve.
        rec["filament_t"] = []
        rec["filament_uid"] = []
        rec["filament_length"] = []

        def snap(t):
            cell = comp.state["agents"]["0"]
            b = _arr(cell["bulk"])
            nf = _arr(cell["unique"]["nascent_flagellum"])
            active = nf["_entryState"].view(bool) if len(nf) else np.array([], dtype=bool)
            n_nascent = int(active.sum())
            rec["t"].append(t)
            rec["n_nascent_flagellum"].append(n_nascent)
            for name in TRACK_IDS:
                rec[name].append(int(b["count"][idx[name]]))
            if track_internal:
                internal = cell.get("nfsim_internal_observables") or {}
                rec["hook_internal"].append(internal.get("flagellar_hook", 0.0))
                rec["flagella_internal_cumulative"].append(internal.get("flagella", 0.0))
                rec["export_apparatus_subunit_internal"].append(
                    internal.get("flagellar_export_apparatus_subunit", 0.0))
            if n_nascent:
                uids = nf["unique_index"][active]
                lengths = nf["filament_length"][active]
                rec["filament_t"].extend([t] * n_nascent)
                rec["filament_uid"].extend(uids.tolist())
                rec["filament_length"].extend(lengths.tolist())

        snap(0.0)
        total = 0.0
        while total < seconds:
            chunk = min(sample, seconds - total)
            comp.run(chunk)
            total += chunk
            snap(total)

        return {k: np.array(v) for k, v in rec.items()}
    finally:
        Division.next_update = _ORIG_NEXT_UPDATE


# One color per species, shared between its individual panel and any
# combined/overlay panel it also appears in, so the same line color always
# means the same species across the whole figure.
COLORS = {
    "EG10320-MONOMER[c]": "#2ca02c",              # free FlhD
    "MONOMER0-2488[c]": "#17becf",                # free FlhC
    "CPLX0-3930[c]": "#bcbd22",                   # FlhDC complex
    "EG11355-MONOMER[c]": "#1f77b4",              # free FliA
    "G369-MONOMER[c]": "#d62728",                 # FlgM
    "FLGM-FLIA-CPLX[c]": "#9467bd",               # FlgM:FliA complex
    "CPLX0-7450[i]": "#1f77b4",                   # C-ring
    "export_apparatus_subunit_internal": "#e377c2",  # export apparatus subunit (internal)
    "CPLX0-7451[j]": "#ff7f0e",                   # export apparatus
    "FLAGELLAR-MOTOR-COMPLEX[j]": "#2ca02c",      # motor complex
    "hook_internal": "#8c564b",                   # hook (internal)
    "flagella_internal_cumulative": "#9467bd",    # hook-basal-body complete (internal)
    "n_nascent_flagellum": "black",               # nascent_flagellum count
    "EG10321-MONOMER[e]": "#8c564b",              # free FliC
    "CPLX0-7452[j]": "#d62728",                   # complete flagella
    "EG11388-MONOMER[c]": "#17becf",               # free FliS
    "FLIS-FLIC-CPLX[e]": "#e377c2",                # FLIS-FLIC-CPLX (protected FliC)
}


def _plot_filament_panel(ax, rec):
    """One line per nascent_flagellum unique_index, from the flat (t, uid,
    length) table snap() records -- plus a reference line at the real
    completion target. If no filament was ever created this run, say so
    explicitly instead of leaving a bare, unexplained blank axis."""
    ft = rec.get("filament_t")
    if ft is not None and len(ft):
        fuid = rec["filament_uid"]
        flen = rec["filament_length"]
        for uid in np.unique(fuid):
            mask = fuid == uid
            ax.plot(ft[mask] / 60.0, flen[mask], "-o", ms=2, lw=1)
        ax.axhline(TARGET_LENGTH, color="#888888", ls="--", lw=1, label=f"target ({TARGET_LENGTH})")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no nascent_flagellum\ncreated this run", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_ylabel("subunits")


def figure(rec, out_path, title, feature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = rec["t"] / 60.0
    track_internal = "hook_internal" in rec
    mechanism = "NFsim-driven" if feature == "flagella_nfsim_complexation" else "custom deterministic Steps (baseline)"

    # Redesigned 2026-08-18 per discussion: every assembly intermediate
    # gets its OWN panel, ordered to match the real temporal order of
    # flagellar assembly: FlhD/FlhC -> FlhDC (master regulator) -> FliA/
    # FlgM -> [FliA/FlgM overlaid, assembly-cascade overlaid -- placed
    # right here, immediately after the individual FliA/FlgM panels, so
    # the "relationship" and "big picture" views land where they're most
    # relevant] -> FlgM:FliA complex -> C-ring -> export-apparatus subunit
    # -> export apparatus -> motor complex -> hook -> hook-basal-body
    # trigger -> nascent flagellum -> filament elongation (+ its FliC
    # substrate) -> complete flagella. The cascade overlay only includes
    # the small-count intermediates (comparable 0-20ish scale) --
    # FlhDC/FlgM:FliA-CPLX sit in the hundreds-thousands and would flatten
    # everything else, so they stay out of it.
    panels = [
        ("Free FlhD", "EG10320-MONOMER[c]"),
        ("Free FlhC", "MONOMER0-2488[c]"),
        ("FlhDC complex (CPLX0-3930[c])", "CPLX0-3930[c]"),
        ("Free FliA", "EG11355-MONOMER[c]"),
        ("FlgM", "G369-MONOMER[c]"),
        # The two combined overlays sit right here -- immediately after
        # seeing FliA/FlgM individually, per 2026-08-18 request -- rather
        # than at the end, so the "relationship" and "big picture" views
        # come right where they're most relevant before diving into the
        # rest of the cascade in detail.
        ("__overlay_regulatory__", None),
        ("__overlay_cascade__", None),
        ("FlgM:FliA complex (FLGM-FLIA-CPLX[c])", "FLGM-FLIA-CPLX[c]"),
        ("C-ring (CPLX0-7450[i])", "CPLX0-7450[i]"),
    ]
    if track_internal:
        panels.append(("Export apparatus subunit (internal, no real bulk ID)", "export_apparatus_subunit_internal"))
    panels.append(("Export apparatus (CPLX0-7451[j])", "CPLX0-7451[j]"))
    panels.append(("Motor complex (FLAGELLAR-MOTOR-COMPLEX[j])", "FLAGELLAR-MOTOR-COMPLEX[j]"))
    if track_internal:
        panels += [
            ("Hook (internal, no real bulk ID)", "hook_internal"),
            ("Hook-basal-body complete (internal, cumulative)", "flagella_internal_cumulative"),
        ]
    panels += [
        ("nascent_flagellum (unique molecules)", "n_nascent_flagellum"),
        ("__filament__", None),  # special: per-filament elongation panel
        ("Free FliC (elongation substrate)", "EG10321-MONOMER[e]"),
        # Added 2026-08-21 for the FliS chaperone recycling fix: placed
        # right next to free FliC since they're the two sides of the same
        # mechanism (elongation.py draws from FLIS-FLIC-CPLX first,
        # releasing FliS 1:1, before falling back to free FliC).
        ("Free FliS (chaperone)", "EG11388-MONOMER[c]"),
        ("FLIS-FLIC-CPLX (protected FliC)", "FLIS-FLIC-CPLX[e]"),
        ("Complete flagella (CPLX0-7452[j])", "CPLX0-7452[j]"),
    ]

    overlay_regulatory = ("EG11355-MONOMER[c]", "G369-MONOMER[c]")
    overlay_cascade = [
        "CPLX0-7450[i]",
        "export_apparatus_subunit_internal",
        "CPLX0-7451[j]",
        "FLAGELLAR-MOTOR-COMPLEX[j]",
        "hook_internal",
        "flagella_internal_cumulative",
        "n_nascent_flagellum",
        "CPLX0-7452[j]",
    ]
    if not track_internal:
        overlay_cascade = [k for k in overlay_cascade
                            if k not in ("export_apparatus_subunit_internal", "hook_internal",
                                         "flagella_internal_cumulative")]

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flat

    for ax, (panel_title, key) in zip(axes_flat, panels):
        if panel_title == "__filament__":
            _plot_filament_panel(ax, rec)
        elif panel_title == "__overlay_regulatory__":
            for k in overlay_regulatory:
                ax.plot(t, rec[k], color=COLORS[k], label=TRACK_IDS.get(k, k))
            ax.set_ylabel("count"); ax.legend(fontsize=7)
            panel_title = "FliA / FlgM overlaid (Class II→III gate)"
        elif panel_title == "__overlay_cascade__":
            for k in overlay_cascade:
                label = {"export_apparatus_subunit_internal": "export apparatus subunit (internal)",
                         "hook_internal": "hook (internal)",
                         "flagella_internal_cumulative": "hook-basal-body complete (internal)",
                         "n_nascent_flagellum": "nascent_flagellum"}.get(k, TRACK_IDS.get(k, k))
                ax.plot(t, rec[k], "-o", ms=2, color=COLORS[k], label=label)
            ax.set_ylabel("count"); ax.legend(fontsize=6, ncol=2)
            panel_title = "Assembly cascade, overlaid"
        else:
            ax.plot(t, rec[key], "-o", ms=3, color=COLORS.get(key, "#333333"))
            ax.set_ylabel("count")
        ax.set_title(panel_title, fontsize=9)

    # Hide any unused cells (n_rows*4 - len(panels)) rather than leaving
    # bare axes.
    for ax in list(axes_flat)[len(panels):]:
        ax.axis("off")

    # x-label goes on the last POPULATED row, not the literal bottom row
    # (which is empty/hidden whenever len(panels) isn't a multiple of 4).
    last_row = (len(panels) - 1) // n_cols
    axes_2d = np.atleast_2d(axes)
    for col in range(n_cols):
        if last_row * n_cols + col < len(panels):
            axes_2d[last_row, col].set_xlabel("time (min)")
    # Avoid duplicate tick labels on the inner rows of the populated area.
    for ax in list(axes_flat)[:len(panels)]:
        ax.label_outer()

    fig.suptitle(f"{title} ({mechanism})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=7200)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=str, default="out/cache_full_flit_v11")
    ap.add_argument("--feature", type=str, default="flagella_nfsim_complexation",
                     choices=["flagella_nfsim_complexation", "flagella_regulation"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--nfsim-interval", type=float, default=None,
                     help="Diagnostic-only override of the NFsim Step's own firing "
                          "interval (default in flagella_nfsim_complexation.py is "
                          "1200s, a performance choice not a biological one). Set "
                          "smaller to see transient intermediate dynamics.")
    args = ap.parse_args()

    rec = run(args.seconds, args.sample, args.seed, args.cache_dir, args.feature,
               nfsim_interval=args.nfsim_interval)
    out = args.out or f"{STUDY_DIR}/charts/16_nfsim_wired_{args.feature}_seed{args.seed}.svg"
    # INIT overrides spelled out in the title (2026-08-19) so a chart is
    # self-describing without cross-referencing this script's source.
    init_str = ", ".join(f"{TRACK_IDS.get(k, k)}={v}" for k, v in INIT.items())
    title = f"{args.feature}, seed={args.seed}, {args.seconds}s | INIT: {init_str}"
    figure(rec, out, title, args.feature)

    final = {k: int(rec[k][-1]) for k in TRACK_IDS}
    final["n_nascent_flagellum"] = int(rec["n_nascent_flagellum"][-1])
    if "hook_internal" in rec:
        final["hook_internal"] = rec["hook_internal"][-1]
        final["flagella_internal_cumulative"] = rec["flagella_internal_cumulative"][-1]
    print("final state:", final)


if __name__ == "__main__":
    main()
