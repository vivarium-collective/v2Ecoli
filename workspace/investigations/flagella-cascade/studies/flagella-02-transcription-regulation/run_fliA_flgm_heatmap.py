"""FliA0 x FlgM0 grid scan -- justify Maya's override values (500 free FliA,
800 FlgM) by showing where they sit in the 2D initial-condition space, not by
claiming they are uniquely "correct".

Biological framing: FlgM0 >= FliA0 means the anti-sigma-factor checkpoint is
essentially intact at t=0 (sigma-28/FliA mostly sequestered) -- the expected
starting state for a cell that hasn't yet completed hook-basal-body assembly
and started type-III secretion. FlgM0 << FliA0 would mean the checkpoint is
already broken at t=0, a biologically wrong starting point for a model meant
to show the cascade unfold from a pre-assembly state. This scan holds
flagella=4, motor=0, K_flhDC=50, K_fliA=600 fixed (current production values)
and sweeps only FliA0 x FlgM0, at the 900s window (not 600s -- a shorter
window can misrepresent the release phase, see chart 04/01 history).

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/run_fliA_flgm_heatmap.py \
        --seconds 900 --sample 30 --cache-dir out/cache_full

    # regenerate just the figure from a previous run's saved results:
    ... run_fliA_flgm_heatmap.py --skip-run
"""
import argparse
import json
import os

import numpy as np

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_JSON = f"{STUDY_DIR}/fliA_flgm_grid_results.json"
DEFAULT_OUT_FIG = f"{STUDY_DIR}/charts/fliA_flgm_heatmap.svg"

FLIA_GRID = [100, 250, 500, 750, 1000, 1500]
FLGM_GRID = [200, 400, 600, 800, 1200, 1600]

# Fixed at Maya's original vEcoli override values -- only FliA0/FlgM0 vary.
FIXED_OVERRIDE = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
}
READ_IDS = ["CPLX0-7452[j]", "EG11355-MONOMER[c]", "G369-MONOMER[c]"]

MAYA_FLIA0 = 500
MAYA_FLGM0 = 800


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run_one(fliA0, flgm0, seconds, sample, seed, cache_dir):
    import v2ecoli
    from v2ecoli.composites.ecoli_baseline import enable_features
    from v2ecoli.library.schema import bulk_name_to_idx

    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    overrides = dict(FIXED_OVERRIDE)
    overrides["EG11355-MONOMER[c]"] = fliA0
    overrides["G369-MONOMER[c]"] = flgm0
    for name, val in overrides.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val
    idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

    rec = {"t": [], "flgM": [], "fliA": [], "flag": []}

    def snap(t):
        b = _arr(comp.state["agents"]["0"]["bulk"])
        rec["t"].append(t)
        rec["flgM"].append(int(b["count"][idx["G369-MONOMER[c]"]]))
        rec["fliA"].append(int(b["count"][idx["EG11355-MONOMER[c]"]]))
        rec["flag"].append(int(b["count"][idx["CPLX0-7452[j]"]]))

    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        try:
            comp.run(chunk)
        except Exception as e:
            s = str(e)
            if "divide" in s.lower() or "_add" in s or "_remove" in s \
               or comp.state.get("agents", {}).get("0") is None:
                # Cell divided partway through -- not a solver failure. Return
                # whatever trajectory was collected before division, flagged
                # distinctly so run_grid doesn't mislabel it as a CRASH.
                rec["divided"] = True
                return rec
            raise
        total += chunk
        cur = comp.state.get("agents", {}).get("0")
        if cur is None:
            rec["divided"] = True
            return rec
        snap(total)
    rec["divided"] = False
    return rec


def run_grid(seconds, sample, seed, cache_dir, out_json, resume=False):
    results = []
    done = set()
    if resume and os.path.exists(out_json):
        with open(out_json) as f:
            results = json.load(f)
        done = {(r["fliA0"], r["flgm0"]) for r in results}
        print(f"resuming: {len(done)} combos already completed, skipping those", flush=True)

    total_combos = len(FLIA_GRID) * len(FLGM_GRID)
    i = 0
    for fliA0 in FLIA_GRID:
        for flgm0 in FLGM_GRID:
            i += 1
            if (fliA0, flgm0) in done:
                continue
            print(f"[{i}/{total_combos}] FliA0={fliA0} FlgM0={flgm0}", flush=True)
            try:
                rec = run_one(fliA0, flgm0, seconds, sample, seed, cache_dir)
                results.append({"fliA0": fliA0, "flgm0": flgm0, "crashed": False, **rec})
                early_i = 1 if len(rec["fliA"]) > 1 else 0
                tag = " (DIVIDED early)" if rec.get("divided") else ""
                print(f"    FliA {rec['fliA'][0]} -> t={rec['t'][early_i]:.0f}s: "
                      f"{rec['fliA'][early_i]} -> t={rec['t'][-1]:.0f}s: {rec['fliA'][-1]}{tag}",
                      flush=True)
            except Exception as e:
                # The equilibrium solver ("Negative values at equilibrium steady
                # state") can fail for some FliA0/FlgM0 combinations -- this is
                # itself a finding worth showing on the heatmap (a real failure
                # region), not something to silently skip or let crash the batch.
                results.append({"fliA0": fliA0, "flgm0": flgm0, "crashed": True,
                                "error": str(e), "t": [], "flgM": [], "fliA": [], "flag": []})
                print(f"    CRASHED: {e}", flush=True)
            # Incremental save after every combo -- a later crash must not lose
            # already-completed runs (each takes ~80s, too expensive to redo).
            with open(out_json, "w") as f:
                json.dump(results, f)
    print("wrote", out_json)
    return results


def figure(results, out_fig, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fliA_vals = sorted({r["fliA0"] for r in results})
    flgm_vals = sorted({r["flgm0"] for r in results})
    nx, ny = len(fliA_vals), len(flgm_vals)
    fliA_idx = {v: i for i, v in enumerate(fliA_vals)}
    flgm_idx = {v: i for i, v in enumerate(flgm_vals)}

    # Panel A: FliA fold-change (FliA_end / FliA0) -- colored on a log2 scale
    # (diverging, centered at 0 = 1x/break-even) but ANNOTATED with the actual
    # fold-change number, not the log, for readability.
    log2_ratio = np.full((ny, nx), np.nan)
    ratio = np.full((ny, nx), np.nan)
    fliA_final = np.full((ny, nx), np.nan)
    flagella_final = np.full((ny, nx), np.nan)
    crashed = np.zeros((ny, nx), dtype=bool)
    divided = np.zeros((ny, nx), dtype=bool)
    for r in results:
        xi, yi = fliA_idx[r["fliA0"]], flgm_idx[r["flgm0"]]
        if r.get("crashed") or not r["fliA"]:
            crashed[yi, xi] = True
            continue
        if r.get("divided"):
            divided[yi, xi] = True
        rat = r["fliA"][-1] / r["fliA0"]
        ratio[yi, xi] = rat
        log2_ratio[yi, xi] = np.log2(rat) if rat > 0 else np.nan
        fliA_final[yi, xi] = r["fliA"][-1]
        flagella_final[yi, xi] = r["flag"][-1]

    n_crashed = int(crashed.sum())
    n_divided = int(divided.sum())
    fig, (a, c, b) = plt.subplots(1, 3, figsize=(18.5, 5.2))

    def _panel(ax, color_mat, annotate_mat, title, cbar_label, fmt,
               cmap_name="viridis", center_zero=False):
        masked = np.ma.masked_invalid(color_mat)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        if center_zero:
            finite = color_mat[np.isfinite(color_mat)]
            vext = float(np.max(np.abs(finite))) if finite.size else 1.0
            im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto",
                           vmin=-vext, vmax=vext)
        else:
            im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(nx)); ax.set_xticklabels(fliA_vals)
        ax.set_yticks(range(ny)); ax.set_yticklabels(flgm_vals)
        ax.set_xlabel("FliA0 (EG11355-MONOMER[c])")
        ax.set_ylabel("FlgM0 (G369-MONOMER[c])")
        ax.set_title(title)
        for yi in range(ny):
            for xi in range(nx):
                if crashed[yi, xi]:
                    ax.text(xi, yi, "CRASH", ha="center", va="center",
                            color="#d62728", fontsize=6, fontweight="bold")
                elif not np.isnan(annotate_mat[yi, xi]):
                    txt = fmt.format(annotate_mat[yi, xi])
                    if divided[yi, xi]:
                        txt += "*"
                    ax.text(xi, yi, txt, ha="center", va="center",
                            color="white", fontsize=7)
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)
        # Maya's point -- offset into the cell's upper-right corner so the star
        # doesn't sit directly on top of (and hide) that cell's number.
        if MAYA_FLIA0 in fliA_idx and MAYA_FLGM0 in flgm_idx:
            ax.scatter([fliA_idx[MAYA_FLIA0] + 0.33], [flgm_idx[MAYA_FLGM0] + 0.33],
                       marker="*", s=220, facecolor="none", edgecolor="red",
                       linewidth=2, zorder=3, clip_on=False)

    _panel(a, log2_ratio, ratio,
           f"FliA fold-change at t={seconds/60:.0f} min (FliA_end / FliA0)\n"
           "white~1x break-even, blue=<1x still sequestered, red=>1x overshoot\n"
           "CAUTION: normalized per-cell by its own FliA0 -- only fair to compare WITHIN a column",
           "log2(fold change)", "{:.2f}x", cmap_name="RdBu_r", center_zero=True)
    _panel(c, fliA_final, fliA_final,
           f"Absolute free FliA at t={seconds/60:.0f} min (no normalization)\n"
           "directly comparable across the WHOLE grid",
           "FliA count", "{:.0f}")
    _panel(b, flagella_final, flagella_final,
           f"Complete flagella (CPLX0-7452[j]) at t={seconds/60:.0f} min",
           "flagella count", "{:.0f}")

    if n_divided:
        fig.text(0.5, -0.02, f"* = cell divided before t={seconds/60:.0f} min "
                 "(value is at the point of division, not the full window)",
                 ha="center", fontsize=8, style="italic")

    fig.suptitle(f"FliA0 x FlgM0 grid -- red star = Maya's override (FliA0=500, FlgM0=800)"
                 + (f"  |  {n_crashed} cell(s) hit the equilibrium-solver error" if n_crashed else ""))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_fig)
    if n_crashed:
        print(f"CRASHED cells ({n_crashed}):")
        for r in results:
            if r.get("crashed"):
                print(f"  FliA0={r['fliA0']} FlgM0={r['flgm0']}: {r.get('error')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full")
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    ap.add_argument("--skip-run", action="store_true",
                     help="regenerate the figure from an existing --out-json instead of re-running")
    ap.add_argument("--resume", action="store_true",
                     help="continue an existing --out-json, only running combos not already in it")
    args = ap.parse_args()

    if args.skip_run:
        with open(args.out_json) as f:
            results = json.load(f)
    else:
        results = run_grid(args.seconds, args.sample, args.seed, args.cache_dir,
                           args.out_json, resume=args.resume)

    figure(results, args.out_fig, args.seconds)


if __name__ == "__main__":
    main()
