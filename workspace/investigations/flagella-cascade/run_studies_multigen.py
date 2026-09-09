"""Run the flagella-cascade studies across >= 2 cell generations.

Chains single-lineage generations (run to division, keep one daughter, rebuild)
following reports/multigeneration_report.py, but (a) enables the opt-in
`flagella_regulation` feature before EVERY generation's build and (b) records
flagella observables directly from composite.state each chunk.

Two lineages:
  * regulation OFF -> flagella-01-overexpression-baseline (multigen)
  * regulation ON  -> flagella-02-transcription-regulation (multigen sumgate + flgm/flia charts)
    "Regulation ON" = the flagella_regulation feature (2 process Steps) PLUS
    Maya's full 4-value initial-state override (FULL_OVERRIDE below), applied
    once at generation 1's start — investigation-wide definition, used
    uniformly for both studies' figures now.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/run_studies_multigen.py \
        --generations 2 --sample 50
"""
import argparse
import os

import numpy as np

from v2ecoli import build_composite
from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.composites.ecoli_baseline import baseline, seed_mass_listener, enable_features
from v2ecoli.composites._helpers import set_null_emitter_override
from v2ecoli.library.division import divide_cell
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.library.schema import bulk_name_to_idx
from process_bigraph import Composite

STUDIES = "workspace/investigations/flagella-cascade/studies"
SNAPSHOT = 50
MAX_GEN_DUR = 3600

_CELL_DATA_KEYS = {
    "bulk", "unique", "listeners", "environment", "boundary", "global_time",
    "timestep", "divide", "division_threshold", "process_state", "allocator_rng",
}

CLASS_II_CISTRONS = ["EG10322", "EG11346", "EG11347", "G358", "G357", "G7028", "EG11355"]
CLASS_III_CISTRONS = ["EG10321", "EG10317", "EG11967", "EG11545",
                      "EG10601", "EG10602", "EG10146", "EG10149", "G369"]

FULL_OVERRIDE = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _extract_cell_data(cell):
    return {k: v for k, v in cell.items()
            if k in _CELL_DATA_KEYS or k.startswith("request_") or k.startswith("allocate_")}


def _tu_indexes(cache_dir):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    cII = [rna_ids.index(r) for r in cfg["flg_classII_rnaids"]]
    cIII = [rna_ids.index(r) for r in cfg["flg_classIII_rnaids"]]
    return set(cII), set(cIII)


def _snap(comp, tu_II, tu_III, idxs):
    cell = comp.state["agents"]["0"]
    b = _arr(cell["bulk"])
    p = _arr(cell["unique"]["promoter"])
    m = p["_entryState"].view(bool)
    tu, ov = p["TU_index"][m], p["init_prob_override"][m]
    II = ov[np.isin(tu, list(tu_II))]
    III = ov[np.isin(tu, list(tu_III))]

    def cnt(i):
        return int(b["count"][i]) if i is not None else 0

    return {
        "dry_mass": fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0)),
        "flag": cnt(idxs["flag"]), "motor": cnt(idxs["motor"]),
        "flhDC": cnt(idxs["flhDC"]), "fliA": cnt(idxs["fliA"]), "flgM": cnt(idxs["flgM"]),
        "II": float(II.mean()) if len(II) else 0.0,
        "III": float(III.mean()) if len(III) else 0.0,
    }


def _idxs(comp):
    bids = _arr(comp.state["agents"]["0"]["bulk"])["id"]

    def i(n):
        try:
            return bulk_name_to_idx(n, bids)
        except Exception:
            return None
    return {"flag": i("CPLX0-7452[j]"), "motor": i("FLAGELLAR-MOTOR-COMPLEX[j]"),
            "flhDC": i("CPLX0-3930[c]"), "fliA": i("EG11355-MONOMER[c]"),
            "flgM": i("G369-MONOMER[c]")}


def _run_gen(comp, tu_II, tu_III, sample, max_dur, t_cum, gen_idx, rows):
    idxs = _idxs(comp)
    total = 0.0
    last_cell = None
    rows.append({"gen": gen_idx, "t_cum": t_cum, "t_gen": 0.0, **_snap(comp, tu_II, tu_III, idxs)})
    divided = False
    while total < max_dur:
        chunk = min(sample, max_dur - total)
        try:
            comp.run(chunk)
        except Exception as e:
            s = str(e)
            if "divide" in s.lower() or "_add" in s or "_remove" in s \
               or comp.state.get("agents", {}).get("0") is None:
                divided = True
                break
            raise
        total += chunk
        cur = comp.state.get("agents", {}).get("0")
        if cur is None:
            divided = True
            break
        last_cell = _extract_cell_data(cur)
        rows.append({"gen": gen_idx, "t_cum": t_cum + total, "t_gen": total,
                     **_snap(comp, tu_II, tu_III, idxs)})
    return total, divided, last_cell


def run_multigen(features, n_gens, sample, max_dur, seed, cache_dir):
    # We read observables straight from composite.state, so no emitter is needed.
    # The default ParquetEmitter races on its partition .tmp files across repeated
    # divisions (post-division partition wipe) — the null emitter avoids that and is
    # faster. See feedback_parquet_post_division_partition_wipe.
    set_null_emitter_override(True)
    tu_II, tu_III = _tu_indexes(cache_dir)
    rows = []
    t_cum = 0.0

    enable_features(*features)
    comp = build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()
    if features:
        # Investigation-wide "regulation ON" = feature + full override, together.
        bulk = _arr(comp.state["agents"]["0"]["bulk"])
        bids = bulk["id"]
        for name, val in FULL_OVERRIDE.items():
            bulk["count"][bulk_name_to_idx(name, bids)] = val
    dur, divided, last_cell = _run_gen(comp, tu_II, tu_III, sample, max_dur, t_cum, 1, rows)
    t_cum += dur
    print(f"    gen 1: sim {dur:.0f}s divided={divided} flag={rows[-1]['flag']}")

    for g in range(2, n_gens + 1):
        if not last_cell or "bulk" not in last_cell:
            print(f"    gen {g}: no daughter state — stopping"); break
        d1, _d2 = divide_cell(last_cell)
        core = build_core()
        enable_features(*features)
        # Derive the daughter RNG seed from the lineage seed so distinct lineages
        # (multi-seed ensemble) actually diverge in their stochastic draws, not just
        # in the carried-forward state.
        doc = baseline(core=core, seed=seed * 1000 + g, cache_dir=cache_dir)
        enable_features()
        agent = doc["state"]["agents"]["0"]
        for k in ("bulk", "unique", "environment", "boundary"):
            if k in d1:
                agent[k] = d1[k]
        agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
        seed_mass_listener(agent, core)
        comp = Composite(doc, core=core)
        dur, divided, last_cell = _run_gen(comp, tu_II, tu_III, sample, max_dur, t_cum, g, rows)
        t_cum += dur
        print(f"    gen {g}: sim {dur:.0f}s divided={divided} flag={rows[-1]['flag']}")
    return rows


def _cols(rows, key):
    return np.array([r[key] for r in rows])


def _gen_bounds(rows):
    """x positions where generation index increments (division markers)."""
    bounds = []
    for i in range(1, len(rows)):
        if rows[i]["gen"] != rows[i - 1]["gen"]:
            bounds.append(rows[i]["t_cum"] / 60.0)
    return bounds


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    return plt


def _shade(ax, rows):
    gens = sorted({r["gen"] for r in rows})
    colors = ["#eef4ff", "#fff4ee", "#eefff2", "#f7eeff"]
    for r in rows:
        pass
    for gi in gens:
        xs = [r["t_cum"] / 60.0 for r in rows if r["gen"] == gi]
        if xs:
            ax.axvspan(min(xs), max(xs), color=colors[(gi - 1) % len(colors)], alpha=0.5, zorder=0)
    for b in _gen_bounds(rows):
        ax.axvline(b, color="#c0392b", ls="--", lw=1, alpha=0.7)


def save(fig, slug, name):
    d = f"{STUDIES}/{slug}/charts"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", path)


def fig_baseline(off, on):
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(_cols(off, "t_cum") / 60, _cols(off, "flag"), "-", color="#9467bd", label="flagella (reg OFF)")
    a.plot(_cols(on, "t_cum") / 60, _cols(on, "flag"), "-", color="#2ca02c", label="flagella (reg ON: feature + full override)")
    _shade(a, off)
    a.set_title("Complete flagella across generations — OFF vs ON (feature + full override)")
    a.set_xlabel("time (min)"); a.set_ylabel("CPLX0-7452 count"); a.legend(fontsize=8)

    a2 = b
    a2.plot(_cols(off, "t_cum") / 60, _cols(off, "dry_mass"), "-", color="#9467bd", label="reg OFF")
    a2.plot(_cols(on, "t_cum") / 60, _cols(on, "dry_mass"), "-", color="#2ca02c", label="reg ON: feature + full override")
    _shade(a2, off)
    a2.set_title("Dry mass across generations (division = dashed)")
    a2.set_xlabel("time (min)"); a2.set_ylabel("dry mass (fg)"); a2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "flagella-01-overexpression-baseline", "02_multigen_flagella_and_mass")


def fig_cascade(on):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(_cols(on, "t_cum") / 60, _cols(on, "II"), "-o", ms=3, color="#1f77b4", label="Class II ⟨override⟩")
    ax.plot(_cols(on, "t_cum") / 60, _cols(on, "III"), "-s", ms=3, color="#d62728", label="Class III ⟨override⟩")
    _shade(ax, on)
    ax.set_title("SUM-gate Class II vs Class III across generations")
    ax.set_xlabel("time (min)"); ax.set_ylabel("mean init_prob_override"); ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "flagella-02-transcription-regulation", "03_multigen_classII_classIII")


def fig_feedback(on):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(_cols(on, "t_cum") / 60, _cols(on, "flgM"), "-", color="#ff7f0e", label="FlgM G369-MONOMER[c]")
    ax.plot(_cols(on, "t_cum") / 60, _cols(on, "fliA"), "-", color="#2ca02c", label="free FliA EG11355-MONOMER[c]")
    axb = ax.twinx()
    axb.plot(_cols(on, "t_cum") / 60, _cols(on, "flag"), "-", color="#9467bd", alpha=0.6, label="flagella")
    axb.set_ylabel("flagella", color="#9467bd")
    _shade(ax, on)
    ax.set_title("FlgM / free FliA across generations")
    ax.set_xlabel("time (min)"); ax.set_ylabel("molecule count")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "flagella-02-transcription-regulation", "05_multigen_flgm_flia")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--sample", type=int, default=SNAPSHOT)
    ap.add_argument("--max-gen-dur", type=int, default=MAX_GEN_DUR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    args = ap.parse_args()

    print("== regulation OFF lineage (study 01 baseline) ==")
    off = run_multigen([], args.generations, args.sample, args.max_gen_dur, args.seed, args.cache_dir)
    print("== regulation ON lineage: feature + full override (studies 01 + 03) ==")
    on = run_multigen(["flagella_regulation"], args.generations, args.sample, args.max_gen_dur, args.seed, args.cache_dir)

    fig_baseline(off, on)
    fig_cascade(on)
    fig_feedback(on)

    ng_off = len({r["gen"] for r in off}); ng_on = len({r["gen"] for r in on})
    print(f"\nOFF generations: {ng_off}  ON (full override) generations: {ng_on}")
    print("OFF flagella by gen-end: " +
          ", ".join(f"g{g}={[r['flag'] for r in off if r['gen']==g][-1]}" for g in sorted({r['gen'] for r in off})))
    print("ON  flagella by gen-end (full override): " +
          ", ".join(f"g{g}={[r['flag'] for r in on if r['gen']==g][-1]}" for g in sorted({r['gen'] for r in on})))


if __name__ == "__main__":
    main()
