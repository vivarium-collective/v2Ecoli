"""Re-verify divide_nascent_flagellum against the CURRENT cache (v11), post
this session's structural/architecture fixes. Follows ONE daughter (chosen
unbiased-randomly, same pattern as run_lineage_multigen.py) across multiple
real divisions to see continued dynamics, not just the first split.

Added 2026-08-11, part of Maya Abdalla's flagella-cascade investigation.

The divider itself (v2ecoli/library/division.py:divide_nascent_flagellum)
was found and fixed on 2026-08-06 -- see CHANGES_2026-08-06.md section 6b --
and verified at the time with a real division producing non-overlapping
length sets in the two daughters. That verification pre-dates every fix in
this session (FliT checkpoint removal, export-apparatus/MS-ring hierarchy
fixes, the SSA-vs-Step race fix, the nucleation first-tick fix, target_length
10000->5000) -- none of those fixes touch division.py, but none of them had
been tested against a run that actually crosses the division boundary either
(every single-gen diagnostic this session deliberately stayed under 2400s to
avoid it). This script closes that gap.

Stronger check than the original 2026-08-06 verification: that one compared
filament_length SETS (two different flagella could coincidentally share a
length). This one compares unique_index sets -- the real molecular identity
-- confirming the exact partition: no duplication (an index in both
daughters), no loss (an index in neither), no invention (a daughter index
absent from the mother).

REMOVED (2026-08-11, same day): an earlier version tried to capture the
Division Step's own internal division_time from its stdout log (via a Tee +
regex), reasoning that it would be more tick-exact than this script's own
--sample-grained clock. That capture had a real bug (a stale buffer that
kept matching generation 1's timestamp for every later generation) AND,
once "fixed," was STILL wrong -- it reported generation 2/3 dividing every
~5-8 minutes, which alarmed Maya (correctly -- real E. coli's fastest known
division time under any condition is ~20 min). Direct comparison against
this script's own independently-recorded sample timestamps (t=0->2520s,
2580->5340s, 5400->8700s -- i.e. ~42/46/55 min per generation, all
biologically ordinary) proved the stdout-parsing path was ITSELF the bug,
not a property of the simulation. Removed entirely; the sampling loop's own
`total` clock was correct the whole time and is now the only clock used.

Now also tracks dry_mass per generation (same read as
run_lineage_multigen.py) to check WHY generation time climbs slightly each
round (42 -> 46 -> 55 min in the run that prompted this) instead of holding
steady at a fixed doubling time.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/verify_division_nascent_flagellum.py \
        --cache-dir out/cache_full_flit_v11 --seed 0 --generations 3
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _active_nf(cell):
    nf = _arr(cell["unique"]["nascent_flagellum"])
    mask = nf["_entryState"].view(bool)
    return nf["unique_index"][mask].copy(), nf["filament_length"][mask].copy()


def run(cache_dir, seed, sample, n_generations, seconds_cap):
    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    flag_idx = bulk_name_to_idx("CPLX0-7452[j]", bids)
    for name, val in INIT.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val

    def _record(seg, t, cell):
        idx, length = _active_nf(cell)
        b = _arr(cell["bulk"])
        dry_mass = fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0))
        seg["t"].append(t)
        seg["n_nascent"].append(len(idx))
        seg["max_len"].append(int(length.max()) if len(length) else 0)
        seg["flag"].append(int(b["count"][flag_idx]))
        seg["dry_mass"].append(dry_mass)
        return idx, length

    segments = []   # one dict per generation, each its own continuous line
    divisions = []  # one dict per real division event

    agent_id = "0"
    gen = 1
    total = 0.0
    seg = {"agent": agent_id, "gen": gen, "t": [], "n_nascent": [], "max_len": [], "flag": [], "dry_mass": []}
    segments.append(seg)
    prev_idx, prev_len = _record(seg, 0.0, comp.state["agents"][agent_id])
    print(f"t=0s  agent={agent_id}  gen={gen}  n_nascent={len(prev_idx)}  "
          f"unique_index={sorted(prev_idx.tolist())}  dry_mass={seg['dry_mass'][-1]:.1f}fg")

    while total < seconds_cap and gen <= n_generations:
        chunk = min(sample, seconds_cap - total)
        comp.run(chunk)
        total += chunk

        agents = comp.state.get("agents", {})
        if len(agents) > 1:
            print(f"\nDIVISION detected -- sampling loop noticed at t~{total:.0f}s "
                  f"(previous full sample at t={seg['t'][-1]:.0f}s) -- daughters: {sorted(agents.keys())}")

            d_idx, d_len = {}, {}
            for aid, cell in agents.items():
                idx, length = _active_nf(cell)
                d_idx[aid], d_len[aid] = idx, length
                print(f"  agent={aid}  n_nascent={len(idx)}  unique_index={sorted(idx.tolist())}  "
                      f"filament_length={length.tolist()}")

            all_daughter_idx = np.concatenate(list(d_idx.values()))
            ids = sorted(agents.keys())
            overlap = np.intersect1d(d_idx[ids[0]], d_idx[ids[1]])
            ok = (
                len(overlap) == 0
                and len(all_daughter_idx) == len(prev_idx)
                and set(all_daughter_idx.tolist()) == set(prev_idx.tolist())
            )
            print(f"  mother (pre-division) unique_index={sorted(prev_idx.tolist())}")
            print(f"  overlap between daughters (should be empty): {sorted(overlap.tolist())}")
            print(f"  {'PASS' if ok else 'FAIL'}: exact partition of the mother's set")

            divisions.append({
                "gen": gen, "division_t": total,
                "mother_idx": prev_idx, "mother_len": prev_len,
                "daughters": {aid: {"idx": d_idx[aid], "len": d_len[aid]} for aid in ids},
                "ok": ok,
            })

            # Pick one daughter to follow, unbiased -- same pattern as
            # run_lineage_multigen.py (decorrelated from ID string order).
            pick_rng = np.random.RandomState(seed=(seed * 1000 + gen) % (2**31 - 1))
            keep_id = ids[pick_rng.randint(len(ids))]
            discard_ids = [aid for aid in ids if aid != keep_id]
            comp.apply({"agents": {"_remove": discard_ids}})
            agent_id = keep_id
            gen += 1

            seg = {"agent": agent_id, "gen": gen, "t": [], "n_nascent": [], "max_len": [], "flag": [], "dry_mass": []}
            segments.append(seg)
            prev_idx, prev_len = _record(seg, total, comp.state["agents"][agent_id])
            print(f"  -> following agent={agent_id} into gen={gen}, "
                  f"dry_mass={seg['dry_mass'][-1]:.1f}fg\n")
            continue

        cell = comp.state["agents"][agent_id]
        prev_idx, prev_len = _record(seg, total, cell)
        print(f"t={total:.0f}s  agent={agent_id}  gen={gen}  n_nascent={len(prev_idx)}  "
              f"unique_index={sorted(prev_idx.tolist())}  filament_length={prev_len.tolist()}  "
              f"dry_mass={seg['dry_mass'][-1]:.1f}fg")

    all_ok = all(d["ok"] for d in divisions) if divisions else None
    print(f"\n{len(divisions)} division(s) observed across {gen if gen <= n_generations else n_generations} "
          f"generation(s) attempted.")
    if divisions:
        print(f"{'PASS' if all_ok else 'FAIL'}: divide_nascent_flagellum "
              f"{'correctly partitioned' if all_ok else 'FAILED to correctly partition'} "
              f"the mother's in-progress flagella at EVERY division (cache={cache_dir}).")
        print("\nGeneration lengths (span between successive full samples, real division fell "
              "somewhere in the last --sample window of each):")
        prev_t = 0.0
        for seg in segments[:-1]:
            span = seg["t"][-1] - prev_t
            print(f"  gen {seg['gen']}: {prev_t:.0f}s -> {seg['t'][-1]:.0f}s "
                  f"(span {span:.0f}s = {span/60:.1f} min), "
                  f"dry_mass {seg['dry_mass'][0]:.1f} -> {seg['dry_mass'][-1]:.1f} fg")
            prev_t = seg["t"][-1]
    return {"segments": segments, "divisions": divisions, "ok": all_ok}


def figure(result, cache_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segments = result["segments"]
    divisions = result["divisions"]
    colors = ["#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))

    ax = axs[0]
    for i, seg in enumerate(segments):
        t = np.array(seg["t"]) / 60.0
        ax.plot(t, seg["max_len"], "-o", ms=3, color=colors[i % len(colors)],
                label=f"gen {seg['gen']} (agent {seg['agent']})")
    for d in divisions:
        ax.axvline(d["division_t"] / 60.0, color="#a53f3f", ls="--", lw=1.2)
    ax.set_xlabel("time (min)"); ax.set_ylabel("subunits")
    ax.set_title("Followed lineage: in-progress filament_length")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axs[1]
    for i, seg in enumerate(segments):
        t = np.array(seg["t"]) / 60.0
        ax.plot(t, seg["flag"], "-o", ms=3, color=colors[i % len(colors)],
                label=f"gen {seg['gen']} CPLX0-7452")
    for d in divisions:
        ax.axvline(d["division_t"] / 60.0, color="#a53f3f", ls="--", lw=1.2,
                    label="real division (sampling loop)" if d is divisions[0] else None)
    ax.set_xlabel("time (min)"); ax.set_ylabel("count")
    ax.set_title("Followed lineage: complete flagella count")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axs[2]
    for i, seg in enumerate(segments):
        t = np.array(seg["t"]) / 60.0
        ax.plot(t, seg["dry_mass"], "-o", ms=3, color=colors[i % len(colors)],
                label=f"gen {seg['gen']} dry_mass")
    for d in divisions:
        ax.axvline(d["division_t"] / 60.0, color="#a53f3f", ls="--", lw=1.2)
    ax.set_xlabel("time (min)"); ax.set_ylabel("fg")
    ax.set_title("Followed lineage: dry_mass")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    verdict = "PASS" if result["ok"] else ("FAIL" if result["ok"] is False else "NO DIVISION")
    div_lines = "  |  ".join(
        f"gen{d['gen']}@t~{d['division_t']:.0f}s: mother{d['mother_idx'].tolist()} -> "
        + ", ".join(f"{aid}:{v['idx'].tolist()}" for aid, v in d["daughters"].items())
        for d in divisions
    ) or "no division observed"
    fig.suptitle(
        f"divide_nascent_flagellum multi-generation re-verification ({cache_dir}) -- {verdict}\n{div_lines}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out = f"{STUDY_DIR}/charts/14_division_nascent_flagellum_verify.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    npz_out = f"{STUDY_DIR}/division_nascent_flagellum_verify.npz"
    np.savez(
        npz_out,
        division_times=np.array([d["division_t"] for d in divisions]),
        division_ok=np.array([d["ok"] for d in divisions]),
        n_generations=len(segments),
        **{f"gen{seg['gen']}_t": np.array(seg["t"]) for seg in segments},
        **{f"gen{seg['gen']}_max_len": np.array(seg["max_len"]) for seg in segments},
        **{f"gen{seg['gen']}_flag": np.array(seg["flag"]) for seg in segments},
        **{f"gen{seg['gen']}_dry_mass": np.array(seg["dry_mass"]) for seg in segments},
    )
    print("wrote", npz_out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=float, default=60.0)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10800.0,
                     help="safety cap -- real division under this IC/cache lineage fires ~t=2520s/generation")
    args = ap.parse_args()
    result = run(args.cache_dir, args.seed, args.sample, args.generations, args.seconds)
    figure(result, args.cache_dir)


if __name__ == "__main__":
    main()
