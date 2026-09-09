"""Diagnostic ONLY: temporarily disable motor_complex_assembly (no-op) so
CPLX0-7450 (motor switch complex) can accumulate and actually be observed,
instead of being immediately consumed the same tick it's built. Same
technique as the earlier division-disabled diagnostic this session.

Added 2026-08-11, in response to Maya's question about whether CPLX0-7450
"consumption" was really happening if it always reads 0 -- confirmed it
was (a real, same-tick burst -- motor_switch_assembly built 5 from the
natural FliN=660 starting pool, motor_complex_assembly consumed all 5,
then filament_nucleation immediately claimed 1 of the resulting 5 motor
complexes, all within the very first 2s tick), then this diagnostic was
built to make that transient value actually visible by disabling the
Step that would otherwise consume it same-tick.

IMPORTANT CAVEAT (also discussed with Maya): the burst this diagnostic
reveals (5 built in one shot at t=0) is largely an artifact of this
diagnostic's/study's override initial condition -- it starts from a
large pre-existing standing pool (natural FliN=660) that gets consumed
all at once on the very first tick. A real, continuously-growing cell
would not have this kind of one-time backlog; FliG/FliM/FliN would be
made gradually and C-ring/motor assembly would happen more continuously.
The qualitative point (assembly is fast relative to filament growth) is
a reasonable match to real biology; the literal same-tick instantaneity
and the size of this one burst specifically are not meant as precise
biological claims.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-02-transcription-regulation/diagnostic_cplx7450_isolated.py \
        --seconds 600 --sample 30 --cache-dir out/cache_full_flit_v6
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx
from v2ecoli.processes.flagella_motor_complex_assembly import FlagellaMotorComplexAssembly

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}
READ_IDS = ["CPLX0-7450[i]", "FLAGELLAR-MOTOR-COMPLEX[j]", "CPLX0-7451[j]",
            "FLIN-FLAGELLAR-C-RING-SWITCH[m]", "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]",
            "FLIM-FLAGELLAR-C-RING-SWITCH[i]"]


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    _original_update = FlagellaMotorComplexAssembly.update
    FlagellaMotorComplexAssembly.update = lambda self, states, interval=None: {
        "next_update_time": states["global_time"] + states["timestep"],
    }
    try:
        enable_features("flagella_regulation")
        comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
        enable_features()

        bulk = _arr(comp.state["agents"]["0"]["bulk"])
        bids = bulk["id"]
        for name, val in INIT.items():
            try:
                bulk["count"][bulk_name_to_idx(name, bids)] = val
            except Exception as e:
                print("  (skip IC", name, "->", e, ")")
        idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

        rec = {"t": [], **{k.split("[")[0]: [] for k in READ_IDS}}

        def snap(t):
            b = _arr(comp.state["agents"]["0"]["bulk"])
            rec["t"].append(t)
            for k in READ_IDS:
                rec[k.split("[")[0]].append(int(b["count"][idx[k]]))
            print(f"t={t:6.0f}s  " + "  ".join(f"{k.split('[')[0]}={rec[k.split('[')[0]][-1]}" for k in READ_IDS))

        snap(0)
        total = 0.0
        while total < seconds:
            chunk = min(sample, seconds - total)
            comp.run(chunk)
            total += chunk
            snap(total)
        return {k: np.array(v) for k, v in rec.items()}
    finally:
        FlagellaMotorComplexAssembly.update = _original_update


def figure(rec, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = rec["t"] / 60.0
    fig, axs = plt.subplots(1, 3, figsize=(17.0, 4.7))
    a, b, c = axs

    a.plot(t, rec["CPLX0-7450"], "-o", ms=4, color="#1f77b4", label="CPLX0-7450 (C-ring, now visible)")
    a.plot(t, rec["FLAGELLAR-MOTOR-COMPLEX"], "-s", ms=4, color="#2ca02c", label="FLAGELLAR-MOTOR-COMPLEX (frozen, consumer disabled)")
    a.plot(t, rec["CPLX0-7451"], "--^", ms=3, color="#7f7f7f", alpha=0.6, label="CPLX0-7451 (export apparatus)")
    a.set_title("Motor switch complex, isolated\n(motor_complex_assembly disabled)")
    a.set_xlabel("time (min)"); a.set_ylabel("count"); a.legend(fontsize=8)

    b.plot(t, rec["FLIN-FLAGELLAR-C-RING-SWITCH"], "-o", ms=4, color="#d62728", label="FliN (111 needed/unit)")
    b.axhline(111, color="gray", ls=":", lw=1, label="threshold (111)")
    b.set_title("FliN -- the limiting reagent")
    b.set_xlabel("time (min)"); b.set_ylabel("count"); b.legend(fontsize=8)

    c.plot(t, rec["FLIG-FLAGELLAR-SWITCH-PROTEIN"], "-o", ms=3, color="#9467bd", label="FliG")
    c.plot(t, rec["FLIM-FLAGELLAR-C-RING-SWITCH"], "-s", ms=3, color="#8c564b", label="FliM")
    c.set_title("FliG / FliM (not limiting -- plenty of headroom)")
    c.set_xlabel("time (min)"); c.set_ylabel("count"); c.legend(fontsize=8)

    fig.suptitle(f"CPLX0-7450 isolated diagnostic (motor_complex_assembly disabled) -- {seconds}s ({seconds/60:.0f} min)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/13_cplx7450_isolated_{seconds}s.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    figure(rec, args.seconds)
    np.savez(f"{STUDY_DIR}/cplx7450_isolated_{args.seconds}s.npz", **rec)
    print(f"wrote {STUDY_DIR}/cplx7450_isolated_{args.seconds}s.npz")


if __name__ == "__main__":
    main()
