"""Population-based (unpruned, real-division) multi-generation test for the
NFsim-wired flagella complexation pipeline (both daughters kept - population level)

Added 2026-08-19. Companion to run_nfsim_lineage_multigen.py -- that script
prunes to ONE followed daughter at every division (mother-machine-style),
which is compute-cheap but means any in-progress structure that happens to
land on the discarded sibling looks like "loss" to the followed lineage even
though it's still alive in the population. This script answers the actual
question that raised: does flagella completion keep up with a ~40min real
division time at the POPULATION level, not just for one unlucky/lucky
tracked cell? Both daughters are kept at every division; per-snapshot stats
are aggregated (sum/mean) across every currently-live agent.

Real division is exponential (each agent can itself divide), so this is
deliberately capped small: MAX_AGENTS stops the run once the live population
reaches that size, regardless of the --generations/--seconds-cap targets,
to keep NFsim's per-agent subprocess overhead bounded. Start small, per
2026-08-19 discussion -- 2 generations, so at most 4 live agents.

Uses the SAME real Division-step machinery as the lineage script (no manual
state splicing -- see that script's docstring for why that matters, a real
dry_mass-drift bug was found and fixed in an earlier manual-splice
approach). Same standard INIT applied once at t=0 to the initial agent
only: 4 flagella, 0 motor, free FliA=500, FlgM=800 -- daughters inherit
their own divided state automatically, no re-application needed.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/run_nfsim_population_multigen.py \
        --generations 2 --sample 120 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os
import re

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

# Full species list, matching run_nfsim_wired_test.py's TRACK_IDS (2026-08-19
# -- "the 4x4 panels we previously had, at the population level, plus FliS").
# Population value = SUM across every currently-live agent (same convention
# already used for flag/flic before this extension).
TRACK_IDS = {
    "EG10320-MONOMER[c]": "FlhD",
    "MONOMER0-2488[c]": "FlhC",
    "CPLX0-3930[c]": "FlhDC complex",
    "EG11355-MONOMER[c]": "FliA",
    "G369-MONOMER[c]": "FlgM",
    "FLGM-FLIA-CPLX[c]": "FlgM:FliA complex",
    "EG11388-MONOMER[c]": "FliS",
    "FLIS-FLIC-CPLX[e]": "FLIS-FLIC-CPLX",
    "CPLX0-7450[i]": "C-ring",
    "CPLX0-7451[j]": "export apparatus",
    "FLAGELLAR-MOTOR-COMPLEX[j]": "motor complex",
    "EG10321-MONOMER[e]": "free FliC",
    "CPLX0-7452[j]": "complete flagella",
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _agent_stats(cell, idx):
    b = _arr(cell["bulk"])
    nf = _arr(cell["unique"]["nascent_flagellum"])
    mask = nf["_entryState"].view(bool)
    lengths = nf["filament_length"][mask]
    uids = nf["unique_index"][mask] if "unique_index" in nf.dtype.names else np.arange(len(lengths))
    internal = cell.get("nfsim_internal_observables") or {}
    stats = {
        "n_nascent": int(len(lengths)),
        "max_len": int(lengths.max()) if len(lengths) else 0,
        "dry_mass": fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0)),
        "hook_internal": float(internal.get("flagellar_hook", 0.0)),
        "export_apparatus_subunit_internal": float(
            internal.get("flagellar_export_apparatus_subunit", 0.0)),
        # Rod and rod+P-ring (added 2026-09-01): both real reaction stages
        # since the 2026-08-27/28 hook/rod/ring hierarchy fix
        "rod_internal": float(internal.get("flagellar_rod", 0.0)),
        "rod_p_ring_internal": float(
            internal.get("flagellar_rod_with_p_ring", 0.0)),
        # Cumulative "ever formed" for the same 4 internal-only stages
        # (added 2026-09-01, see flagella_nfsim_complexation.py's new
        # gross-positive-delta cumulative block) -- these read flat 0
        # otherwise, same net-delta-within-one-chunk blind spot already
        # fixed for C-ring/export apparatus/motor complex.
        "hook_internal_cumulative": float(
            internal.get("flagellar_hook__cumulative", 0.0)),
        "export_apparatus_subunit_cumulative": float(
            internal.get("flagellar_export_apparatus_subunit__cumulative", 0.0)),
        "rod_cumulative": float(internal.get("flagellar_rod__cumulative", 0.0)),
        "rod_p_ring_cumulative": float(
            internal.get("flagellar_rod_with_p_ring__cumulative", 0.0)),
        "flagella_internal_cumulative": float(internal.get("flagella", 0.0)),
        # Cumulative "total ever formed" for C-ring/export apparatus/motor
        # complex (2026-08-27) -- see flagella_nfsim_complexation.py's
        # _CUMULATIVE_TRACKED_REAL_IDS. Piggybacked on the same
        # internal_observables dict as the 3 no-real-bulk-ID species above,
        # under a distinct "__cumulative" key so it doesn't collide with
        # those species' real, live bulk count (tracked separately below).
        "cring_cumulative": float(internal.get("CPLX0-7450[i]__cumulative", 0.0)),
        "export_apparatus_cumulative": float(
            internal.get("CPLX0-7451[j]__cumulative", 0.0)),
        "motor_complex_cumulative": float(
            internal.get("FLAGELLAR-MOTOR-COMPLEX[j]__cumulative", 0.0)),
        "filaments": list(zip(uids.tolist(), lengths.tolist())),
    }
    for real_id in TRACK_IDS:
        stats[real_id] = int(b["count"][idx[real_id]])
    # Back-compat aliases used elsewhere in this script/older charts.
    stats["flag"] = stats["CPLX0-7452[j]"]
    stats["flic"] = stats["EG10321-MONOMER[e]"]
    return stats


def _snap_population(comp, idx, t_cum):
    agents = comp.state.get("agents", {})
    per_agent = {aid: _agent_stats(cell, idx) for aid, cell in agents.items()}
    n = len(per_agent)
    row = {"t_cum": t_cum, "n_agents": n}
    scalar_keys = (
        list(TRACK_IDS.keys()) + ["flag", "flic", "n_nascent", "max_len", "dry_mass",
        "hook_internal", "export_apparatus_subunit_internal", "flagella_internal_cumulative",
        "rod_internal", "rod_p_ring_internal",
        "hook_internal_cumulative", "export_apparatus_subunit_cumulative",
        "rod_cumulative", "rod_p_ring_cumulative",
        "cring_cumulative", "export_apparatus_cumulative", "motor_complex_cumulative"]
    )
    for key in scalar_keys:
        vals = [v[key] for v in per_agent.values()]
        row[f"{key}_total"] = float(sum(vals))
        row[f"{key}_mean"] = float(sum(vals) / n) if n else 0.0
        row[f"{key}_max"] = float(max(vals)) if n else 0.0
    # Per-filament flat table, keyed by (agent_id, unique_index) -- see
    # figure()'s per-filament panel docstring for why this is a
    # simplification (a filament inherited across a division shows as a
    # new line segment under the new agent_id rather than one continuous
    # line, since divide_nascent_flagellum reassigns it to one daughter).
    filament_rows = []
    for aid, stats in per_agent.items():
        for uid, length in stats["filaments"]:
            filament_rows.append((t_cum, f"{aid}:{uid}", length))
    row["filament_rows"] = filament_rows
    row["per_agent"] = per_agent
    return row


def run_population(n_gens, sample, seconds_cap, seed, cache_dir, max_agents, nfsim_interval=None,
                    media="minimal"):
    enable_features("flagella_nfsim_complexation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed, media=media)
    enable_features()

    if nfsim_interval is not None:
        for path, subtree in comp.step_paths.items():
            if path and path[-1] == "ecoli-flagella-nfsim-complexation":
                instance = subtree.get("instance") if isinstance(subtree, dict) else None
                if instance is not None:
                    instance.interval = float(nfsim_interval)
                    break

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in INIT.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val
    idx = {real_id: bulk_name_to_idx(real_id, bids) for real_id in TRACK_IDS}

    rows = [_snap_population(comp, idx, 0.0)]
    print(f"  t_cum=0s n_agents=1 flag_total={rows[0]['flag_total']:.0f}")

    total = 0.0
    max_gen_seen = 1
    while total < seconds_cap:
        chunk = min(sample, seconds_cap - total)
        comp.run(chunk)
        total += chunk

        agents = comp.state.get("agents", {})
        n_agents = len(agents)
        max_gen_seen = max(len(aid) for aid in agents) if agents else max_gen_seen

        row = _snap_population(comp, idx, total)
        rows.append(row)
        if int(total) % 600 < sample:
            print(f"    t_cum={total:.0f}s ({total/60:.0f}min) n_agents={n_agents} "
                  f"max_gen={max_gen_seen} flag_total={row['flag_total']:.0f} "
                  f"flag_mean={row['flag_mean']:.2f} n_nascent_total={row['n_nascent_total']:.0f} "
                  f"dry_mass_mean={row['dry_mass_mean']:.1f}fg")
            # Per-agent breakdown at the same cadence (added 2026-09-03,
            # tracing why n_nascent_total jumped 1->7 in one sample of an
            # earlier run -- the aggregate line alone can't distinguish a
            # gradual per-lineage buildup from a sudden population-wide
            # event; only the final sample got a per-agent print before this).
            for aid, stats in sorted(row["per_agent"].items()):
                print(f"        agent '{aid}': n_nascent={stats['n_nascent']} "
                      f"max_len={stats['max_len']} flag={stats['flag']} "
                      f"dry_mass={stats['dry_mass']:.1f}fg")

        if n_agents >= max_agents:
            print(f"  reached MAX_AGENTS={max_agents} at t_cum={total:.0f}s ({total/60:.0f}min) — stopping")
            break
        if max_gen_seen > n_gens:
            print(f"  reached generation {max_gen_seen} > target {n_gens} at t_cum={total:.0f}s — stopping")
            break

    return rows


# One color per species, reused across its individual panel and any
# overlay it also appears in (same convention as run_nfsim_wired_test.py).
COLORS = {
    "EG10320-MONOMER[c]": "#2ca02c",
    "MONOMER0-2488[c]": "#17becf",
    "CPLX0-3930[c]": "#bcbd22",
    "EG11355-MONOMER[c]": "#1f77b4",
    "G369-MONOMER[c]": "#d62728",
    "FLGM-FLIA-CPLX[c]": "#9467bd",
    "EG11388-MONOMER[c]": "#8c564b",
    "FLIS-FLIC-CPLX[e]": "#e377c2",
    "CPLX0-7450[i]": "#1f77b4",
    "export_apparatus_subunit_internal": "#e377c2",
    "CPLX0-7451[j]": "#ff7f0e",
    "FLAGELLAR-MOTOR-COMPLEX[j]": "#2ca02c",
    "hook_internal": "#8c564b",
    "rod_internal": "#9467bd",
    "rod_p_ring_internal": "#17becf",
    "flagella_internal_cumulative": "#9467bd",
    "n_nascent": "black",
    "EG10321-MONOMER[e]": "#8c564b",
    "CPLX0-7452[j]": "#d62728",
}


def _plot_filament_panel(ax, rows):
    """One line per (agent_id, unique_index) filament, population-wide.
    A filament inherited across a division shows up as a new line segment
    under the daughter's agent_id rather than one continuous line --
    divide_nascent_flagellum (library/division.py) reassigns each
    in-progress filament to exactly one daughter, so tracking by raw
    unique_index alone would risk merging two UNRELATED filaments that
    happen to share an index in two different agents (each agent's own
    unique_index counter starts from what it inherited at division, so
    collisions across siblings are a real risk) -- (agent_id, uid) avoids
    that at the cost of visually splitting inherited filaments at the
    division boundary."""
    flat_t, flat_key, flat_len = [], [], []
    for row in rows:
        for t_cum, key, length in row["filament_rows"]:
            flat_t.append(t_cum / 60.0)
            flat_key.append(key)
            flat_len.append(length)
    if flat_key:
        flat_t = np.array(flat_t); flat_len = np.array(flat_len)
        flat_key = np.array(flat_key)
        for key in np.unique(flat_key):
            mask = flat_key == key
            ax.plot(flat_t[mask], flat_len[mask], "-o", ms=2, lw=1)
        ax.axhline(5000, color="#888888", ls="--", lw=1, label="target (5,000)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no filaments this run", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_ylabel("subunits")


def figure(rows, n_gens, media="minimal"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    t = np.array([r["t_cum"] for r in rows]) / 60.0
    n_agents = np.array([r["n_agents"] for r in rows])

    def tot(key):
        return np.array([r[f"{key}_total"] for r in rows])

    # Panels, in temporal/logical order -- population context first, then
    # the same regulatory -> FliS protection -> assembly cascade ->
    # terminal-output chain used in run_nfsim_wired_test.py's "full panel"
    # figure, all summed across the live population instead of one agent.
    panels = [
        ("Live agent count (population size)", "__agents__"),
        ("Mean dry mass per agent", "__dry_mass__"),
        ("FlhD, population total", "EG10320-MONOMER[c]"),
        ("FlhC, population total", "MONOMER0-2488[c]"),
        ("FlhDC complex, population total", "CPLX0-3930[c]"),
        ("FliA, population total", "EG11355-MONOMER[c]"),
        ("FlgM, population total", "G369-MONOMER[c]"),
        ("__overlay_regulatory__", None),
        ("FlgM:FliA complex, population total", "FLGM-FLIA-CPLX[c]"),
        ("FliS, population total", "EG11388-MONOMER[c]"),
        ("FLIS-FLIC-CPLX (protected FliC), population total", "FLIS-FLIC-CPLX[e]"),
        ("__overlay_cascade__", None),
        ("C-ring, population total", "CPLX0-7450[i]"),
        ("Export apparatus subunit (internal), population total", "export_apparatus_subunit_internal"),
        ("Export apparatus, population total", "CPLX0-7451[j]"),
        # Motor complex removed 2026-09-01 (Maya's request) -- FLAGELLAR-
        # MOTOR-COMPLEX[j] is real (it's L-ring's own product, same
        # species, not a separate stage), but its cumulative tracker was
        # still stuck flat and rod/P-ring -- two real reaction stages
        # added 2026-08-27/28 -- were never plotted anywhere at all. Swap
        # in the two that were actually missing. Old line kept per
        # standing preserve-old-code rule:
        # ("Motor complex, population total", "FLAGELLAR-MOTOR-COMPLEX[j]"),
        ("Rod (internal), population total", "rod_internal"),
        ("Rod+P-ring (internal), population total", "rod_p_ring_internal"),
        ("Hook (internal), population total", "hook_internal"),
        ("Hook-basal-body complete (internal), population total", "flagella_internal_cumulative"),
        ("nascent_flagellum, population total", "n_nascent"),
        ("__filament__", None),
        ("Free FliC, population total", "EG10321-MONOMER[e]"),
        ("__complete_flagella__", None),
    ]

    # Added 2026-08-27 (Maya's request): C-ring/export apparatus/motor
    # complex are fast-flowing real bulk intermediates whose LIVE count is
    # usually 0-1 even when real throughput is happening -- overlay each
    # with its cumulative "total ever formed" counter (see
    # flagella_nfsim_complexation.py's _CUMULATIVE_TRACKED_REAL_IDS) so a
    # flat live-count line doesn't read as "nothing happened here."
    _cumulative_overlay = {
        "CPLX0-7450[i]": "cring_cumulative",
        "CPLX0-7451[j]": "export_apparatus_cumulative",
        # "FLAGELLAR-MOTOR-COMPLEX[j]": "motor_complex_cumulative",  -- no
        # panel uses this key anymore (Motor complex removed 2026-09-01),
        # dead per standing preserve-old-code rule.
        # Added 2026-09-01: same overlay for the 4 internal-only (no real
        # bulk ID) stages, using the new gross-positive-delta cumulative
        # keys from flagella_nfsim_complexation.py.
        "rod_internal": "rod_cumulative",
        "rod_p_ring_internal": "rod_p_ring_cumulative",
        "export_apparatus_subunit_internal": "export_apparatus_subunit_cumulative",
        "hook_internal": "hook_internal_cumulative",
    }

    overlay_regulatory = ("EG11355-MONOMER[c]", "G369-MONOMER[c]")
    overlay_cascade = ["CPLX0-7450[i]", "export_apparatus_subunit_internal", "CPLX0-7451[j]",
                        "rod_internal", "rod_p_ring_internal", "hook_internal",
                        "flagella_internal_cumulative", "n_nascent", "CPLX0-7452[j]"]

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flat
    # Captured as a plain list BEFORE the main loop below consumes axes_flat
    # (a stateful iterator -- reusing axes_flat again after zip() has walked
    # it returns whatever's left, not the same 22 axes again). used_axes is
    # the real, independent list of the panels actually being plotted.
    used_axes = list(axes_flat)[:len(panels)]

    for ax, (panel_title, key) in zip(used_axes, panels):
        if key == "__agents__":
            ax.plot(t, n_agents, "-o", ms=2, color="#2ca02c")
            ax.set_ylabel("n_agents")
        elif key == "__dry_mass__":
            ax.plot(t, np.array([r["dry_mass_mean"] for r in rows]), "-o", ms=2, color="#1f77b4")
            ax.set_ylabel("fg")
        elif panel_title == "__overlay_regulatory__":
            for k in overlay_regulatory:
                ax.plot(t, tot(k), color=COLORS[k], label=TRACK_IDS.get(k, k))
            ax.set_ylabel("count"); ax.legend(fontsize=7)
            panel_title = "FliA / FlgM overlaid, population totals"
        elif panel_title == "__overlay_cascade__":
            for k in overlay_cascade:
                label = {"export_apparatus_subunit_internal": "export apparatus subunit (internal)",
                         "rod_internal": "rod (internal)",
                         "rod_p_ring_internal": "rod+P-ring (internal)",
                         "hook_internal": "hook (internal)",
                         "flagella_internal_cumulative": "hook-basal-body complete (internal)",
                         "n_nascent": "nascent_flagellum"}.get(k, TRACK_IDS.get(k, k))
                ax.plot(t, tot(k), "-o", ms=2, color=COLORS[k], label=label)
            ax.set_ylabel("count"); ax.legend(fontsize=6, ncol=2)
            panel_title = "Assembly cascade, overlaid, population totals"
        elif panel_title == "__filament__":
            _plot_filament_panel(ax, rows)
            panel_title = "Per-filament elongation (population-wide, one line per filament)"
        elif panel_title == "__complete_flagella__":
            # "mean per agent" line removed 2026-08-27 (Maya's request) -- kept
            # per standing preserve-old-code rule:
            # ax.plot(t, np.array([r["flag_mean"] for r in rows]), "-s", ms=2,
            #         color="#9467bd", label="mean per agent")
            ax.plot(t, tot("CPLX0-7452[j]"), "-o", ms=2, color="#d62728")
            ax.set_ylabel("count")
            panel_title = "Complete flagella — population aggregate"
        elif key in _cumulative_overlay:
            ax.plot(t, tot(key), "-o", ms=2, color=COLORS.get(key, "#333333"), label="live count")
            ax.plot(t, tot(_cumulative_overlay[key]), "-o", ms=2, color="#7f7f7f",
                    ls="--", label="cumulative (ever formed)")
            ax.set_ylabel("count"); ax.legend(fontsize=6)
        else:
            ax.plot(t, tot(key), "-o", ms=2, color=COLORS.get(key, "#333333"))
            ax.set_ylabel("count")
        ax.set_title(panel_title, fontsize=9)

    # Division markers (added 2026-09-01, Maya's request): a vertical
    # dashed line on every panel at each real division event, so
    # population-level jumps/dips can be read directly against when a
    # division actually happened rather than inferred from the
    # "Live agent count" panel alone. Division detected as any row where
    # n_agents increases over the previous row; marked at that row's own
    # t_cum (the first sample AFTER the division, not an interpolated
    # estimate of the exact tick it happened on).
    division_times = t[1:][n_agents[1:] > n_agents[:-1]]
    for ax in used_axes:
        for dt_div in division_times:
            ax.axvline(dt_div, color="#555555", ls=":", lw=1, alpha=0.6, zorder=0)

    # Unused trailing grid slots (panels doesn't evenly fill n_rows*n_cols):
    # used to just hide them (ax.axis("off")), still leaving an empty boxed
    # subplot visible. Changed 2026-08-27 (Maya's request) to actually remove
    # them from the figure instead. Old version kept per standing
    # preserve-old-code rule:
    # for ax in list(np.atleast_1d(axes).flat)[len(panels):]:
    #     ax.axis("off")
    # axes_flat was already fully consumed capturing used_axes above -- get
    # the trailing (unused) axes fresh from axes itself, not from the
    # exhausted iterator.
    for ax in list(np.atleast_1d(axes).flat)[len(panels):]:
        fig.delaxes(ax)
    last_row = (len(panels) - 1) // n_cols
    axes_2d = np.atleast_2d(axes)
    for col in range(n_cols):
        if last_row * n_cols + col < len(panels):
            axes_2d[last_row, col].set_xlabel("time (min)")
    # label_outer() removed 2026-09-01: it strips BOTH x and y tick labels
    # on interior-grid axes, correct only when both axes are shared. Only
    # x (time) is shared here (sharex=True) -- every panel has its own
    # independent y-scale, so label_outer() was silently deleting y-axis
    # numbers from every panel except the leftmost column. sharex's own
    # default behavior already suppresses x-tick labels on non-bottom-row
    # axes; removing label_outer() leaves that intact while restoring
    # every panel's own y-axis numbers.

    fig.suptitle(f"NFsim-driven population test, target {n_gens} generations, media={media} "
                 f"(real division, BOTH daughters kept) — does completion keep up at the "
                 f"population level?")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # Auto-number the output (added 2026-08-25): this used to be a fixed
    # "26_..." path regardless of seed/cache/media, so every run silently
    # clobbered whatever the last run wrote (confirmed: a 6-seed stress test
    # overwrote its own plot 5 times, and separately clobbered a
    # git-committed chart with the same name). Every run now gets its own
    # never-reused chart number instead.
    charts_dir = f"{STUDY_DIR}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    existing = [int(m.group(1)) for f in os.listdir(charts_dir)
                if (m := re.match(r"^(\d+)_", f))]
    next_n = max(existing, default=0) + 1
    out = f"{charts_dir}/{next_n}_nfsim_population_multigen_{n_gens}gen_{media}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seconds-cap", type=int, default=7200,
                     help="hard stop on total simulated time, as a safety ceiling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    ap.add_argument("--max-agents", type=int, default=8,
                     help="hard stop on live population size, to bound NFsim subprocess cost")
    ap.add_argument("--nfsim-interval", type=float, default=None)
    ap.add_argument("--media", type=str, default="minimal",
                     help="growth condition, any key in the cache's dry_mass_inc_dict "
                          "(e.g. minimal, minimal_acetate, minimal_succinate, "
                          "minimal_minus_oxygen, minimal_plus_amino_acids)")
    args = ap.parse_args()
    rows = run_population(args.generations, args.sample, args.seconds_cap, args.seed,
                           args.cache_dir, args.max_agents, nfsim_interval=args.nfsim_interval,
                           media=args.media)
    figure(rows, args.generations, media=args.media)
    last = rows[-1]
    print(f"\nFINAL (t_cum={last['t_cum']:.0f}s / {last['t_cum']/60:.0f}min): "
          f"n_agents={last['n_agents']}  flag_total={last['flag_total']:.0f}  "
          f"flag_mean={last['flag_mean']:.2f}  n_nascent_total={last['n_nascent_total']:.0f}  "
          f"dry_mass_mean={last['dry_mass_mean']:.1f}fg  "
          f"FliS_total={last['EG11388-MONOMER[c]_total']:.0f}  "
          f"FLIS-FLIC-CPLX_total={last['FLIS-FLIC-CPLX[e]_total']:.0f}  "
          f"free_FliC_total={last['EG10321-MONOMER[e]_total']:.0f}")
    for aid, stats in last["per_agent"].items():
        print(f"    agent '{aid}': flag={stats['flag']} n_nascent={stats['n_nascent']} "
              f"max_len={stats['max_len']} dry_mass={stats['dry_mass']:.1f}fg")


if __name__ == "__main__":
    main()
