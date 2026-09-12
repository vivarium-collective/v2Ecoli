# v2ecoli Native Analyses — Status & Roadmap

**Date:** 2026-06-09
**PRs:** framework + 5 ports in `main` (orig. PR #144, closed/superseded); bulk ports in **PR #152**.

This document records what the native-analyses effort delivered, what was deliberately left out, and the roadmap — in particular for the original goal of **offering ptools as a visualization in the v2ecoli dashboard**.

---

## TL;DR

vEcoli's DuckDB/`sim_data` analyses now run **natively inside v2ecoli** as process-bigraph `Analysis` steps — no vEcoli `Analysis`/`plot()` wrapper, no viva-api/HPC. **24 analyses are ported and tested.** The **ptools omics data pipeline works end-to-end** (v2ecoli sim → EcoCyc-frame-ID × timepoint TSVs in the exact Pathway-Tools format). The **ptools Cellular-Overview visualization is now wired** via the `ptools_overview` step, which turns those TSVs into upload-ready Omics Viewer datasets and launches the painted EcoCyc map against a running `sms-ptools` server (see Roadmap Tier 3 + the appendix on running the server).

---

## What works

### The `Analysis` abstraction (in `main`)
- `Analysis(V2Step)` — sibling of the record-based `AnalysisStep`; declares DuckDB `conn` + scale-scoped `history_sql` + ParCa `sim_data` input ports; emits `{view: html, data: map}`. Shared `ANALYSIS_REGISTRY`.
- Runner (`workflow/analysis_runner.py`) provisions one DuckDB connection + the paired `sim_data` per run, builds a per-scale `history_sql`, writes `data → analysis.json` and `view → <sweep>/viz/*.html`.
- Dashboard integration is **done** (generic, no per-analysis code): vivarium-workbench reads `v2ecoli.workflow.analysis.ANALYSIS_REGISTRY` to list `Analysis` classes in its picker, runs the ones declared in a study's `analyses:` list via a post-run hook (`_run_study_analyses` → `analysis_runner.run_analyses`), and surfaces the resulting `<sweep>/viz/*.html` per study/investigation (`_discover_viz_html_files` / `_discover_investigation_viz_html_files`). Any registered analysis (incl. `ptools_overview`) appears automatically once v2ecoli is importable in the serving venv. See vivarium-workbench `docs/post-run-analyses.md`.

### 24 ported analyses (`Analysis`, DuckDB/`sim_data`)
| Scale | Analyses |
|---|---|
| single (5) | `ptools_rna`, `ptools_rxns`, `ptools_proteins`, `mass_fraction_voronoi`, `mass_fraction_summary_view` |
| multigeneration (8) | `ptools_{rna,rxns,proteins}_multigeneration`, `new_gene_counts`, `replication`, `ribosome_components`, `ribosome_production`, `ribosome_usage` |
| multiseed (6) | `ptools_{rna,rxns,proteins}_multiseed`, `central_carbon_metabolism_scatter`, `protein_counts_validation`, `subgenerational_expression_table` |
| multivariant (5) | `average_monomer_counts`, `cell_mass`, `doubling_time_hist`, `doubling_time_line`, `dummy` |

(Plus the 5 pre-existing record-based `AnalysisStep`s: `mass_fraction_summary`, `daughter_mass_symmetry`, `mass_growth_across_generations`, `doubling_time_distribution`, `metric_across_variants`.)

Output types: the **ptools** modules emit **data** (frame-ID × timepoint TSV); the **plot/table** analyses emit a **rendered HTML `view`** (matplotlib→SVG or Altair/Vega-Lite).

### Supporting infrastructure
- **`_shims.py`** — bridges v2ecoli's parquet schema to vEcoli's: `bulk__id`/`bulk__count` → reindexed bulk matrix; `active_ribosome`/`oriC`/`active_RNAP` derived columns.
- **`_helpers.collapse_cross_seed`** — multiseed analyses sum list columns element-wise across seeds (v2ecoli's pandas `groupby.sum` would concatenate).
- **`_wholecell_compat.py`** — one shim for the vendored `wholecell` plotting utils' matplotlib-3.10 incompatibilities (Voronoi etc.).
- **Validation data** — Schmidt 2015 / Wisniewski 2014 copied from vEcoli into `v2ecoli/validation/ecoli/flat/`; minimal loader `v2ecoli/library/validation_data.py`; `resolve_validation_data()` wired into the runner.

### Verified correctness (highlights)
- `ptools_rna` output matches the viva-api oracle format (`EG#####` frame IDs; EG10002 anchored at 1.0).
- Reaction analyses align FBA fluxes **1:1** with `base_reaction_ids` and **assert** on mismatch (no silent truncation — caught a real ~2687-reaction mislabel in review).
- Cross-seed sums verified exact against a real 2-seed sweep (e.g. active_ribosome 12801+12802=25603; array length stays 16321, no concatenation artifact).
- `protein_counts_validation`: log10 Pearson r = **0.74** (Schmidt) / **0.61** (Wisniewski) vs proteomics — a real, sensible validation.
- **61 tests** green across the analysis suite.

### Critical conventions (don't trip on these)
- **sim_data ↔ parquet pairing:** an analysis must use the `sim_data` the sweep ran with. The compare_harness sweep pairs with `out/workflow/simData.cPickle` (2820 `base_reaction_ids`), **not** `out/kb/simData.cPickle` (2821). `resolve_sim_data` no longer matches `kb/`.
- **Fail loud:** never truncate/pad/concatenate where you mean to align or sum; raise `ValueError` on shape/length mismatch.
- Run via `.venv/bin/python` / `.venv/bin/pytest` (bare `python` lacks `unum`).

---

## Not done (deliberate)

- **Emit-trim analyses — descoped (maintainer: not needed):** `ribosome_crowding`, `rna_decay_03_high`, `ribosome_spacing` (+ `new_gene_translation_efficiency_heatmaps`, which also needs an `exp_trl_eff` variant grid). Their listeners (`rna_degradation_listener.count_RNA_degraded_per_cistron`, `ribosome_data.{target,actual}_prob_translation_per_transcript`, `ribosome_init_event_per_monomer`) **are computed** by the processes but **not emitted** — v2ecoli's feature-based emit (`composites/baseline.py:183`, `feat['listeners']`) was deliberately trimmed for clutter (176 cols vs vEcoli's 231). Unblocking = re-declare those listeners in the emit features + re-run a sweep.
- **`ecocyc_table`** (multiseed) — deferred for size (large multi-TSV); its data blockers (validation_data + active_ribosome shim) are now resolved, so it's portable when wanted.
- **`blame`** (single) — permanent skip: vEcoli's upstream `plot()` is `raise NotImplementedError`.

---

## Roadmap

### A. ptools as a dashboard visualization (the original goal)
The ptools data pipeline is done; the *visualization* is not. In increasing depth:
1. **Tier 1 — native render.** Give the ptools ports a `view` (Plotly/Altair heatmap of the frame-ID × timepoint matrix), so they render in the dashboard's Visualizations tab instead of returning raw TSV.
2. **Tier 2 — BioCyc annotation.** Resolve `EG#####` frame IDs to readable reaction/compound/pathway names (vEcoli's `biocyc_service` queries `websvc.biocyc.org`; or use the reconstruction flat files offline). Lets the viz group/label by pathway.
3. **Tier 3 — Pathway Tools Omics Viewer. ✓ DONE (`ptools_overview` step).** The
   `ptools_overview` Analysis (single scale; `v2ecoli/workflow/analyses/ptools_overview.py`)
   reuses `ptools_rna`/`rxns`/`proteins`, reformats their `$`-indexed TSVs into
   upload-ready Omics Viewer datasets (genes / reactions / proteins, with the `$`
   header commented out and provenance + entity-type headers added), and emits a
   `view` that launches the live Cellular Overview (`celOv.shtml`) and offers a
   one-click download of each dataset plus the exact upload recipe. `data["tsv"]`
   is a combined "mixture" file (one upload paints all three). Ingestion is the
   Omics Viewer's manual file upload (no documented auto-load URL). Verified
   end-to-end against `sms-ptools:0.8.2`: emitted `EG#####`/reaction frame IDs
   resolve in the live ECOLI PGDB. Run the server per the appendix below.

### B. Dashboard surfacing — DONE
The `vivarium-workbench` integration is merged (on its `main`) and is **generic**
— it reads `v2ecoli.workflow.analysis.ANALYSIS_REGISTRY` directly, so no
per-analysis dashboard change is needed:
- **Picker** lists every registered `Analysis` class (`_build_analysis_options`).
- **Post-run hook** runs the analyses declared in a study's `analyses:` list
  (`name` + optional `params`, matching a registry key) after each run, via
  `_run_study_analyses` → `analysis_runner.run_analyses`.
- **Surfacing**: the emitted `<sweep>/viz/*.html` are discovered per study
  (`_discover_viz_html_files`) and per investigation
  (`_discover_investigation_viz_html_files`) and shown in the Visualizations area.

So `ptools_overview` (and any future analysis) shows up automatically. The only
prerequisites are operational: v2ecoli must be importable in the dashboard's
serving venv (cf. the pbg editable-install gap), and — for the painted
overview — the external `sms-ptools` server must be running (see the appendix /
`scripts/ptools_server.sh`). Reference: vivarium-workbench `docs/post-run-analyses.md`.

### C. Other follow-ups
- **`validation_data`:** the current minimal loader covers Schmidt/Wisniewski protein counts only. A fuller `ValidationDataEcoli` port (or adding a `validation/` tier to the **ecoli-sources** package, which is reconstruction-only today) would generalize it.
- **Emit completeness:** if the descoped analyses are ever wanted, re-enable their listeners in the emit features (a clutter trade-off) and re-run.
- **`ecocyc_table`:** port when its size is worth the effort.
- **DRY:** the ptools modules share `build_query`/`read_outputs`; further consolidation into `_helpers.py` is possible.

---

## Appendix: Running the `sms-ptools` Pathway Tools server locally

The Tier 3 viz needs a live Pathway Tools server. It ships as the licensed
`sms-ptools` image on GitHub Container Registry (private to the
`vivarium-collective` org). Verified working on Apple-Silicon macOS via colima.

**TL;DR — once authenticated (step 2 below), just use the helper:**

```sh
scripts/ptools_server.sh up        # idempotent: starts a fresh container, waits until ready
scripts/ptools_server.sh status    # is it up?
scripts/ptools_server.sh restart   # force a clean container (use this, not `docker start` — see gotchas)
```

To **paint a run's ptools TSV** onto the EcoCyc Cellular Overview (auto-loaded,
no manual upload), pass one of the emitted `ptools/*.tsv` (or a `ptools_overview_*`
file) to the launcher:

```sh
scripts/ptools_launch.sh <sweep>/ptools/ptools_rna__<group>.tsv   # class inferred → opens the painted map
```

It stages the file into the server (this image's Omics Viewer only loads
server-local files) and opens `celOv.shtml?omics=t&url=…&class=…&column1=1-N`.

**The dashboard "Launch in Omics Viewer" button** (Analyses tab) now works the
same way for any study with `ptools/*.tsv` exports — see the team setup below.

The manual steps below explain what that script does and why.

### Team setup — using the dashboard "Launch in Omics Viewer" button

Works on any teammate's machine (macOS/Linux/Windows Docker). The button needs a
**local** dashboard + the **local** `sms-ptools` container; it can't work on the
published gh-pages site (no server to build the launch URL). One-time:

1. **Start the PTools server** (mounts this repo's `workspace/` into the
   container — required so PTools can read the omics file off disk; it does *not*
   fetch URLs):
   ```sh
   scripts/ptools_server.sh up        # see auth/runtime prereqs in the appendix below
   ```
2. **Run the dashboard locally**, pointed at this repo:
   ```sh
   vivarium-workbench serve --workspace . --port 8771
   ```
   (Run from a venv where `v2ecoli` + `vivarium-workbench` are installed. The
   `ui.ptools_server_url` and `ui.ptools_data_dir: "/ptools-data"` keys are
   already set in `workspace.yaml`.)
3. Open **http://localhost:8771** → **Analyses** tab → **Launch in Omics Viewer**
   for a study that has `studies/<name>/**/ptools/*.tsv` (e.g. the committed
   `showcase-2-baseline-figures`). To generate TSVs for other studies, run the
   `ptools_rna` / `ptools_rxns` / `ptools_proteins` analyses (config:
   `v2ecoli/configs/ptools_baseline.json`).

How it works: PTools reads the file from its own filesystem, so the launcher
passes `url=/ptools-data/workspace/studies/.../x.tsv` (the mount), not an HTTP
URL. The mount lives only on the running container — `scripts/ptools_server.sh`
adds it automatically; if you ever `docker run` by hand, include
`-v "$PWD/workspace:/ptools-data/workspace:ro"`.

**1. A container runtime.** On macOS without Docker Desktop, colima gives a
headless daemon + the `docker` CLI. **Give the VM ≥ 8 GiB** — the overview's
tile generation OOM-kills Pathway Tools (exit 137) under colima's 2 GiB default:

```sh
brew install colima docker
colima start --cpu 4 --memory 8   # boots the Linux VM + Docker daemon
# (after a reboot: `colima start` again; `brew services start colima` to autostart)
```

**2. Authenticate to GHCR.** The package is private, so you need a **classic**
GitHub PAT with the `read:packages` scope (fine-grained tokens and the `gh`
CLI's default token do **not** work for org-owned packages), and your account
must be granted read access to the package:

```sh
docker login ghcr.io -u <your-github-username>   # paste the classic PAT as the password
```

**3. Pull + run.** The image is **amd64-only**, so on Apple Silicon pass
`--platform linux/amd64` (colima runs it under qemu emulation — startup is slow,
~1 min, because Pathway Tools loads the full EcoCyc KB):

```sh
docker pull  --platform linux/amd64 ghcr.io/vivarium-collective/sms-ptools:0.8.2
docker run -d --platform linux/amd64 --name sms-ptools --restart unless-stopped \
  -e SERVER_HOST_NAME=localhost \
  -p 1555:1555 -p 5008:5008 \
  -v "$PWD/workspace:/ptools-data/workspace:ro" \
  ghcr.io/vivarium-collective/sms-ptools:0.8.2
docker logs -f sms-ptools          # wait for "Starting Pathway Tools WWW server at http://localhost:1555/"
```

The `SERVER_HOST_NAME` override matters: the image defaults to a
Kubernetes-internal hostname (`ptools.sms-api-eks.svc.cluster.local`) that won't
resolve locally; set it to `localhost`.

The `-v "$PWD/workspace:/ptools-data/workspace:ro"` mount matters for the
**dashboard button**: this image's Omics Viewer loads omics data from the PTools
server's *own filesystem* (it does not fetch a remote URL), so the dashboard
launcher passes a server-local path under `/ptools-data` (configured by
`ui.ptools_data_dir`). Without the mount the dashboard button reports
*"No file containing valid data was received."* (The CLI `ptools_launch.sh`
docker-cp's the file in, so it works without the mount.)

**Gotchas (learned the hard way):**
- **OOM.** Browsing the overview triggers on-demand tile generation, which is
  memory-hungry; under a 2 GiB VM the container dies with exit `137` mid-session
  ("Safari can't connect"). Give the VM ≥ 8 GiB (above).
- **Never `docker start` a stopped container.** It reuses the dirty filesystem,
  whose stale Xvfb `/tmp/.X1-lock` makes Pathway Tools fail with `cannot open
  the display: :1` and drop into a Lisp REPL — the web server never starts. To
  restart, **recreate fresh** (`docker rm -f sms-ptools` then `docker run …`, or
  `scripts/ptools_server.sh restart`).
- `--restart unless-stopped` lets the container come back after a daemon/VM
  restart.

**Ports:**
- **1555** — WWW server. Serves the **Cellular Overview Omics Viewer**
  (`/overviewsWeb/celOv.shtml?orgid=ECOLI`), an OpenLayers app that paints a
  tab-delimited time-series data file onto the EcoCyc overview map. This is the
  target for the Tier 3 "painted overview." Returns HTTP 200 once booted.
- **5008** — pythoncyc Python API (its own non-HTTP socket protocol; a plain
  GET/POST yields no HTTP response). Use this for **querying/annotating** the
  PGDB (e.g. resolving `EG#####` frame IDs → names for Tier 2), not painting.

The omics-viewer's expected data format is the same `$`-indexed, frame-ID ×
timepoint TSV the ptools analyses already emit (`<sweep>/ptools/*.tsv`); see
`htdocs/time-series.txt` / `htdocs/sample.dat` in the container for the canonical
omics-viewer example format.

> **Note:** As of 2026-06, neither vEcoli (`ptools_viz` branch) nor viva-api
> contains code that actually feeds the TSVs to this server — both only *emit*
> the TSVs. viva-api orchestrates SLURM jobs and delegates painting to vEcoli,
> but the painting client was never written. The v2ecoli integration is net-new.

---

## Provenance
Spec/plan/parity notes under `docs/superpowers/`. The bulk port was built largely by a headless agent on the persistent mini, then unblocked + consolidated locally. See memory `project_v2ecoli_native_analyses` for cross-session context.
