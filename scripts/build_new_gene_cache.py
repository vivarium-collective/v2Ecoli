"""Build a NAMED ParCa cache with a heterologous gene set INDUCED.

ParCa inserts a new gene *silent* — its ``rna_expression`` entries are exactly
zero, so nothing is transcribed and nothing is translated. The wired v2ecoli
path (``parca_options.new_genes``) is therefore presence-or-absence only: two
builds that differ in how strongly the construct is expressed are
indistinguishable. This script is the other half. It:

  1. hydrates the live SimulationDataEcoli from a ParCa state file (no ParCa
     re-run — the same fast path as scripts/build_cache.py and
     scripts/build_condition_cache.py),
  2. applies ``v2ecoli.perturbations.build_new_gene_cache``, which sets the
     new-gene expression + translation efficiency on a deep copy and writes the
     bundle via ``save_sim_input``,
  3. writes a NEW-GENE MANIFEST (base state, the resolved per-target values,
     git sha) next to the cache so the induction level a run used is provable.

⚠ The state file MUST have been built with new genes. The shipped fixture
``models/parca/parca_state.pkl.gz`` is a basal fit with no new-gene cistrons and
will fail fast with "no new-gene cistrons in this sim_data". Point ``--state``
at a state produced by ``v2ecoli-parca --new-genes <gene_set>``.

⚠ ``--translation-efficiency`` is a WEIGHT, not an achieved rate — the cached
array is L1-normalised across every monomer, so only ratios survive. See the
:mod:`v2ecoli.perturbations.new_gene_cache` docstring.

One induction level per invocation; a design grid is a loop over invocations
(or over ``build_new_gene_cache`` directly, which is why the deep copy lives in
the library function rather than here).

Usage:
    python scripts/build_new_gene_cache.py \
        --state out/parca-gfp/parca_state.pkl \
        --expression 1e6 --translation-efficiency 1.0
    # -> out/cache-new-genes/ + its new_genes.json manifest

    # per-gene relative weights, paired positionally against the new-gene RNAs
    # / monomers in the order v2ecoli.perturbations.new_gene_indices returns
    python scripts/build_new_gene_cache.py --state ... \
        --expression 1e6 --translation-efficiency 1.0 \
        --rel-exp-adj 1,2,4 --rel-trl-eff-adj 1,1,1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.perturbations import build_new_gene_cache
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)

# No default state: unlike build_condition_cache.py there is no shipped fixture
# that would work here (the committed one carries no new genes), so defaulting
# to it would only produce a confusing failure one step later.
DEFAULT_CACHE_DIR = "out/cache-new-genes"


def _weights(raw: str | None, name: str) -> list[float] | None:
    """Parse a comma-separated relative-weight vector, or None."""
    if raw is None:
        return None
    try:
        return [float(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise SystemExit(f"--{name} must be comma-separated numbers: {exc}")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _pos(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def induction_problems(manifest: dict) -> list[str]:
    """Every reason this new-gene cache is NOT a genuine induction. Empty = ok.

    The whole point of this script (see the module docstring) is that ParCa
    inserts a new gene SILENT -- rna_expression exactly zero. A cache that
    requested induction but whose ``applied`` block came back silent is a basal
    build wearing a genotype's name, bit-indistinguishable from a real one at
    exit 0. Verify the induction actually took, at the one place that can: right
    after the library assigns it, before the manifest is trusted downstream."""
    problems: list[str] = []
    req = manifest.get("requested") or {}
    if not _pos(req.get("expression")):
        problems.append(
            f"requested.expression is {req.get('expression')!r}: an induced "
            f"cache must request a positive expression multiplier")

    applied = manifest.get("applied") or {}
    rna_ids = applied.get("rna_ids") or []
    factors = applied.get("expression_factors") or []
    if not rna_ids:
        problems.append(
            "applied.rna_ids is empty: no new-gene cistron was found to "
            "induce -- the cache is basal under a genotype's name (point "
            "--state at a v2ecoli-parca --new-genes build)")
    if not factors or not all(_pos(f) for f in factors):
        problems.append(
            f"applied.expression_factors {factors!r} are not all positive: the "
            f"new gene was inserted SILENT (rna_expression 0) -- the exact "
            f"silent-basal defect this script exists to prevent")
    return problems


def build(state_path: str, cache_dir: str, expression: float,
          translation_efficiency: float,
          rel_exp_adj: list[float] | None = None,
          rel_trl_eff_adj: list[float] | None = None,
          seed: int = 0,
          media_condition: str | None = None,
          fixed_media: str | None = None,
          verify: bool = True) -> dict:
    # Resolve BEFORE the chdir: `build()` is importable as a function, and a
    # caller passing a relative path would otherwise have it silently resolved
    # against the repo root rather than their own cwd.
    state_path = os.path.abspath(state_path)
    cache_dir = os.path.abspath(cache_dir)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading ParCa state {state_path} ...")
    state = load_parca_state(state_path)
    print(f"[{time.strftime('%H:%M:%S')}] Hydrating sim_data ...")
    sim_data = hydrate_sim_data_from_state(state)

    print(f"[{time.strftime('%H:%M:%S')}] Inducing new genes "
          f"(expression x{expression}, translation efficiency "
          f"{translation_efficiency}) -> {cache_dir} ...")
    result = build_new_gene_cache(
        sim_data, cache_dir,
        expression=expression,
        translation_efficiency=translation_efficiency,
        rel_exp_adj=rel_exp_adj,
        rel_trl_eff_adj=rel_trl_eff_adj,
        seed=seed,
        condition=media_condition,
        fixed_media=fixed_media,
    )
    applied = result["applied"]
    print(f"    RNAs      {applied['rna_ids']} @ {applied['expression_factors']}")
    print(f"    monomers  {applied['monomer_ids']} @ "
          f"{applied['translation_efficiencies']} (as assigned; the cached "
          f"array is L1-normalised, so only ratios survive)")

    manifest = {
        "created_at": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "base_state": state_path,
        "seed": seed,
        "media_condition": media_condition,
        "fixed_media": fixed_media,
        "requested": {
            "expression": expression,
            "translation_efficiency": translation_efficiency,
            "rel_exp_adj": rel_exp_adj,
            "rel_trl_eff_adj": rel_trl_eff_adj,
        },
        # As ASSIGNED to sim_data, not as cached: get_polypeptide_initiation_config
        # normalises the efficiency array, so an "as-cached" number would be
        # cache-relative and would move whenever any other monomer moved.
        "applied": applied,
    }
    manifest_path = os.path.join(cache_dir, "new_genes.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    new-gene manifest -> {manifest_path}")

    # BUILD-TIME INDUCTION GATE. The manifest is written first (so a failed
    # build leaves a diagnosable artifact), then verified. A silent/zero
    # induction raises here -- refusing to hand back a basal cache that would
    # dispatch as a genotype and pass every downstream check at exit 0.
    # --no-verify-induction / verify=False overrides for an intentional
    # zero-induction control.
    if verify:
        problems = induction_problems(manifest)
        if problems:
            raise SystemExit(
                f"\nbuild-time induction gate FAILED for {cache_dir}:\n"
                + "\n".join(f"  - {p}" for p in problems)
                + "\n\nThe manifest was written for inspection but this cache "
                "is NOT a genuine induction. Fix the state/expression, or pass "
                "--no-verify-induction (verify=False) for a deliberate control.\n")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True,
                    help="parca_state.pkl[.gz] from a build WITH new genes "
                         "(v2ecoli-parca --new-genes ...)")
    ap.add_argument("--cache", dest="cache_dir", default=DEFAULT_CACHE_DIR,
                    help=f"output cache dir (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--expression", type=float, required=True,
                    help="multiplier on the baseline new-gene expression")
    ap.add_argument("--translation-efficiency", type=float, required=True,
                    help="efficiency assigned to each new-gene monomer "
                         "(a weight — the cached array is L1-normalised)")
    ap.add_argument("--rel-exp-adj", default=None,
                    help="comma-separated per-RNA relative expression weights "
                         "(default: all 1.0)")
    ap.add_argument("--rel-trl-eff-adj", default=None,
                    help="comma-separated per-monomer relative efficiency "
                         "weights (default: all 1.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--media-condition", default=None,
                    help="ParCa nutrient condition for the initial state / "
                         "doubling time (default basal)")
    ap.add_argument("--fixed-media", default=None,
                    help="media id pinned for the whole run")
    ap.add_argument("--no-verify-induction", dest="verify",
                    action="store_false",
                    help="skip the build-time induction gate (allow a silent / "
                         "zero-induction cache -- only for a deliberate control)")
    args = ap.parse_args()

    build(args.state, args.cache_dir, args.expression,
          args.translation_efficiency,
          rel_exp_adj=_weights(args.rel_exp_adj, "rel-exp-adj"),
          rel_trl_eff_adj=_weights(args.rel_trl_eff_adj, "rel-trl-eff-adj"),
          seed=args.seed,
          media_condition=args.media_condition,
          fixed_media=args.fixed_media,
          verify=args.verify)


if __name__ == "__main__":
    main()
