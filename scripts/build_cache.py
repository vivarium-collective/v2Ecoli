"""Build out/cache/ from the shipped ParCa fixture.

Hydrates models/parca/parca_state.pkl.gz into a SimulationDataEcoli
in memory and emits the simulation-input bundle (initial_state.json,
sim_data_cache.dill, metadata.json, .cache_version) directly — no
intermediate ``simData.cPickle`` round-trip. No ParCa re-run.

Use this whenever:
  - You pulled a branch that changed sim_data.py, the pint boundary,
    or models/parca/parca_state.pkl.gz.
  - Tests or reports abort with StaleCacheError.
  - You removed out/cache/ and want to recreate it.

Usage:
    python scripts/build_cache.py                 # default: out/cache
    python scripts/build_cache.py --cache out/my  # custom destination
    python scripts/build_cache.py --fixture path/to/parca_state.pkl.gz

For a full ParCa re-run (several hours) see docs/generate_full_parca.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import read_cache_version
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)


DEFAULT_FIXTURE = "models/parca/parca_state.pkl.gz"
DEFAULT_CACHE_DIR = "out/cache"


def _normalize_strain(value: str | None) -> str | None:
    """Map wild-type sentinels to ``None`` for build_params consistency.

    ``v2ecoli-parca --new-genes off`` means "no heterologous insertion", i.e.
    the wild-type build, which :data:`cache_version.DEFAULT_BUILD_PARAMS`
    represents as ``None`` (and which the in-process ``core.save_sim_input``
    path records as ``None`` when no strain is passed). Stamping the literal
    ``"off"`` instead would make a wild-type cache's stored ``new_genes`` differ
    from a wild-type request's ``None`` and trip ``verify_cache_version``'s
    wrong-strain check on a cache that is in fact correct. Normalize here so the
    CLI build path stamps the same value the other build paths do.
    """
    if value is None:
        return None
    v = value.strip()
    if v in ("", "off"):
        return None
    return v


def build_cache(fixture: str, cache_dir: str,
                media_condition: str | None = None,
                fixed_media: str | None = None,
                new_genes: str | None = None,
                bundle_overrides: str | None = None) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading fixture {fixture} ...")
    state = load_parca_state(fixture)
    print(f"    loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Hydrating sim_data ...")
    sim_data = hydrate_sim_data_from_state(state)
    print(f"    hydrated in {time.time()-t1:.1f}s")

    if media_condition is not None:
        avail = dict(getattr(sim_data, "condition_to_doubling_time", {}) or {})
        if media_condition not in avail:
            raise SystemExit(f"unknown media_condition {media_condition!r}; "
                             f"known: {sorted(avail)}")
        print(f"    nutrient condition {media_condition!r} "
              f"(doubling {avail[media_condition]}), media {fixed_media!r}")

    t2 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building bundle at {cache_dir} ...")
    # save_sim_input (-> v2ecoli.core._write_sim_input_bundle) already writes
    # a complete cache_version.json — with build_params (condition/
    # fixed_media/seed/n_seeds, T3/A7-A9) and configs (A6) — as its last
    # step. A SECOND write_cache_version(cache_dir, repo_root=repo_root)
    # call here used to re-derive a version with none of that (build_params
    # all None, configs empty) and clobber the correct file with it, purely
    # to have a `version` object to print inputs_hash from. Read the
    # already-written file back instead: same print, no clobber.
    # Declare the ParCa chassis this cache is derived from (schema 3): the
    # fixture pickle becomes the bundle's ``derived_from`` chain, so a swap of
    # the chassis moves ``inputs_hash`` and its provenance sidecar (if one
    # sits beside the fixture) is embedded. A schema-3 cache with no declared
    # chain is rejected by ``verify_cache_version`` (guard a).
    save_sim_input(sim_data, cache_dir,
                   condition=media_condition, fixed_media=fixed_media,
                   new_genes=_normalize_strain(new_genes),
                   bundle_overrides=_normalize_strain(bundle_overrides),
                   sources=[{"layer": "chassis",
                             "path": os.path.abspath(fixture)}])

    version = read_cache_version(cache_dir)
    print(f"    bundle built in {time.time()-t2:.1f}s")
    print(f"    inputs_hash: {version.inputs_hash[:16]}...")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print("Bundle contents:")
    for f in sorted(os.listdir(cache_dir)):
        p = os.path.join(cache_dir, f)
        print(f"  {f}: {os.path.getsize(p)/1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help=f"ParCa fixture pickle (default: {DEFAULT_FIXTURE})")
    parser.add_argument("--cache", default=DEFAULT_CACHE_DIR, dest="cache_dir",
                        help=f"output bundle dir (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--media-condition", default=None,
                        help="ParCa nutrient condition for the initial state / "
                             "doubling time (e.g. acetate; default basal)")
    parser.add_argument("--fixed-media", default=None,
                        help="media id pinned for the run (e.g. minimal_acetate)")
    parser.add_argument("--new-genes", default=None,
                        help="strain new-gene insertion subdir this cache was built "
                             "for (e.g. violacein). Recorded into the bundle's "
                             "cache_version.json build_params so verify_cache_version "
                             "can reject a wrong-strain cache (P1-6). 'off'/empty = "
                             "wild-type. MUST match the value passed to v2ecoli-parca.")
    parser.add_argument("--bundle-overrides", default=None,
                        help="bundle-overrides manifest path this cache was built for. "
                             "Recorded into build_params alongside --new-genes; same "
                             "wrong-strain-guard purpose. MUST match v2ecoli-parca.")
    args = parser.parse_args()
    build_cache(args.fixture, args.cache_dir,
                media_condition=args.media_condition, fixed_media=args.fixed_media,
                new_genes=args.new_genes, bundle_overrides=args.bundle_overrides)


if __name__ == "__main__":
    main()
