"""Step 10 — write the simulation-input bundle.

Persists the live ``SimulationDataEcoli`` the ParCa step network leaves in the
``sim_data_root`` port (written by ``FinalAdjustmentsStep``) into a cache bundle
-- ``initial_state.json`` + ``sim_data_cache.dill`` + ``cache_version.json`` --
via ``v2ecoli.core.save_sim_input``.

Without this step the nine ParCa steps leave a live sim_data in the store and
persist nothing (persistence lives in the CLI, ``v2ecoli/cli/parca.py``; the
standalone equivalent is ``scripts/build_cache.py``). Adding it makes ParCa a
genuine in-document step network whose *output* is a content-addressed cache
bundle -- the piece the pbg-native one-document run (sms-ecoli #166) needs so a
ParCa document can produce the very cache the lineage document then reads.

The bundle's identity params (``new_genes`` / ``bundle_overrides`` /
``bundle_manifest`` / ``perturbations``) are folded into ``cache_version.json``'s
``inputs_hash`` by ``save_sim_input``, so a bundle built for one genotype cannot
be silently reused for another -- ``verify_cache_version`` rejects a mismatch at
stage time.
"""
from __future__ import annotations

from typing import Any

from process_bigraph import Step

# ``sim_data_root`` is the live SimulationDataEcoli the ParCa chain overwrites in
# place; every producing step declares it 'overwrite', so this consumer does too.
INPUT_PORTS = {
    'sim_data_root': 'overwrite',
}

# The written bundle directory, surfaced so a downstream step / the caller can
# read where it landed (and so the write is observable in the document state).
OUTPUT_PORTS = {
    'bundle_dir': 'overwrite',
}


def _or_none(value: Any) -> Any:
    """Treat empty-string config defaults as "unset" for save_sim_input, whose
    identity params expect ``None`` (not '') when absent."""
    return value or None


class SimInputWriteStep(Step):
    """Persist the ParCa chain's ``sim_data_root`` as a cache bundle."""

    config_schema = {
        'cache_dir': {'_type': 'string', '_default': 'out/cache'},
        'seed': {'_type': 'integer', '_default': 0},
        'condition': {'_type': 'string', '_default': ''},
        'fixed_media': {'_type': 'string', '_default': ''},
        'new_genes': {'_type': 'string', '_default': ''},
        'bundle_overrides': {'_type': 'string', '_default': ''},
        'bundle_manifest': {'_type': 'string', '_default': ''},
        # Non-string / structured identity input (e.g. a design-variant dict);
        # threaded verbatim into cache_version's inputs_hash.
        'perturbations': {'_default': None},
    }

    def inputs(self) -> dict:
        return dict(INPUT_PORTS)

    def outputs(self) -> dict:
        return dict(OUTPUT_PORTS)

    def update(self, state: dict) -> dict:
        # The REAL SimulationDataEcoli (not the attr-proxy make_sim_data_facade
        # builds for the fit steps) -- save_sim_input dills this object into
        # sim_data_cache.dill, so it must be the genuine instance.
        sim_data = state['sim_data_root']
        cache_dir = self.config['cache_dir']

        from v2ecoli.core import save_sim_input

        save_sim_input(
            sim_data,
            bundle_dir=cache_dir,
            seed=int(self.config.get('seed', 0)),
            condition=_or_none(self.config.get('condition')),
            fixed_media=_or_none(self.config.get('fixed_media')),
            new_genes=_or_none(self.config.get('new_genes')),
            bundle_overrides=_or_none(self.config.get('bundle_overrides')),
            bundle_manifest=_or_none(self.config.get('bundle_manifest')),
            perturbations=self.config.get('perturbations'),
        )
        return {'bundle_dir': cache_dir}
