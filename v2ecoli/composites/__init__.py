"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``viva_superpowers.composite_generator._REGISTRY``.
"""

import dataclasses as _dataclasses

from viva_superpowers.composite_generator import _REGISTRY as _COMPOSITE_REGISTRY

from v2ecoli.composites import (  # noqa: F401
    ecoli_baseline,
    ecoli_millard,
    ecoli_population,
    ecoli_time_varying_env,
    ecoli_colony,
    lineage_ray_batch,
    millard_fba_bridge_harness,
    parca,
    parca_prep,
    reactor_bird_coupled,
    reactor_bird_coupled_millard,
    vecoli,
    workflow_nf,
)


# --- Alias registration -----------------------------------------------------
# A generator ids as ``{__module__}.{name}``; because module slug == composite
# name (e.g. ``ecoli_baseline`` in ``ecoli_baseline.py``), that id DOUBLES to
# ``v2ecoli.composites.ecoli_baseline.ecoli_baseline``. The dashboard resolver
# matches on the trailing segment after ``.composites.``, so studies that
# reference the short ``ecoli_baseline`` need a clean-id alias whose trailing
# segment is exactly that. All alias keys point at the identical
# ``GeneratorEntry.func``, so ``build_composite`` dedupes them by function
# identity rather than raising "ambiguous architecture name".
def _register_alias(alias_id: str, source_id: str) -> None:
    orig = _COMPOSITE_REGISTRY.get(source_id)
    if orig is not None and alias_id not in _COMPOSITE_REGISTRY:
        _COMPOSITE_REGISTRY[alias_id] = _dataclasses.replace(orig, id=alias_id)


def _register_clean_alias(name: str, module: str | None = None) -> None:
    module = module or name
    _register_alias(f"v2ecoli.composites.{name}",
                    f"v2ecoli.composites.{module}.{name}")


# The ecoli_* whole-cell family was renamed from the legacy ``baseline*`` /
# ``colony`` scheme (2026-07). Every in-repo reference (workspace studies,
# scripts, tests, committed composite-state snapshots) was updated to the new
# ids in the same change, so no legacy-id alias is registered — a stale
# ``baseline`` id resolving silently would only hide a missed reference.
for _new in (
    "ecoli_baseline",
    "ecoli_millard",
    "ecoli_population",
    "ecoli_time_varying_env",
    "ecoli_colony",
    "lineage_ray_batch",
    "parca",
    "reactor_bird_coupled",
    "reactor_bird_coupled_millard",
    "vecoli",
):
    _register_clean_alias(_new)


__all__ = [
    "ecoli_baseline",
    "ecoli_millard",
    "ecoli_population",
    "ecoli_time_varying_env",
    "ecoli_colony",
    "lineage_ray_batch",
    "millard_fba_bridge_harness",
    "parca",
    "reactor_bird_coupled",
    "reactor_bird_coupled_millard",
    "vecoli",
]
