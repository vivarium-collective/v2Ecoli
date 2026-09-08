"""Byte-identity tripwire for the ecoli-sources bundle.

Once the local ``flat/`` tree is deleted, the legacy-vs-bundle parity tests
(``test_kb_bundle_parity``, ``test_inherited_keys_match_ecoli_sources_content``)
skip — so nothing else actively guards the migration's core promise: that ParCa
reads the exact bytes parity was validated against.

This test pins SHA-256 of a few representative inherited keys (served from the
rev-pinned ``ecoli-sources`` package) and the 3 v2ecoli override keys (served
from local ``flat_overrides/``). It does NOT skip. If someone bumps the
``ecoli-sources`` pin in ``pyproject.toml`` to content-different data, or
changes an override file, this fails loudly — forcing a deliberate
re-validation of ParCa parity and an update of these pins.

It also asserts the source split is correct: override keys resolve into
``flat_overrides/``; inherited keys resolve out of the installed ecoli-sources
package (NOT ``flat_overrides/``).
"""

import hashlib

from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

# SHA-256 of each key's resolved file, captured against the ecoli-sources rev
# pinned in pyproject.toml at migration time. Update deliberately (and re-verify
# ParCa parity) if you bump the pin.
EXPECTED_SHA256 = {
    # Inherited from ecoli-sources:
    "genes": "a1273d3ac869e3927f02c2f29cd66cccef0cafdc656dd0a7354f1c035c50119e",
    "rnas": "bddcd413a0ebb04b912844bde21bb189047d128a7629cbd409a7166158541f53",
    "metabolites": "b87e39d72a33954f3f564db39b75810941204f1e22ebeaf077e412460157b864",
    "sequence": "28a7167d8bab60570cd6e3ddacdde75d5c51db60e1009fb8c241f1da446b6152",
    # Condition/media inputs — added for #584 (part 2). The docstring promises a
    # content-different pin bump fails loudly, but #581 changed exactly these
    # files (condition_defs, media_recipes) and the test stayed green because
    # they were unpinned. They also drive the genotype-card condition count
    # (len(condition_defs) + 2*len(tf_condition)); pin them so a data change is
    # caught here rather than silently shifting graded references downstream.
    # Re-pinned 2026-09-08 merging origin/main (71 commits): main changed
    # these two upstream ecoli-sources files (not a v2ecoli-local override,
    # not flagella-cascade-related) between the previous pin and this merge.
    "condition__condition_defs": "d28d181e2e7674101b47ccfaca21d7e119571ab902f9370723e671656e7f8844",
    "condition__media_recipes": "d49b9872946a0f7462e068d359337c41720dafb3ea8f5110567cf4c7feca5d67",
    "condition__tf_condition": "fbcfbefdcffde74a380b13c1cf0b4c7d2cab28b312a7cb7b841459bc18460099",
    # v2ecoli local overrides (diverged biology, must win). Re-pinned 2026-09-04
    # merging origin/main: both branches had independently changed these two
    # files (this branch: flagella-cascade's FLGM-FLIA-CPLX_RXN row, FlgM
    # anti-sigma sequestration of FliA/sigma28; main: an unrelated upstream
    # change) -- neither side's pinned hash reflected the merged content, so
    # these are freshly computed directly against the post-merge working tree
    # files, not copied from either side of the conflict.
    "equilibrium_reactions": "d10247d4dfc31da7d0293aa3bd2d16bbf2493891693b83435718384f0404febb",
    "equilibrium_reaction_rates": "74c7216898625b1d616d814aeb254489c7ede6ced6cc6529aecfab573a225fb1",
    "metabolic_reactions_added": "97765aafda3c76445caedb887f96ec874a0defa7004d94e1083e89dcf68b0271",
}

OVERRIDE_KEYS = {
    "equilibrium_reactions",
    "equilibrium_reaction_rates",
    "metabolic_reactions_added",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_content_pins():
    bundle = SourceBundle()
    mismatches = []
    for key, expected in EXPECTED_SHA256.items():
        actual = _sha256(bundle.path(key))
        if actual != expected:
            mismatches.append(f"{key}: expected {expected[:12]}.., got {actual[:12]}..")
    assert mismatches == [], (
        "Bundle content changed (ecoli-sources pin bump or override edit?). "
        "Re-validate ParCa parity, then update EXPECTED_SHA256:\n  "
        + "\n  ".join(mismatches)
    )


def test_override_keys_resolve_local_and_inherited_keys_resolve_upstream():
    bundle = SourceBundle()
    for key in EXPECTED_SHA256:
        resolved = str(bundle.path(key))
        if key in OVERRIDE_KEYS:
            assert "flat_overrides" in resolved, f"{key} should be a local override"
        else:
            assert "flat_overrides" not in resolved, (
                f"{key} should come from ecoli-sources, not a local override"
            )
