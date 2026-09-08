# v2ecoli/workflow/report_cards/__init__.py
"""Report cards: concrete ``ReportCardStep`` subclasses + their runner plumbing.

The ``ReportCardStep`` base + ``StudyContext``/``write_card``/``prune`` now live
in ``viva_superpowers.post_sim`` (the post-sim Step family's one home) and are
re-exported here for back-compat with the concrete cards and the runner. A
report card emits a rendered ``view`` (the card HTML) plus ``data`` (the
verdict_json map). Unlike ``Analysis`` — which consumes a live DuckDB
sim-output connection — a report card's input is a ``StudyContext`` (the
study's spec + dir), so cards grade run-free. Subclasses that set ``name``
auto-register in ``REPORT_CARD_REGISTRY``.

``applicable``/``narrows_by_name`` stay defined here rather than re-exported:
they carry v2ecoli-specific eligibility logic (the #439 fix distinguishing a
hand-authored card-NAME allowlist from a machine-generated embed-PATH list —
see ``narrows_by_name``) that the shared ``viva_superpowers.post_sim`` version
does not (yet) implement.

The runner (``scripts/study_report_cards.py``) builds a ``bigraph_schema`` core,
instantiates each registered card, calls ``applies``/``build``, and writes the
``view`` → ``viz/report_card/<name>.html`` and ``data`` → ``<name>.verdict.json``
(the files the dashboard discovers).
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.post_sim import (  # noqa: F401
    REPORT_CARD_REGISTRY,
    ReportCardStep,
    StudyContext,
    prune,
    write_card,
)


def narrows_by_name(declared: Any) -> bool:
    """Whether a study's ``report_cards:`` value is a card-NAME allowlist.

    The key carries two different vocabularies. Hand-authored studies list card
    **names** (``[vs_literature]``) — a genuine allowlist. Comparison studies get
    the key machine-generated as a list of HTML embed **paths** by
    ``scripts/_compare/materialize.py`` (``[f"viz/report_card/{c}.html" ...]``),
    which the investigation summary reads as files to embed. A registry name can
    never equal a path, so treating a path list as an allowlist silently excludes
    *every* registered card — the two vocabularies must not be compared.

    A list narrows by name only when no entry looks like a path. An empty list is
    an explicit "no cards" allowlist and is honored as such.
    """
    if not isinstance(declared, (list, tuple, set)):
        return False
    return not any("/" in str(e) or str(e).endswith(".html") for e in declared)


def applicable(ctx: StudyContext, core, only: "str | None" = None) -> list:
    """Instantiated report-card Steps to emit for a study. If the study spec lists
    `report_cards:` as card NAMES, only those names are eligible; if it holds embed
    paths (machine-generated — see `narrows_by_name`) it is not a name allowlist and
    every registered card stays eligible. A card is emitted when eligible AND its
    applies(ctx) is True. `only` (a name, or None/'all') narrows to a single card.
    `core` is a bigraph-schema core (built once by the caller) to instantiate Steps."""
    declared = ctx.spec.get("report_cards")
    by_name = declared is not None and narrows_by_name(declared)
    want = None if (only in (None, "all")) else {only}
    out = []
    for nm, cls in REPORT_CARD_REGISTRY.items():
        if want is not None and nm not in want:
            continue
        if by_name and nm not in declared:
            continue
        try:
            step = cls({}, core=core)
            if step.applies(ctx):
                out.append(step)
        except Exception:  # noqa: BLE001 — one broken card never aborts selection
            continue
    return out


# Register built-in cards (import for side effect). These modules all exist, so
# import unconditionally — a real import error must surface, not be masked.
from . import acceptance_card  # noqa: E402,F401
from . import acetate_overflow_card  # noqa: E402,F401
from . import genotype_build_integrity_card  # noqa: E402,F401
from . import panel_screen_card  # noqa: E402,F401
from . import tests_card  # noqa: E402,F401
from . import vs_literature_card  # noqa: E402,F401
from . import vs_vecoli_card  # noqa: E402,F401
