"""``run_analyses`` must self-register the built-in analyses.

``run_analyses`` resolves every requested analysis name against
``ANALYSIS_REGISTRY`` and skips (with an ``"unknown analysis"`` warning) any
name that isn't there. Those names only register as a side effect of importing
the analysis modules, and nothing in the workflow run path imports the
``v2ecoli.workflow.analyses`` suite. So before this fix a bare
``python -m v2ecoli.workflow.run`` found the registry empty and silently
dropped EVERY declared built-in analysis. ``_register_builtin_analyses``
(called at the top of ``run_analyses``) closes that gap.

Verified in a fresh subprocess so the registry genuinely starts unpopulated —
importing analysis modules is a cached, process-wide side effect, so an
in-process test would be contaminated by any earlier test that imported them.
"""
import subprocess
import sys
import textwrap


def test_builtin_analyses_absent_until_registered():
    """In a clean interpreter, importing the base ``analysis`` module does NOT
    register the built-in suite; ``_register_builtin_analyses`` does."""
    code = textwrap.dedent(
        """
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        # Base module alone: the built-in suite is NOT yet registered.
        assert "cd1_transcriptomics" not in ANALYSIS_REGISTRY, (
            "precondition failed: something imported the analyses suite before "
            "_register_builtin_analyses was called")
        from v2ecoli.workflow.analysis_runner import _register_builtin_analyses
        _register_builtin_analyses()
        assert "cd1_transcriptomics" in ANALYSIS_REGISTRY
        assert "ptools_rna" in ANALYSIS_REGISTRY
        print("REGISTERED_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "REGISTERED_OK" in r.stdout


def test_register_builtin_analyses_is_idempotent():
    """Calling it twice is safe (the underlying import is cached)."""
    code = textwrap.dedent(
        """
        from v2ecoli.workflow.analysis_runner import _register_builtin_analyses
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        _register_builtin_analyses()
        n1 = len(ANALYSIS_REGISTRY)
        _register_builtin_analyses()
        n2 = len(ANALYSIS_REGISTRY)
        assert n1 == n2, (n1, n2)
        print("IDEMPOTENT_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "IDEMPOTENT_OK" in r.stdout


def test_analyses_package_imports_without_scripts_harness():
    """The comparison analyses import the repo-root ``scripts._compare`` harness
    at import time, but ``scripts/`` is not shipped in the installed v2ecoli
    wheel. So when v2ecoli is a bare dependency without that harness on the path,
    ``import v2ecoli.workflow.analyses`` (and hence ``_register_builtin_analyses``
    / ``run_analyses``) must still succeed — the comparison analyses just don't
    register. Simulated in a subprocess by making ``scripts`` unimportable.
    """
    code = textwrap.dedent(
        """
        import sys
        class _BlockScripts:
            def find_spec(self, name, path, target=None):
                if name == "scripts" or name.startswith("scripts."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None
        sys.meta_path.insert(0, _BlockScripts())
        for m in [m for m in sys.modules if m == "scripts" or m.startswith("scripts.")]:
            del sys.modules[m]

        from v2ecoli.workflow.analysis_runner import _register_builtin_analyses
        _register_builtin_analyses()  # must NOT raise ModuleNotFoundError('scripts')
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        # a normal analysis registered despite scripts being blocked
        assert "cd1_transcriptomics" in ANALYSIS_REGISTRY
        # the scripts-dependent comparison analyses were skipped, not fatal
        assert "comparison_summary" not in ANALYSIS_REGISTRY
        print("IMPORTS_WITHOUT_SCRIPTS_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "IMPORTS_WITHOUT_SCRIPTS_OK" in r.stdout


def test_plugin_analyses_imports_entry_point_modules(monkeypatch):
    """An analysis package advertised under the ``v2ecoli.analyses`` entry-point
    group is imported (its registration side effect fires) even though nothing in
    v2ecoli imports it by name -- the mechanism sms_modules' ptools_metabolites
    needs to reach the dispatch path."""
    import importlib
    import importlib.metadata as md
    from v2ecoli.workflow.analysis_runner import _register_plugin_analyses

    ep = md.EntryPoint(name="demo", value="os.path", group="v2ecoli.analyses")
    monkeypatch.setattr(
        md, "entry_points",
        lambda **kw: [ep] if kw.get("group") == "v2ecoli.analyses" else [],
    )
    imported = []
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module",
        lambda name, *a, **k: (imported.append(name), real_import(name, *a, **k))[1],
    )
    _register_plugin_analyses()
    assert "os.path" in imported


def test_plugin_analyses_broken_plugin_warns_not_fatal(monkeypatch):
    """A plugin whose module cannot import warns and is skipped -- it must never
    take down the built-in suite."""
    import importlib.metadata as md
    import pytest
    from v2ecoli.workflow.analysis_runner import _register_plugin_analyses

    ep = md.EntryPoint(name="broken", value="v2ecoli_no_such_module_zzz",
                       group="v2ecoli.analyses")
    monkeypatch.setattr(
        md, "entry_points",
        lambda **kw: [ep] if kw.get("group") == "v2ecoli.analyses" else [],
    )
    with pytest.warns(UserWarning, match="failed to import"):
        _register_plugin_analyses()  # must not raise
