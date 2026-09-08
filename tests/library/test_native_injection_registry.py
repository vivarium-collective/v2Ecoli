"""Tests for the native-injection registry in ``v2ecoli.library.inject``.

STEP 1 of the "inject seam" reconciliation replaces the former hardcoded native
process-class map with a registry (``register_native_injection``) that resolves a
native process class lazily and by absolute import, and that FAILS LOUD by
construction:

* a name that is registered but whose class cannot be imported RAISES
  ``InjectionError`` (no silent fallback to an empty/default config), and
* a built config missing any declared ``required_config_keys`` RAISES.

These are the two behaviors Fable's design turns on. The module must also be
import-clean with no ``sms_modules`` anywhere.
"""
from __future__ import annotations

import sys

import pytest

import v2ecoli.library.inject as inj
from v2ecoli.library.inject import InjectionError, register_native_injection


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot + restore the module-level registry and caches so each test runs
    against a clean registry and does not leak entries/memoized specs."""
    reg = dict(inj._NATIVE_INJECTION_REGISTRY)
    cache = dict(inj._fork_class_cache)
    resolve_cache = dict(inj._RESOLVE_CACHE)
    inj._NATIVE_INJECTION_REGISTRY.clear()
    inj._fork_class_cache.clear()
    inj._RESOLVE_CACHE.clear()
    try:
        yield
    finally:
        inj._NATIVE_INJECTION_REGISTRY.clear()
        inj._NATIVE_INJECTION_REGISTRY.update(reg)
        inj._fork_class_cache.clear()
        inj._fork_class_cache.update(cache)
        inj._RESOLVE_CACHE.clear()
        inj._RESOLVE_CACHE.update(resolve_cache)


class FakeNativeProcess:
    """Minimal pbg_native-looking process: has ``inputs``/``outputs`` (so
    ``classify_process`` returns 'pbg_native') and nothing partitioned."""

    def __init__(self, config=None, core=None):
        self.config = config or {}

    def inputs(self):
        return {}

    def outputs(self):
        return {}


def _register_fake(name, *, required_config_keys=(), topology=None):
    """Register ``name`` -> FakeNativeProcess and pre-seed the fork-class cache so
    ``_import_class`` resolves it WITHOUT importing a real module (keeps the test
    independent of pytest's import mode / sys.path). The registry still records the
    (module_path, class_name) pair; the cache short-circuit is exactly the path
    ``_import_class`` takes for a resolved class."""
    module_path, class_name = "fake_native_pkg.processes", "FakeNativeProcess"
    register_native_injection(
        name, module_path, class_name,
        topology=topology, required_config_keys=required_config_keys)
    inj._fork_class_cache[(module_path, class_name)] = FakeNativeProcess


# --------------------------------------------------------------------------- #
# register + resolve
# --------------------------------------------------------------------------- #
def test_register_and_resolve_returns_the_class():
    _register_fake("fake-proc")
    assert inj._resolve_native_injection("fake-proc") is FakeNativeProcess


def test_unregistered_name_resolves_to_none():
    # Not registered -> None (caller falls through to the fork path).
    assert inj._resolve_native_injection("not-registered") is None


# --------------------------------------------------------------------------- #
# fail loud: registered but un-importable RAISES (not a silent fallback)
# --------------------------------------------------------------------------- #
def test_registered_but_unimportable_class_raises():
    # Registered to a module/class that genuinely cannot be imported, and NOT
    # seeded into the fork-class cache -> _import_class hits importlib and fails.
    register_native_injection(
        "broken-proc",
        "v2ecoli._definitely_not_a_real_module_xyz",
        "Nope",
    )
    with pytest.raises(InjectionError) as excinfo:
        inj._resolve_native_injection("broken-proc")
    msg = str(excinfo.value)
    assert "broken-proc" in msg
    assert "could not be" in msg  # "...class could not be imported..."


def test_registered_but_missing_attribute_raises():
    # Module imports fine (sys is real) but the class attribute is absent ->
    # getattr fails inside _import_class -> InjectionError, still fail-loud.
    register_native_injection("attr-proc", "sys", "NoSuchAttribute")
    with pytest.raises(InjectionError):
        inj._resolve_native_injection("attr-proc")


# --------------------------------------------------------------------------- #
# fail loud: required_config_keys enforced in resolve_injections
# --------------------------------------------------------------------------- #
def _resolve_one(name, process_config):
    """Run resolve_injections fork-free for a single add_process ``name`` whose
    process_configs entry is ``process_config`` (a dict, or 'default')."""
    config = {
        "add_processes": [name],
        "process_configs": {name: process_config},
        "time_step": 1.0,
    }
    return inj.resolve_injections("", config)


def test_required_config_keys_present_passes():
    _register_fake("needs-key", required_config_keys=("homeostatic_concentrations",))
    specs = _resolve_one("needs-key", {"homeostatic_concentrations": {"MET": 1.0}})
    assert len(specs) == 1
    assert specs[0]["name"] == "needs-key"
    assert specs[0]["kind"] == "pbg_native"
    assert specs[0]["config"]["homeostatic_concentrations"] == {"MET": 1.0}


def test_required_config_keys_missing_raises():
    _register_fake("needs-key", required_config_keys=("homeostatic_concentrations",))
    with pytest.raises(InjectionError) as excinfo:
        _resolve_one("needs-key", {"some_other_key": 1})
    assert "homeostatic_concentrations" in str(excinfo.value)


def test_required_config_keys_empty_value_raises():
    # A key that is present but EMPTY (falsy) is treated as missing.
    _register_fake("needs-key", required_config_keys=("homeostatic_concentrations",))
    with pytest.raises(InjectionError) as excinfo:
        _resolve_one("needs-key", {"homeostatic_concentrations": {}})
    assert "homeostatic_concentrations" in str(excinfo.value)


def test_no_required_keys_declared_does_not_gate():
    # No required_config_keys -> a bare add_process may run on defaults.
    _register_fake("free-proc")
    specs = _resolve_one("free-proc", "default")
    assert len(specs) == 1
    assert specs[0]["config"] is None


# --------------------------------------------------------------------------- #
# import-clean: no sms_modules
# --------------------------------------------------------------------------- #
def test_module_imports_without_sms_modules():
    assert "sms_modules" not in sys.modules
    with pytest.raises(ImportError):
        __import__("sms_modules")


def test_source_contains_no_sms_modules_reference():
    with open(inj.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "sms_modules" not in source
