"""Tests for `memtrust.__version__`'s two-distribution-name fallback chain.

memtrust is published on PyPI under two lockstep names -- `memtrust-cli`
(canonical) and a mirror literally named `memtrust` (see
CONTRIBUTING.md's Release process section for why the mirror exists).
`src/memtrust/__init__.py` runs its version lookup at *import time*, so
these tests reload the module under a monkeypatched
`importlib.metadata.version` to exercise each branch of the fallback
chain, rather than only checking whatever the ambient test environment
happens to have installed.
"""

from __future__ import annotations

import importlib
from typing import NoReturn

import pytest


def _reload_memtrust_with_patched_version(
    monkeypatch: pytest.MonkeyPatch, fake_version: object
) -> str:
    import memtrust

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    importlib.reload(memtrust)
    return memtrust.__version__


def test_version_uses_memtrust_cli_when_that_distribution_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(name: str) -> str:
        if name == "memtrust-cli":
            return "0.3.3"
        raise importlib.metadata.PackageNotFoundError(name)

    resolved = _reload_memtrust_with_patched_version(monkeypatch, fake_version)
    assert resolved == "0.3.3"


def test_version_falls_back_to_memtrust_mirror_when_cli_name_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the real bug: a `pip install memtrust` environment has no
    `memtrust-cli` entry in its installed-package metadata at all, so
    the lookup must fall through to the mirror name rather than jumping
    straight to the "not installed" fallback."""

    def fake_version(name: str) -> str:
        if name == "memtrust":
            return "0.3.3"
        raise importlib.metadata.PackageNotFoundError(name)

    resolved = _reload_memtrust_with_patched_version(monkeypatch, fake_version)
    assert resolved == "0.3.3"


def test_version_falls_back_to_unknown_when_neither_name_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(name: str) -> NoReturn:
        raise importlib.metadata.PackageNotFoundError(name)

    resolved = _reload_memtrust_with_patched_version(monkeypatch, fake_version)
    assert resolved == "0.0.0+unknown"
