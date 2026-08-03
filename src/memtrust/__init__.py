"""memtrust: an independent, reproducible benchmark harness for agent-memory backends."""

from importlib.metadata import PackageNotFoundError, version

# This package is published on PyPI as `memtrust-cli` (canonical, the only
# name this project actually publishes -- see CONTRIBUTING.md's Release
# process section). A `memtrust`-named PyPI project does not currently
# exist (verified live: pypi.org/pypi/memtrust/json returns 404); the
# `memtrust` lookup below is a defensive fallback only, for the rare case
# of a local/manual install under that distribution name, not evidence
# such a project is published. Hardcoding a single name here previously
# broke that fallback case: an environment with only a `memtrust`-named
# distribution installed has no `memtrust-cli` entry in its
# installed-package metadata at all, so `version("memtrust-cli")` always
# raised PackageNotFoundError and silently fell through to the
# "not installed" fallback below, even though the package genuinely was
# installed and importlib.metadata genuinely did have its real version
# on file under the other name. Try both; only fall back if neither
# distribution name is registered.
try:
    __version__ = version("memtrust-cli")
except PackageNotFoundError:
    try:
        __version__ = version("memtrust")
    except PackageNotFoundError:
        # Not installed at all (e.g. importing directly from a raw source
        # checkout with no `pip install -e .` run) -- there is no installed
        # package metadata to read version() from in this case.
        __version__ = "0.0.0+unknown"
