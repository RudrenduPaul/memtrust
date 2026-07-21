"""memtrust: an independent, reproducible benchmark harness for agent-memory backends."""

from importlib.metadata import PackageNotFoundError, version

# This package is published on PyPI under two distribution names kept in
# lockstep: `memtrust-cli` (canonical) and a mirror literally named
# `memtrust` (which the npm wrapper's `bin/memtrust.js` pins its `uv tool
# run --from` call to -- see CONTRIBUTING.md's Release process section).
# Hardcoding a single name here previously broke the mirror install: a
# `pip install memtrust` environment has no `memtrust-cli` entry in its
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
