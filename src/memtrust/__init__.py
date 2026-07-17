"""memtrust: an independent, reproducible benchmark harness for agent-memory backends."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("memtrust")
except PackageNotFoundError:
    # Not installed at all (e.g. importing directly from a raw source
    # checkout with no `pip install -e .` run) -- there is no installed
    # package metadata to read version() from in this case.
    __version__ = "0.0.0+unknown"
