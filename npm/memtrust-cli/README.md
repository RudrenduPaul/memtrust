# memtrust-cli

npx-runnable wrapper for [memtrust](https://github.com/RudrenduPaul/memtrust), an open agent-memory
eval harness that benchmarks memory backends (MemPalace, Mem0, Zep/Graphiti, OpenViking) against
real failure modes — silent write loss, contradiction handling, ranking degradation, and more.

This package does not implement memtrust itself. It bootstraps a genuine, SHA-256-verified copy of
[Astral's `uv`](https://github.com/astral-sh/uv) for your platform, then uses it to fetch and run
the real `memtrust` package from PyPI — no separate Python toolchain to install by hand.

## Usage

```bash
npx memtrust-cli --help
```

or, if installed globally:

```bash
npm install -g memtrust-cli
memtrust --help
```

The first run downloads `memtrust` from PyPI via `uv tool run --from memtrust memtrust` and caches
it; subsequent runs reuse the cache.

## Supported platforms

macOS (arm64/x64), Linux (arm64/x64), Windows (arm64/x64) — the matching platform binary is pulled
in automatically via `optionalDependencies`.

## Links

- memtrust: https://github.com/RudrenduPaul/memtrust
- memtrust on PyPI: https://pypi.org/project/memtrust/
- uv (bundled bootstrap tool, not affiliated with or endorsed by Astral Software Inc.): https://github.com/astral-sh/uv

## License

Apache-2.0
