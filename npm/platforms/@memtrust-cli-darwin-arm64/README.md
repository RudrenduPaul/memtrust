# @memtrust-cli/darwin-arm64

macOS arm64 binary for [memtrust-cli](https://github.com/RudrenduPaul/memtrust)'s npx wrapper.

This package is not meant to be installed directly. Install the main
[`memtrust-cli`](https://www.npmjs.com/package/memtrust-cli) package instead; it
pulls in this platform binary automatically through `optionalDependencies` based
on your OS and CPU architecture.

## What's inside

A genuine, SHA-256-verified copy of [Astral's uv](https://github.com/astral-sh/uv),
fetched from uv's own GitHub release 0.11.28 at publish time. uv is dual-licensed
MIT OR Apache-2.0; both license texts are reproduced in this package's LICENSE
file. Not affiliated with or endorsed by Astral Software Inc.

## Documentation

Full documentation, usage, and benchmarks live in the main
[memtrust repository](https://github.com/RudrenduPaul/memtrust).

## License

Apache-2.0 for memtrust's own wrapper code. See LICENSE for the bundled uv
binary's MIT OR Apache-2.0 terms.
