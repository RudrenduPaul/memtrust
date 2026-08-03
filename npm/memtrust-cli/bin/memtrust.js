#!/usr/bin/env node
"use strict";
const { spawnSync } = require("child_process");
const path = require("path");
// Pinned to this npm package's own version, not left floating: "uv tool run
// --from memtrust-cli" with no version qualifier always fetches whatever is
// currently newest on PyPI. That decouples what a user actually installed
// (memtrust-cli@X from npm, a fixed point in time) from what runs on their
// machine (PyPI's latest, which changes underneath them and could be a
// compromised or unintended publish) -- a supply-chain determinism gap.
// Pinning to this package's own version keeps the two registries in
// lockstep: bump this package.json's version when PyPI's memtrust-cli ships
// a new release, and every subsequent npm install/npx run resolves to that
// exact pinned release, not whatever is newest at run time.
//
// Fetches from PyPI's "memtrust-cli" project, not a separate "memtrust"
// project -- an earlier version of this script pinned "memtrust", a PyPI
// project name that was never actually published (confirmed live:
// https://pypi.org/pypi/memtrust/json returns 404), which made every
// `npx memtrust-cli` invocation fail with "memtrust was not found in the
// package registry" regardless of platform. "memtrust-cli" is the real,
// live PyPI project, and it registers a `memtrust` console-script entry
// point (see pyproject.toml's [project.scripts]), so "uv tool run --from
// memtrust-cli memtrust <args>" resolves correctly.
const PACKAGE_VERSION = require(path.join(__dirname, "..", "package.json")).version;
const PLATFORM_PACKAGES = {
  "darwin-x64": "@memtrust-cli/darwin-x64",
  "darwin-arm64": "@memtrust-cli/darwin-arm64",
  "linux-x64": "@memtrust-cli/linux-x64",
  "linux-arm64": "@memtrust-cli/linux-arm64",
  "win32-x64": "@memtrust-cli/win32-x64",
  "win32-arm64": "@memtrust-cli/win32-arm64",
};
function fail(message) { console.error(message); process.exit(1); }
const platformKey = `${process.platform}-${process.arch}`;
const pkgName = PLATFORM_PACKAGES[platformKey];
const supportedList = Object.keys(PLATFORM_PACKAGES).join(", ");
if (!pkgName) fail(`memtrust: no prebuilt uv binary available for ${process.platform}/${process.arch}.\nSupported platforms: ${supportedList}.`);
const binaryName = process.platform === "win32" ? "uv.exe" : "uv";
let uvPath;
try {
  uvPath = require.resolve(`${pkgName}/bin/${binaryName}`);
} catch (err) {
  fail(`memtrust: the platform package "${pkgName}" is not installed. Try: npm install memtrust-cli --include=optional`);
}
// uv is a bootstrap tool here, not the final CLI: "uv tool run --from
// memtrust-cli==X.Y.Z memtrust <args>" transparently provisions a Python
// interpreter (if needed) and installs/caches that exact pinned
// memtrust-cli release from PyPI on first use, then runs it.
const result = spawnSync(
  uvPath,
  ["tool", "run", "--from", `memtrust-cli==${PACKAGE_VERSION}`, "memtrust", ...process.argv.slice(2)],
  { stdio: "inherit" }
);
if (result.error) fail(`memtrust: failed to execute uv at ${uvPath}: ${result.error.message}`);
if (result.signal) fail(`memtrust: uv terminated by signal ${result.signal}`);
process.exit(result.status === null ? 1 : result.status);
