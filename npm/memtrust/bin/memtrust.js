#!/usr/bin/env node
"use strict";
const { spawnSync } = require("child_process");
const PLATFORM_PACKAGES = {
  "darwin-x64": "@memtrust/darwin-x64",
  "darwin-arm64": "@memtrust/darwin-arm64",
  "linux-x64": "@memtrust/linux-x64",
  "linux-arm64": "@memtrust/linux-arm64",
  "win32-x64": "@memtrust/win32-x64",
  "win32-arm64": "@memtrust/win32-arm64",
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
  fail(`memtrust: the platform package "${pkgName}" is not installed. Try: npm install memtrust --include=optional`);
}
// uv is a bootstrap tool here, not the final CLI: "uv tool run --from memtrust
// memtrust <args>" transparently provisions a Python interpreter (if needed)
// and installs/caches memtrust from PyPI on first use, then runs it.
const result = spawnSync(
  uvPath,
  ["tool", "run", "--from", "memtrust", "memtrust", ...process.argv.slice(2)],
  { stdio: "inherit" }
);
if (result.error) fail(`memtrust: failed to execute uv at ${uvPath}: ${result.error.message}`);
if (result.signal) fail(`memtrust: uv terminated by signal ${result.signal}`);
process.exit(result.status === null ? 1 : result.status);
