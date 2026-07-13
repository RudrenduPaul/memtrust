#!/usr/bin/env node
"use strict";

/*
 * Fetches and verifies Astral's real `uv` binary for one platform/arch from
 * uv's own pinned 0.11.28 GitHub release, then extracts it into the calling
 * package's bin/ directory. Runs ONLY as a "prepack" lifecycle script --
 * never at end-user npm install time. Verified against the release's own
 * per-archive `<archive>.sha256` file (SHA-256) before anything is
 * extracted; fails loudly on any mismatch.
 *
 * memtrust does not publish its own binary here -- this script redistributes
 * a genuine, unmodified copy of astral-sh/uv, dual-licensed MIT OR
 * Apache-2.0. See the LICENSE-MIT / LICENSE-APACHE files shipped alongside
 * this script's output in each platform package.
 *
 * Usage: node fetch-binary.js <goos> <goarch> <archive-ext> <binary-name>
 */

const https = require("https");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const RELEASE_TAG = "0.11.28";
const RELEASE_BASE = `https://github.com/astral-sh/uv/releases/download/${RELEASE_TAG}`;
const MAX_REDIRECTS = 5;

// uv publishes one archive per Rust target triple, not per goos/goarch pair,
// so the platform package's (goos, goarch) args are mapped to the triple uv
// actually uses in its release asset names.
const TARGET_TRIPLES = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-arm64": "aarch64-unknown-linux-gnu",
  "linux-x64": "x86_64-unknown-linux-gnu",
  "win32-arm64": "aarch64-pc-windows-msvc",
  "win32-x64": "x86_64-pc-windows-msvc",
};

function fail(message) {
  console.error(`fetch-binary: ${message}`);
  process.exit(1);
}

const [, , goos, goarch, ext, binaryName] = process.argv;
if (!goos || !goarch || !ext || !binaryName) {
  fail("usage: node fetch-binary.js <goos> <goarch> <archive-ext> <binary-name>");
}
if (ext !== "tar.gz" && ext !== "zip") {
  fail(`unsupported archive extension "${ext}" (expected "tar.gz" or "zip")`);
}

const platformKey = `${goos}-${goarch}`;
const targetTriple = TARGET_TRIPLES[platformKey];
if (!targetTriple) {
  fail(`no known uv release target triple for "${platformKey}"`);
}

const archiveName = `uv-${targetTriple}.${ext}`;
const archiveUrl = `${RELEASE_BASE}/${archiveName}`;
// uv publishes one checksum file per archive (not one aggregate
// checksums.txt like some goreleaser projects) -- e.g.
// uv-aarch64-apple-darwin.tar.gz.sha256 -- verified by direct download and
// inspection ahead of writing this script.
const checksumUrl = `${archiveUrl}.sha256`;

function get(url, redirectsLeft) {
  if (redirectsLeft === undefined) redirectsLeft = MAX_REDIRECTS;
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: { "User-Agent": "memtrust-npm-fetch-binary" } }, (res) => {
        const status = res.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status) && res.headers.location) {
          res.resume();
          if (redirectsLeft <= 0) return reject(new Error(`too many redirects fetching ${url}`));
          return resolve(get(res.headers.location, redirectsLeft - 1));
        }
        if (status !== 200) {
          res.resume();
          return reject(new Error(`GET ${url} failed: HTTP ${status}`));
        }
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => resolve(Buffer.concat(chunks)));
        res.on("error", reject);
      })
      .on("error", reject);
  });
}

function sha256Hex(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

// uv's per-archive .sha256 files contain exactly one line, in one of two
// observed formats depending on platform:
//   "<64-hex-hash>  <filename>"   (two spaces, no asterisk -- e.g. darwin/linux)
//   "<64-hex-hash> *<filename>"   (one space + asterisk -- e.g. windows, "binary mode")
// Both are handled by treating the asterisk as optional after required
// whitespace, and matching the archive by its basename regardless of format.
function parseChecksumLine(text, expectedFilename) {
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = line.match(/^([0-9a-fA-F]{64})\s+\*?(.+)$/);
    if (!match) continue;
    const hash = match[1].toLowerCase();
    const filename = match[2].trim();
    if (filename === expectedFilename) return hash;
  }
  return null;
}

function extractFromTarGz(tarGzBuf, targetBasename) {
  const tarBuf = zlib.gunzipSync(tarGzBuf);
  let offset = 0;
  while (offset + 512 <= tarBuf.length) {
    const header = tarBuf.subarray(offset, offset + 512);
    if (header.every((b) => b === 0)) break;
    const rawName = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "");
    const sizeField = header.subarray(124, 136).toString("utf8").replace(/\0.*$/, "").trim();
    const size = sizeField ? parseInt(sizeField, 8) : 0;
    const dataStart = offset + 512;
    // uv's tar.gz archives nest the binaries one level deep, e.g.
    // "uv-aarch64-apple-darwin/uv" and "uv-aarch64-apple-darwin/uvx" --
    // matching on basename handles that transparently.
    if (path.basename(rawName) === targetBasename) {
      return tarBuf.subarray(dataStart, dataStart + size);
    }
    const blocks = Math.ceil(size / 512);
    offset = dataStart + blocks * 512;
  }
  return null;
}

function extractFromZip(zipBuf, targetBasename) {
  const EOCD_SIG = 0x06054b50;
  const CENTRAL_DIR_SIG = 0x02014b50;
  let eocdOffset = -1;
  for (let i = zipBuf.length - 22; i >= 0; i--) {
    if (zipBuf.readUInt32LE(i) === EOCD_SIG) { eocdOffset = i; break; }
  }
  if (eocdOffset === -1) throw new Error("zip: end-of-central-directory record not found");
  const entryCount = zipBuf.readUInt16LE(eocdOffset + 10);
  let offset = zipBuf.readUInt32LE(eocdOffset + 16);
  for (let i = 0; i < entryCount; i++) {
    const sig = zipBuf.readUInt32LE(offset);
    if (sig !== CENTRAL_DIR_SIG) throw new Error("zip: malformed central directory entry");
    const method = zipBuf.readUInt16LE(offset + 10);
    const compressedSize = zipBuf.readUInt32LE(offset + 20);
    const nameLen = zipBuf.readUInt16LE(offset + 28);
    const extraLen = zipBuf.readUInt16LE(offset + 30);
    const commentLen = zipBuf.readUInt16LE(offset + 32);
    const localHeaderOffset = zipBuf.readUInt32LE(offset + 42);
    // uv's Windows zip archives store uv.exe / uvw.exe / uvx.exe directly at
    // archive root (no nested directory), unlike the tar.gz layout above --
    // basename matching handles both without special-casing.
    const name = zipBuf.subarray(offset + 46, offset + 46 + nameLen).toString("utf8");
    if (path.basename(name) === targetBasename) {
      const lfhNameLen = zipBuf.readUInt16LE(localHeaderOffset + 26);
      const lfhExtraLen = zipBuf.readUInt16LE(localHeaderOffset + 28);
      const dataStart = localHeaderOffset + 30 + lfhNameLen + lfhExtraLen;
      const compressed = zipBuf.subarray(dataStart, dataStart + compressedSize);
      if (method === 0) return compressed;
      if (method === 8) return zlib.inflateRawSync(compressed);
      throw new Error(`zip: unsupported compression method ${method} for ${name}`);
    }
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return null;
}

async function main() {
  console.log(`fetch-binary: downloading ${archiveUrl}`);
  const [archiveBuf, checksumBuf] = await Promise.all([get(archiveUrl), get(checksumUrl)]);
  const expected = parseChecksumLine(checksumBuf.toString("utf8"), archiveName);
  if (!expected) { fail(`no checksum entry for "${archiveName}" in ${checksumUrl}`); return; }
  const actual = sha256Hex(archiveBuf);
  if (actual !== expected) { fail(`checksum mismatch for ${archiveName}: expected ${expected}, got ${actual}`); return; }
  console.log(`fetch-binary: checksum verified for ${archiveName} (sha256:${actual})`);
  let binaryData;
  try {
    binaryData = ext === "zip" ? extractFromZip(archiveBuf, binaryName) : extractFromTarGz(archiveBuf, binaryName);
  } catch (err) { fail(`failed to extract ${binaryName}: ${err.message}`); return; }
  if (!binaryData || binaryData.length === 0) { fail(`could not find a non-empty "${binaryName}" entry`); return; }
  const outDir = path.join(process.cwd(), "bin");
  const outPath = path.join(outDir, binaryName);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(outPath, binaryData);
  if (process.platform !== "win32") fs.chmodSync(outPath, 0o755);
  console.log(`fetch-binary: wrote verified binary to ${outPath} (${binaryData.length} bytes)`);
}

main().catch((err) => fail(err && err.message ? err.message : String(err)));
