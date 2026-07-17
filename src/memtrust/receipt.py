"""Cryptographic signing and verification for memtrust's own JSON run output.

This exists to close a specific, narrow credibility gap: a `memtrust run`
JSON report is just a file on disk. Anyone with filesystem or transport
access between the run and a reader can edit a number in it, and nothing
about the file itself would show that. A *signed receipt* fixes that one
problem -- it does not fix, and does not claim to fix, whether the
underlying benchmark numbers are accurate in the first place. See
docs/methodology.md's "Signed receipts" section for the full claim/caveat
split; the short version is repeated in `verify_receipt()`'s docstring
below because that is the function whose return value people will actually
build trust decisions on.

Design:

- `canonicalize()` produces a deterministic byte encoding of a JSON-able
  payload (sorted keys, no extraneous whitespace, UTF-8) so the same
  logical report always signs to the same bytes regardless of dict
  insertion order or formatting.
- A receipt is a small JSON document: the canonicalized payload's SHA-256
  hex digest, an Ed25519 signature over the *canonical payload bytes*
  (not the hash -- Ed25519 handles arbitrary-length messages natively, so
  hashing first buys nothing and only adds a place for the two to
  silently disagree), the signer's public key (raw 32 bytes, base64), an
  ISO-8601 UTC timestamp, and the payload itself, embedded so the receipt
  is self-contained and reproducible without the original report file.
- Verification always requires a public key supplied by the *caller*
  (a file path or an env var) -- never the public key embedded in the
  receipt being verified. Trusting the embedded key would let an attacker
  who can rewrite the payload and signature also rewrite the key field
  and produce an internally-"consistent" but meaningless forgery. The
  embedded key is carried only for display/reference (e.g. "who does this
  receipt claim signed it") and is cross-checked against the caller's
  trusted key as an extra diagnostic, never as the basis for trust.

Generating a keypair:

    memtrust keygen --private-key-out mykey.pem --public-key-out mykey.pub

or, using plain `cryptography`/`openssl` directly (this is exactly what
`memtrust keygen` does under the hood, so either path produces
interoperable PEM files):

    openssl genpkey -algorithm ed25519 -out mykey.pem
    openssl pkey -in mykey.pem -pubout -out mykey.pub
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

#: Bumped only if the receipt's JSON shape changes in a way that would
#: break an older verifier -- new optional fields do not require a bump.
RECEIPT_FORMAT_VERSION = 1

ALGORITHM = "Ed25519"

#: Fallback env var `memtrust verify` reads a trusted public key from when
#: no --public-key path is given on the command line.
PUBLIC_KEY_ENV_VAR = "MEMTRUST_RECEIPT_PUBLIC_KEY"


class ReceiptError(Exception):
    """Raised for malformed receipts, keys, or signing inputs -- never for
    "signature didn't match" (that's a normal, expected verify() outcome
    reported via VerifyResult.valid=False, not an exception)."""


def canonicalize(payload: dict[str, Any]) -> bytes:
    """Deterministic JSON encoding: sorted keys, compact separators, UTF-8.

    Two payloads that are equal as Python dicts always canonicalize to the
    same bytes regardless of original key order. This is what actually
    gets signed and hashed -- never the payload's original on-disk
    formatting, which is not guaranteed stable.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")


def payload_sha256_hex(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the payload's canonical bytes."""
    return hashlib.sha256(canonicalize(payload)).hexdigest()


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a fresh Ed25519 keypair. Callers own persisting it (or not)."""
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def write_keypair(
    private_key_path: Path, public_key_path: Path, *, overwrite: bool = False
) -> None:
    """Generate a keypair and write both halves as PEM files.

    Private key: PKCS8, unencrypted (matches `openssl genpkey -algorithm
    ed25519`). Public key: SubjectPublicKeyInfo (matches `openssl pkey
    -pubout`). Either file produced by the `openssl` commands in this
    module's docstring loads correctly here, and vice versa.
    """
    if not overwrite:
        existing = [p for p in (private_key_path, public_key_path) if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise ReceiptError(f"refusing to overwrite existing file(s): {names} (use --force)")

    private_key, public_key = generate_keypair()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_key_path.write_bytes(private_pem)
    public_key_path.write_bytes(public_pem)
    # Best-effort -- some filesystems (e.g. certain CI/tmpfs mounts) don't
    # support chmod; the file still gets written correctly either way.
    with contextlib.suppress(OSError):
        private_key_path.chmod(0o600)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file (unencrypted PKCS8 or
    traditional format -- whatever `serialization.load_pem_private_key`
    accepts)."""
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ReceiptError(f"{path} does not contain an Ed25519 private key")
    return key


def _public_key_from_text(text: str, *, source: str) -> Ed25519PublicKey:
    stripped = text.strip()
    if not stripped:
        raise ReceiptError(f"{source} is empty")
    if "BEGIN PUBLIC KEY" in stripped:
        key = serialization.load_pem_public_key(stripped.encode("utf-8"))
        if not isinstance(key, Ed25519PublicKey):
            raise ReceiptError(f"{source} does not contain an Ed25519 public key")
        return key
    # Otherwise treat it as the base64 encoding of the raw 32-byte key --
    # the same form receipts embed and `memtrust keygen` never writes to
    # disk on its own, but is convenient for passing via an env var.
    try:
        raw = base64.b64decode(stripped, validate=True)
    except ValueError as exc:
        raise ReceiptError(f"{source} is neither a PEM public key nor valid base64: {exc}") from exc
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise ReceiptError(f"{source} is not a valid Ed25519 public key: {exc}") from exc


def load_public_key(path: Path) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a file -- PEM or raw-base64, either works."""
    return _public_key_from_text(path.read_text(), source=str(path))


def load_public_key_from_env(env_var: str = PUBLIC_KEY_ENV_VAR) -> Ed25519PublicKey:
    value = os.environ.get(env_var)
    if value is None:
        raise ReceiptError(f"environment variable {env_var} is not set")
    return _public_key_from_text(value, source=f"${env_var}")


def resolve_public_key(
    key_path: Path | None, *, env_var: str = PUBLIC_KEY_ENV_VAR
) -> Ed25519PublicKey:
    """Resolve the caller's trusted public key: an explicit --public-key
    path always wins; otherwise fall back to the env var. Raises
    ReceiptError if neither is available -- verification never silently
    trusts the receipt's own embedded key as a substitute."""
    if key_path is not None:
        return load_public_key(key_path)
    return load_public_key_from_env(env_var)


def _public_key_b64(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return base64.b64encode(raw).decode("ascii")


def sign_report(report: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Build a signed receipt for `report`. Returns the receipt as a plain
    dict, ready to be `json.dump`ed."""
    canonical = canonicalize(report)
    signature = private_key.sign(canonical)
    return {
        "memtrust_receipt_version": RECEIPT_FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "signed_at": datetime.now(UTC).isoformat(),
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "public_key": _public_key_b64(private_key.public_key()),
        "signature": base64.b64encode(signature).decode("ascii"),
        "payload": report,
    }


def sign_report_with_keyfile(report: dict[str, Any], private_key_path: Path) -> dict[str, Any]:
    return sign_report(report, load_private_key(private_key_path))


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying a receipt.

    `valid=True` proves exactly two things: (1) `payload` is byte-for-byte
    identical, under canonicalization, to what was signed -- nothing in it
    was altered after signing -- and (2) the signature was produced by the
    holder of the private key matching the trusted public key the caller
    supplied. It proves nothing about whether the benchmark numbers inside
    `payload` are themselves correct, complete, or run against a real
    backend -- that is a separate concern docs/methodology.md covers on
    its own terms, not something a signature can attest to.
    """

    valid: bool
    reason: str
    embedded_key_matches_trusted_key: bool | None = None


def verify_receipt(receipt: dict[str, Any], trusted_public_key: Ed25519PublicKey) -> VerifyResult:
    """Verify `receipt` against `trusted_public_key` (supplied by the
    caller -- never derived from the receipt itself; see module docstring).
    """
    try:
        payload = receipt["payload"]
        payload_sha256 = receipt["payload_sha256"]
        signature_b64 = receipt["signature"]
        embedded_public_key_b64 = receipt.get("public_key")
    except KeyError as exc:
        return VerifyResult(valid=False, reason=f"receipt is missing required field {exc}")

    if receipt.get("algorithm", ALGORITHM) != ALGORITHM:
        return VerifyResult(
            valid=False,
            reason=f"unsupported algorithm {receipt.get('algorithm')!r} (expected {ALGORITHM})",
        )

    canonical = canonicalize(payload)
    recomputed_hash = hashlib.sha256(canonical).hexdigest()
    if recomputed_hash != payload_sha256:
        return VerifyResult(
            valid=False,
            reason=(
                "payload does not match its own recorded payload_sha256 -- the "
                "payload was edited after the receipt was written "
                f"(recomputed {recomputed_hash}, receipt says {payload_sha256})"
            ),
        )

    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except ValueError as exc:
        return VerifyResult(valid=False, reason=f"signature is not valid base64: {exc}")

    embedded_key_matches: bool | None = None
    if embedded_public_key_b64 is not None:
        embedded_key_matches = embedded_public_key_b64 == _public_key_b64(trusted_public_key)

    try:
        trusted_public_key.verify(signature, canonical)
    except InvalidSignature:
        return VerifyResult(
            valid=False,
            reason=(
                "signature does not verify against the supplied trusted public key "
                "-- either the payload was tampered with, or it was signed by a "
                "different key than the one supplied to `memtrust verify`"
            ),
            embedded_key_matches_trusted_key=embedded_key_matches,
        )

    return VerifyResult(
        valid=True,
        reason="signature verified: payload is unaltered and was signed by the supplied key",
        embedded_key_matches_trusted_key=embedded_key_matches,
    )


def verify_receipt_file(
    receipt_path: Path, *, public_key_path: Path | None, env_var: str = PUBLIC_KEY_ENV_VAR
) -> VerifyResult:
    trusted_key = resolve_public_key(public_key_path, env_var=env_var)
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        return VerifyResult(valid=False, reason="receipt file does not contain a JSON object")
    return verify_receipt(receipt, trusted_key)


def receipt_path_for(report_path: Path) -> Path:
    """The conventional receipt filename for a given report path:
    `memtrust-report-2026-07-16.json` -> `memtrust-report-2026-07-16.receipt.json`.
    """
    return report_path.with_name(report_path.stem + ".receipt.json")
