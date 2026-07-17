"""Unit tests for src/memtrust/receipt.py -- real Ed25519 signing and
verification, no mocking of the cryptography primitives themselves. Every
test in this file exercises the actual `cryptography` library end to end:
a real keypair is generated, a real signature is produced, and a real
signature-verification call is made. Only the *contents* being signed
(a sample run-shaped report dict) are test fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memtrust.receipt import (
    ReceiptError,
    canonicalize,
    generate_keypair,
    load_public_key,
    payload_sha256_hex,
    receipt_path_for,
    resolve_public_key,
    sign_report,
    verify_receipt,
    verify_receipt_file,
    write_keypair,
)

SAMPLE_REPORT = {
    "run_id": "mt_2026-07-16T120000Z",
    "memtrust_version": "0.1.0",
    "timestamp": "2026-07-16T12:00:00+00:00",
    "backends_requested": ["mem0"],
    "evals_requested": ["contradiction"],
    "results": {
        "mem0": {
            "status": "configured",
            "evals": {
                "contradiction": {
                    "backend": "mem0",
                    "flagged_rate": 0.8,
                    "silent_overwrite_rate": 0.1,
                    "n_cases": 10,
                }
            },
        }
    },
    "cost": {"total_usd": 0.0123, "total_input_tokens": 1000, "total_output_tokens": 200},
}


# ---------------------------------------------------------------------------
# canonicalize / hashing
# ---------------------------------------------------------------------------


def test_canonicalize_is_key_order_independent() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_is_sensitive_to_value_changes() -> None:
    a = {"accuracy": 0.9}
    b = {"accuracy": 0.8}
    assert canonicalize(a) != canonicalize(b)


def test_payload_sha256_hex_matches_manual_recompute() -> None:
    digest = payload_sha256_hex(SAMPLE_REPORT)
    import hashlib

    expected = hashlib.sha256(canonicalize(SAMPLE_REPORT)).hexdigest()
    assert digest == expected


# ---------------------------------------------------------------------------
# sign + verify round trip
# ---------------------------------------------------------------------------


def test_sign_and_verify_round_trip_is_valid() -> None:
    private_key, public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)

    assert receipt["algorithm"] == "Ed25519"
    assert receipt["payload"] == SAMPLE_REPORT
    assert receipt["payload_sha256"] == payload_sha256_hex(SAMPLE_REPORT)

    result = verify_receipt(receipt, public_key)
    assert result.valid is True
    assert "signature verified" in result.reason
    assert result.embedded_key_matches_trusted_key is True


def test_receipt_is_json_serializable() -> None:
    private_key, _public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)
    # Must round-trip through json.dumps/json.loads cleanly -- this is
    # exactly what `memtrust run --sign` writes to disk.
    reloaded = json.loads(json.dumps(receipt))
    assert reloaded == receipt


def test_tampering_with_payload_after_signing_is_detected() -> None:
    private_key, public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)

    tampered = json.loads(json.dumps(receipt))
    tampered["payload"]["results"]["mem0"]["evals"]["contradiction"]["flagged_rate"] = 0.99

    result = verify_receipt(tampered, public_key)
    assert result.valid is False
    assert "payload" in result.reason.lower()


def test_tampering_with_signature_after_signing_is_detected() -> None:
    private_key, public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)

    tampered = dict(receipt)
    # Flip the signature to a different (still validly base64) value.
    tampered["signature"] = "AAAA" + receipt["signature"][4:]

    result = verify_receipt(tampered, public_key)
    assert result.valid is False


def test_verify_with_wrong_public_key_fails() -> None:
    private_key, _correct_public_key = generate_keypair()
    _other_private_key, wrong_public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)

    result = verify_receipt(receipt, wrong_public_key)
    assert result.valid is False
    assert "does not verify" in result.reason
    assert result.embedded_key_matches_trusted_key is False


def test_verify_receipt_missing_field_is_reported_not_raised() -> None:
    private_key, public_key = generate_keypair()
    receipt = sign_report(SAMPLE_REPORT, private_key)
    del receipt["signature"]

    result = verify_receipt(receipt, public_key)
    assert result.valid is False
    assert "missing required field" in result.reason


# ---------------------------------------------------------------------------
# keypair file I/O
# ---------------------------------------------------------------------------


def test_write_keypair_and_load_round_trip(tmp_path: Path) -> None:
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub"
    write_keypair(priv_path, pub_path)

    assert priv_path.exists()
    assert pub_path.exists()
    assert "BEGIN PRIVATE KEY" in priv_path.read_text()
    assert "BEGIN PUBLIC KEY" in pub_path.read_text()

    from memtrust.receipt import load_private_key

    private_key = load_private_key(priv_path)
    public_key = load_public_key(pub_path)

    receipt = sign_report(SAMPLE_REPORT, private_key)
    result = verify_receipt(receipt, public_key)
    assert result.valid is True


def test_write_keypair_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub"
    write_keypair(priv_path, pub_path)

    with pytest.raises(ReceiptError, match="refusing to overwrite"):
        write_keypair(priv_path, pub_path)


def test_write_keypair_force_overwrites(tmp_path: Path) -> None:
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub"
    write_keypair(priv_path, pub_path)
    original = priv_path.read_bytes()
    write_keypair(priv_path, pub_path, overwrite=True)
    assert priv_path.read_bytes() != original


def test_load_public_key_accepts_raw_base64(tmp_path: Path) -> None:
    from memtrust.receipt import _public_key_b64

    _private_key, public_key = generate_keypair()
    b64_path = tmp_path / "key.b64"
    b64_path.write_text(_public_key_b64(public_key))

    loaded = load_public_key(b64_path)
    assert _public_key_b64(loaded) == _public_key_b64(public_key)


# ---------------------------------------------------------------------------
# resolve_public_key (file vs env var precedence)
# ---------------------------------------------------------------------------


def test_resolve_public_key_prefers_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub"
    write_keypair(priv_path, pub_path)
    monkeypatch.setenv("MEMTRUST_RECEIPT_PUBLIC_KEY", "not-a-real-key")

    resolved = resolve_public_key(pub_path)
    expected = load_public_key(pub_path)
    from memtrust.receipt import _public_key_b64

    assert _public_key_b64(resolved) == _public_key_b64(expected)


def test_resolve_public_key_falls_back_to_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memtrust.receipt import _public_key_b64

    _private_key, public_key = generate_keypair()
    monkeypatch.setenv("MEMTRUST_RECEIPT_PUBLIC_KEY", _public_key_b64(public_key))

    resolved = resolve_public_key(None)
    assert _public_key_b64(resolved) == _public_key_b64(public_key)


def test_resolve_public_key_raises_when_neither_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMTRUST_RECEIPT_PUBLIC_KEY", raising=False)
    with pytest.raises(ReceiptError, match="not set"):
        resolve_public_key(None)


# ---------------------------------------------------------------------------
# verify_receipt_file / receipt_path_for
# ---------------------------------------------------------------------------


def test_verify_receipt_file_end_to_end(tmp_path: Path) -> None:
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub"
    write_keypair(priv_path, pub_path)

    from memtrust.receipt import load_private_key

    receipt = sign_report(SAMPLE_REPORT, load_private_key(priv_path))
    receipt_path = tmp_path / "report.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    result = verify_receipt_file(receipt_path, public_key_path=pub_path)
    assert result.valid is True


def test_verify_receipt_file_rejects_non_object_json(tmp_path: Path) -> None:
    _priv, pub = generate_keypair()
    pub_path = tmp_path / "key.pub"
    from memtrust.receipt import _public_key_b64

    pub_path.write_text(_public_key_b64(pub))

    receipt_path = tmp_path / "bad.json"
    receipt_path.write_text("[1, 2, 3]")

    result = verify_receipt_file(receipt_path, public_key_path=pub_path)
    assert result.valid is False
    assert "JSON object" in result.reason


def test_receipt_path_for_convention() -> None:
    assert receipt_path_for(Path("memtrust-report-2026-07-16.json")) == Path(
        "memtrust-report-2026-07-16.receipt.json"
    )
