"""Regression tests for the tamper-evident audit log.

The headline test here is test_timestamp_tamper_is_detected: the
timestamp used to be excluded from the hashed blob, so an insider
could backdate when a CRITICAL alert fired — say, to after a transfer
was approved — and verify_chain() would still report the log as
valid. That is precisely the edit a tamper-evident log exists to
catch.
"""

import json

import pytest

from src.audit_log import AuditLog, GENESIS_HASH


@pytest.fixture
def log(tmp_path):
    audit = AuditLog(path=str(tmp_path / "audit.jsonl"))
    audit.append({"speaker_id": "ceo", "level": "HIGH"})
    audit.append({"speaker_id": "ceo", "level": "CRITICAL"})
    return audit


def _read(log):
    return [json.loads(line) for line in open(log.path) if line.strip()]


def _write(log, records):
    with open(log.path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_clean_chain_verifies(log):
    result = log.verify_chain()
    assert result["valid"] is True
    assert result["record_count"] == 2
    assert result["broken_at"] is None


def test_first_record_chains_from_genesis(log):
    assert _read(log)[0]["prev_hash"] == GENESIS_HASH


def test_timestamp_tamper_is_detected(log):
    """THE regression test for the original bug."""
    records = _read(log)
    records[1]["timestamp"] -= 86400        # backdate by a day
    _write(log, records)

    result = log.verify_chain()
    assert result["valid"] is False, "backdating an alert must break the chain"
    assert result["broken_at"] == 1


def test_payload_tamper_is_detected(log):
    records = _read(log)
    records[1]["payload"]["level"] = "LOW"   # downgrade the alert
    _write(log, records)
    assert log.verify_chain()["valid"] is False


def test_record_deletion_is_detected(log):
    records = _read(log)
    del records[0]
    _write(log, records)
    assert log.verify_chain()["valid"] is False


def test_record_reordering_is_detected(log):
    records = _read(log)
    records.reverse()
    _write(log, records)
    assert log.verify_chain()["valid"] is False


@pytest.mark.parametrize("junk", ["\n", "   \n", "\n\n  \n"])
def test_blank_lines_do_not_break_verification(log, junk):
    """verify_chain() used to raise JSONDecodeError on a trailing blank
    line while _last_hash() silently skipped it — an HTTP 500 on
    /audit/verify waiting to happen."""
    with open(log.path, "a") as f:
        f.write(junk)
    result = log.verify_chain()
    assert result["valid"] is True
    assert result["record_count"] == 2


def test_appending_after_blank_lines_still_chains(log):
    """_last_hash() and verify_chain() must agree about what a record
    is, or a new record written after a blank line would fork."""
    with open(log.path, "a") as f:
        f.write("\n\n")
    log.append({"speaker_id": "ceo", "level": "HIGH"})
    result = log.verify_chain()
    assert result["valid"] is True
    assert result["record_count"] == 3


def test_corrupt_line_is_reported_not_raised(log):
    with open(log.path, "a") as f:
        f.write("{not json\n")
    result = log.verify_chain()          # must not raise
    assert result["valid"] is False
    assert "JSON" in result["reason"]


def test_empty_log_is_valid(tmp_path):
    assert AuditLog(path=str(tmp_path / "empty.jsonl")).verify_chain()["valid"] is True


def test_missing_file_reads_as_empty_chain(tmp_path):
    """The log file is created in __init__, but can be deleted or
    rotated afterwards. That must not 500 every subsequent alert."""
    path = str(tmp_path / "audit.jsonl")
    audit = AuditLog(path=path)
    audit.append({"level": "HIGH"})

    import os
    os.remove(path)                      # rotated away underneath us

    assert audit.verify_chain()["valid"] is True       # empty, not a crash
    audit.append({"level": "CRITICAL"})                # must still record
    result = audit.verify_chain()
    assert result["valid"] is True
    assert result["record_count"] == 1
