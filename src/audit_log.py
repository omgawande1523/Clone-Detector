"""
audit_log.py — tamper-evident logging for every alert the system
raises, prototyping the "Blockchain & Cybersecurity" theme tag on
this PS honestly rather than just name-dropping the word.

What this actually is: a local hash chain (each record embeds the
hash of the previous record, like a block header referencing its
parent). Anyone can verify after the fact that no record was
inserted, deleted, or edited — if even one byte changes, every
subsequent hash breaks. That is the property that matters for
compliance/forensics: a bank's fraud team, or a regulator, needs to
trust that an alert log was not quietly edited after the fact.

WHAT IS COVERED BY THE HASH

    Every field of the record except the hash itself: prev_hash,
    timestamp, and payload.

    The timestamp used to be excluded, which quietly defeated the
    whole point — an insider could rewrite *when* a CRITICAL alert
    fired (say, moving it to after the transfer was approved) and
    verify_chain() would still report the log as valid. Backdating an
    alert is exactly the edit a tamper-evident log exists to catch.
    If you add fields to a record, add them to _hash_record() too.

What this is NOT (be upfront about this in your PPT): a real
distributed ledger. It runs on one machine with no consensus and no
external anchoring. The honest production upgrade is to periodically
anchor the running hash to a permissioned blockchain (e.g. Hyperledger
Fabric, or even a public chain's OP_RETURN) shared across the bank's
nodes — that gets you the same tamper-evidence PLUS "no single admin
can rewrite history unilaterally". Say exactly that to judges: this
prototype demonstrates the mechanism, production would anchor it.
"""

import hashlib
import json
import os
import threading
import time


GENESIS_HASH = "0" * 64

# Anchored to the project root, not the current working directory, so
# the audit trail does not silently fork depending on where uvicorn
# was launched from.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "audit_log.jsonl",
)


def _hash_record(prev_hash: str, timestamp: float, payload: dict) -> str:
    """Hashes every field of the record except the hash itself.

    sort_keys=True makes the encoding canonical, so the same logical
    record always hashes identically regardless of dict ordering.
    """
    blob = json.dumps(
        {"prev_hash": prev_hash, "timestamp": timestamp, "payload": payload},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str = None):
        self.path = path or DEFAULT_PATH
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            open(self.path, "w").close()

    def _iter_records(self):
        """Single definition of "what counts as a record in this file",
        used by BOTH _last_hash() and verify_chain() so they can never
        disagree about blank lines again. Yields (index, record) and
        raises ValueError on a malformed line.

        Blank/whitespace-only lines are skipped: a trailing newline is
        a normal property of a text log, not evidence of tampering.
        """
        with open(self.path) as f:
            index = 0
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"record {index} is not valid JSON: {e}")
                yield index, record
                index += 1

    def _last_hash(self) -> str:
        # Simple and robust beats clever-and-fragile for an audit log
        # that is realistically thousands, not millions, of lines.
        last_hash = GENESIS_HASH
        for _, record in self._iter_records():
            last_hash = record["hash"]
        return last_hash

    def append(self, payload: dict) -> dict:
        # Read-then-append must be atomic, or two concurrent alerts can
        # read the same prev_hash and fork the chain.
        with self._lock:
            prev_hash = self._last_hash()
            timestamp = time.time()
            record = {
                "timestamp": timestamp,
                "prev_hash": prev_hash,
                "payload": payload,
            }
            record["hash"] = _hash_record(prev_hash, timestamp, payload)
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def verify_chain(self) -> dict:
        """Walks the whole chain and confirms every hash lines up.

        Returns {"valid": bool, "record_count": int, "broken_at": int|None,
                 "reason": str|None}. broken_at is a record index, not a
        raw file line number.
        """
        prev_hash = GENESIS_HASH
        count = 0
        try:
            for index, record in self._iter_records():
                if not all(k in record for k in ("prev_hash", "timestamp", "payload", "hash")):
                    return {"valid": False, "record_count": count,
                            "broken_at": index, "reason": "record is missing required fields"}
                if record["prev_hash"] != prev_hash:
                    return {"valid": False, "record_count": count,
                            "broken_at": index,
                            "reason": "prev_hash does not match the previous record "
                                      "(a record was inserted, deleted or reordered)"}
                expected = _hash_record(
                    record["prev_hash"], record["timestamp"], record["payload"]
                )
                if record["hash"] != expected:
                    return {"valid": False, "record_count": count,
                            "broken_at": index,
                            "reason": "record hash does not match its contents "
                                      "(timestamp or payload was edited)"}
                prev_hash = record["hash"]
                count += 1
        except (ValueError, OSError) as e:
            return {"valid": False, "record_count": count,
                    "broken_at": count, "reason": str(e)}
        return {"valid": True, "record_count": count, "broken_at": None, "reason": None}
