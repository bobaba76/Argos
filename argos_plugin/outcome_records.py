"""Procedure outcome records — make the convention live (#16).

The procedure-outcome-records convention (docs/procedure-outcome-records.md)
specifies: procedures carry version, success_count/fail_count, per-step
(n✓/m✗) counters, a ⚑ tripwire marker for failing steps, and an ## Evolution
log. The executor updates the record as part of the loop; a deterministic
tripwatch (circuit breaker) reads the record and alerts on repeated failures.

This module implements:
1. **Outcome record format** — a structured payload for procedure outcome
   records, stored as memory content with payload.kind="outcome".
2. **Counter updates** — update_success / update_fail functions that
   increment the counters and maintain the tripwire marker.
3. **Tripwatch** — a deterministic checker that reads outcome records and
   alerts on repeated failures. Zero LLM, zero storage schema.

The record is content with payload metadata — no schema migration, no
retrieval changes. The tripwatch is a pure function that reads records
and returns alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Tripwire marker — Unicode flag character.
TRIPWIRE_MARKER = "\u2691"  # ⚑

# Tripwatch thresholds (deterministic, zero-LLM).
# A step trips when: m >= 2 fails AND m >= n holds.
# A record trips when: fail_count >= 2 AND fail_count >= success_count.
_TRIPWATCH_STEP_MIN_FAILS = 2
_TRIPWATCH_STEP_FAIL_DOMINANT = True  # m >= n
_TRIPWATCH_RECORD_MIN_FAILS = 2
_TRIPWATCH_RECORD_FAIL_DOMINANT = True  # fail_count >= success_count


@dataclass
class StepCounter:
    """Per-step counter for a procedure outcome record."""
    name: str
    success_count: int = 0
    fail_count: int = 0

    def total(self) -> int:
        return self.success_count + self.fail_count

    def is_tripped(self) -> bool:
        """True if this step has tripped the tripwire."""
        return (
            self.fail_count >= _TRIPWATCH_STEP_MIN_FAILS
            and self.fail_count >= self.success_count
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepCounter":
        return cls(
            name=d.get("name", ""),
            success_count=int(d.get("success_count", 0)),
            fail_count=int(d.get("fail_count", 0)),
        )


@dataclass
class OutcomeRecord:
    """A procedure outcome record (#16).

    Stored as memory content with payload.kind="outcome". The content is
    a human-readable summary; the payload carries the structured counters.
    """
    procedure_name: str
    version: str = "1.0"
    success_count: int = 0
    fail_count: int = 0
    steps: List[StepCounter] = field(default_factory=list)
    evolution_log: List[str] = field(default_factory=list)

    def total(self) -> int:
        return self.success_count + self.fail_count

    def is_tripped(self) -> bool:
        """True if this record has tripped the tripwire."""
        if self.fail_count >= _TRIPWATCH_RECORD_MIN_FAILS:
            if _TRIPWATCH_RECORD_FAIL_DOMINANT:
                return self.fail_count >= self.success_count
            return True
        return False

    def tripped_steps(self) -> List[StepCounter]:
        """Steps that have tripped the tripwire."""
        return [s for s in self.steps if s.is_tripped()]

    def update_success(self, step_name: str = "") -> None:
        """Record a successful execution."""
        self.success_count += 1
        if step_name:
            step = self._find_or_create_step(step_name)
            step.success_count += 1

    def update_fail(self, step_name: str = "", reason: str = "") -> None:
        """Record a failed execution."""
        self.fail_count += 1
        if step_name:
            step = self._find_or_create_step(step_name)
            step.fail_count += 1
        if reason:
            self.evolution_log.append(
                f"FAIL ({self.fail_count}): {reason}"
            )

    def _find_or_create_step(self, name: str) -> StepCounter:
        for s in self.steps:
            if s.name == name:
                return s
        step = StepCounter(name=name)
        self.steps.append(step)
        return step

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to a payload dict for storage."""
        return {
            "kind": "outcome",
            "procedure_name": self.procedure_name,
            "version": self.version,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "steps": [s.to_dict() for s in self.steps],
            "evolution_log": self.evolution_log,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "OutcomeRecord":
        """Deserialize from a payload dict."""
        return cls(
            procedure_name=payload.get("procedure_name", ""),
            version=payload.get("version", "1.0"),
            success_count=int(payload.get("success_count", 0)),
            fail_count=int(payload.get("fail_count", 0)),
            steps=[
                StepCounter.from_dict(s)
                for s in payload.get("steps", [])
            ],
            evolution_log=list(payload.get("evolution_log", [])),
        )

    def to_content(self) -> str:
        """Render as human-readable content for the memory record."""
        lines = [
            f"Procedure: {self.procedure_name} (v{self.version})",
            f"Overall: {self.success_count}\u2713/{self.fail_count}\u2717",
        ]
        for step in self.steps:
            marker = f" {TRIPWIRE_MARKER}" if step.is_tripped() else ""
            lines.append(
                f"  {step.name}: {step.success_count}\u2713/{step.fail_count}\u2717{marker}"
            )
        if self.evolution_log:
            lines.append("## Evolution")
            for entry in self.evolution_log[-10:]:  # last 10 entries
                lines.append(f"- {entry}")
        return "\n".join(lines)


# --- Tripwatch (layer 3) ---------------------------------------------------

@dataclass
class TripwatchAlert:
    """An alert from the tripwatch checker."""
    record_id: str
    procedure_name: str
    reason: str
    severity: str  # "record" or "step"
    step_name: str = ""


def tripwatch_check(
    records: List[tuple[str, Dict[str, Any]]],
) -> List[TripwatchAlert]:
    """Run the deterministic tripwatch on a set of outcome records.

    Args:
        records: A list of (record_id, payload) tuples for outcome records.

    Returns:
        A list of TripwatchAlert for tripped records and steps.
        Empty list if all records are healthy.
    """
    alerts: List[TripwatchAlert] = []
    for record_id, payload in records:
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") != "outcome":
            continue
        record = OutcomeRecord.from_payload(payload)
        # Check record-level trip.
        if record.is_tripped():
            alerts.append(TripwatchAlert(
                record_id=record_id,
                procedure_name=record.procedure_name,
                reason=(
                    f"Record tripped: {record.fail_count} fails >= "
                    f"{record.success_count} successes"
                ),
                severity="record",
            ))
        # Check step-level trips.
        for step in record.tripped_steps():
            alerts.append(TripwatchAlert(
                record_id=record_id,
                procedure_name=record.procedure_name,
                reason=(
                    f"Step '{step.name}' tripped: {step.fail_count} fails >= "
                    f"{step.success_count} successes"
                ),
                severity="step",
                step_name=step.name,
            ))
    return alerts


def tripwatch_check_store(store) -> List[TripwatchAlert]:
    """Run the tripwatch against all outcome records in a store.

    Reads active memory records with payload.kind="outcome" and checks
    them for tripwire conditions. Zero LLM, zero storage schema.
    """
    try:
        with store._lock:
            assert store.connection is not None
            rows = store.connection.execute(
                """SELECT memory_id, payload FROM memory_records
                   WHERE valid_to IS NULL
                     AND (user_scope IS NULL OR user_scope = ?)
                     AND COALESCE(status, 'active') = 'active'""",
                [store.user_id],
            ).fetchall()
        records = []
        for memory_id, payload_raw in rows:
            try:
                import json
                payload = json.loads(payload_raw) if payload_raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            records.append((memory_id, payload))
        return tripwatch_check(records)
    except Exception:
        return []
