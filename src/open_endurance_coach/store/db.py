import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport

from .records import Feedback, FeedbackWithReport, Proposal, ProposalStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_activities (
    activity_id TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    focus TEXT NOT NULL,
    user_feedback TEXT,
    context_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    approved_json TEXT,
    decided_at TEXT,
    applied_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id INTEGER REFERENCES proposals(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    content TEXT NOT NULL,
    report_json TEXT
);
"""


logger = logging.getLogger(__name__)
_SQL_VARIABLE_BATCH = 900
_VACUUM_MIN_ROWS = 100


class CoachStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._restrict_permissions()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.executescript(
            "DELETE FROM feedback WHERE proposal_id IN"
            " (SELECT id FROM proposals WHERE status = 'rejected');"
            "DELETE FROM proposals WHERE status = 'rejected';"
        )
        self._connection.commit()

    def discard_proposal(self, proposal_id: int) -> None:
        self._connection.execute("DELETE FROM proposals WHERE id = ?", (proposal_id,))
        self._connection.commit()

    def _restrict_permissions(self) -> None:
        if str(self._path) == ":memory:":
            return
        try:
            self._path.chmod(0o600)
        except OSError:
            logger.warning("could not restrict permissions on %s", self._path)

    def prune_before(self, cutoff: datetime) -> dict[str, int]:
        stamp = cutoff.isoformat()
        statements = {
            "feedback": (
                "DELETE FROM feedback WHERE proposal_id IN"
                " (SELECT id FROM proposals WHERE created_at < ?)"
            ),
            "proposals": "DELETE FROM proposals WHERE created_at < ?",
            "seen_activities": "DELETE FROM seen_activities WHERE seen_at < ?",
        }
        counts: dict[str, int] = {}
        try:
            for name, statement in statements.items():
                cursor = self._connection.execute(statement, (stamp,))
                counts[name] = cursor.rowcount
            self._connection.commit()
        except sqlite3.Error:
            self._connection.rollback()
            raise
        if sum(counts.values()) >= _VACUUM_MIN_ROWS:
            self._connection.execute("VACUUM")
        return counts

    def close(self) -> None:
        self._connection.close()

    def mark_activities_seen(self, activity_ids: Iterable[str]) -> int:
        rows = [(activity_id, self._clock().isoformat()) for activity_id in activity_ids]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "INSERT OR IGNORE INTO seen_activities (activity_id, seen_at) VALUES (?, ?)",
            rows,
        )
        self._connection.commit()
        return cursor.rowcount

    def is_activity_seen(self, activity_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM seen_activities WHERE activity_id = ?", (activity_id,)
        ).fetchone()
        return row is not None

    def unseen_activity_ids(self, activity_ids: Iterable[str]) -> set[str]:
        ids = list(activity_ids)
        if not ids:
            return set()
        seen: set[str] = set()
        for start in range(0, len(ids), _SQL_VARIABLE_BATCH):
            batch = ids[start : start + _SQL_VARIABLE_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows = self._connection.execute(
                f"SELECT activity_id FROM seen_activities WHERE activity_id IN ({placeholders})",
                batch,
            ).fetchall()
            seen.update(row["activity_id"] for row in rows)
        return {item for item in ids if item not in seen}

    def save_proposal(
        self,
        *,
        focus: str,
        report: DecisionReport,
        context: CoachContext,
        user_feedback: str | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO proposals (created_at, status, focus, user_feedback, context_json,"
            " report_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._clock().isoformat(),
                ProposalStatus.PENDING.value,
                focus,
                user_feedback,
                json.dumps(context.model_dump(mode="json")),
                json.dumps(report.model_dump(mode="json")),
            ),
        )
        self._connection.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        return lastrowid

    def _proposal_from_row(self, row: sqlite3.Row) -> Proposal:
        approved_json = row["approved_json"]
        decided_at = row["decided_at"]
        applied_at = row["applied_at"]
        return Proposal(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=ProposalStatus(row["status"]),
            focus=row["focus"],
            user_feedback=row["user_feedback"],
            context=CoachContext.model_validate(json.loads(row["context_json"])),
            report=DecisionReport.model_validate(json.loads(row["report_json"])),
            approved_report=(
                DecisionReport.model_validate(json.loads(approved_json))
                if approved_json is not None
                else None
            ),
            decided_at=datetime.fromisoformat(decided_at) if decided_at else None,
            applied_at=datetime.fromisoformat(applied_at) if applied_at else None,
        )

    def get_proposal(self, proposal_id: int) -> Proposal | None:
        row = self._connection.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        return self._proposal_from_row(row) if row else None

    def list_proposals(self, status: ProposalStatus | None = None) -> list[Proposal]:
        if status is None:
            rows = self._connection.execute("SELECT * FROM proposals ORDER BY id DESC").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM proposals WHERE status = ? ORDER BY id DESC", (status.value,)
            ).fetchall()
        return [self._proposal_from_row(row) for row in rows]

    def list_approved_proposals(self) -> list[Proposal]:
        return self.list_proposals(ProposalStatus.APPROVED)

    def update_proposal_report(
        self,
        proposal_id: int,
        *,
        report: DecisionReport,
        user_feedback: str | None,
        context: CoachContext | None = None,
    ) -> None:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id} is {proposal.status.value};"
                " only pending proposals can be updated"
            )
        if context is not None:
            self._connection.execute(
                "UPDATE proposals SET report_json = ?, user_feedback = ?, context_json = ?"
                " WHERE id = ?",
                (
                    json.dumps(report.model_dump(mode="json")),
                    user_feedback,
                    json.dumps(context.model_dump(mode="json")),
                    proposal_id,
                ),
            )
        else:
            self._connection.execute(
                "UPDATE proposals SET report_json = ?, user_feedback = ? WHERE id = ?",
                (json.dumps(report.model_dump(mode="json")), user_feedback, proposal_id),
            )
        self._connection.commit()

    def add_feedback(
        self, proposal_id: int, content: str, *, report: DecisionReport | None = None
    ) -> int:
        if self.get_proposal(proposal_id) is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        cursor = self._connection.execute(
            "INSERT INTO feedback (proposal_id, created_at, content, report_json)"
            " VALUES (?, ?, ?, ?)",
            (
                proposal_id,
                self._clock().isoformat(),
                content,
                json.dumps(report.model_dump(mode="json")) if report is not None else None,
            ),
        )
        self._connection.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        return lastrowid

    def set_feedback_report(self, feedback_id: int, report: DecisionReport) -> None:
        self._connection.execute(
            "UPDATE feedback SET report_json = ? WHERE id = ?",
            (json.dumps(report.model_dump(mode="json")), feedback_id),
        )
        self._connection.commit()

    def list_feedback(self, proposal_id: int) -> list[Feedback]:
        rows = self._connection.execute(
            "SELECT * FROM feedback WHERE proposal_id = ? ORDER BY id", (proposal_id,)
        ).fetchall()
        return [
            Feedback(
                id=row["id"],
                proposal_id=row["proposal_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                content=row["content"],
            )
            for row in rows
        ]

    def recent_feedback(
        self, limit: int, *, max_age_days: int | None = None
    ) -> list[FeedbackWithReport]:
        if limit <= 0:
            return []
        query = (
            "SELECT f.id AS id, f.proposal_id AS proposal_id, f.created_at AS created_at,"
            " f.content AS content,"
            " COALESCE(f.report_json, p.report_json) AS report_json"
            " FROM feedback f JOIN proposals p ON p.id = f.proposal_id"
        )
        params: tuple[object, ...] = ()
        if max_age_days is not None:
            cutoff = (self._clock() - timedelta(days=max_age_days)).isoformat()
            query += " WHERE f.created_at >= ?"
            params += (cutoff,)
        query += " ORDER BY f.id DESC LIMIT ?"
        params += (limit,)
        rows = self._connection.execute(query, params).fetchall()
        return [
            FeedbackWithReport(
                feedback=Feedback(
                    id=row["id"],
                    proposal_id=row["proposal_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    content=row["content"],
                ),
                report=DecisionReport.model_validate(json.loads(row["report_json"])),
            )
            for row in rows
        ]

    def approve_proposal(
        self, proposal_id: int, *, report: DecisionReport | None = None
    ) -> Proposal:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id} is {proposal.status.value};"
                " only pending proposals can be approved"
            )
        approved = report if report is not None else proposal.report
        decided_at = self._clock()
        with self._connection:
            self._connection.execute(
                "UPDATE proposals SET status = ?, approved_json = ?, decided_at = ? WHERE id = ?",
                (
                    ProposalStatus.APPROVED.value,
                    json.dumps(approved.model_dump(mode="json")),
                    decided_at.isoformat(),
                    proposal_id,
                ),
            )
        updated = self.get_proposal(proposal_id)
        assert updated is not None
        return updated

    def reject_proposal(self, proposal_id: int) -> None:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id} is {proposal.status.value};"
                " only pending proposals can be rejected"
            )
        self._connection.execute(
            "UPDATE proposals SET status = ? WHERE id = ?",
            (ProposalStatus.REJECTED.value, proposal_id),
        )
        self._connection.commit()

    def list_unapplied_proposals(self) -> list[Proposal]:
        rows = self._connection.execute(
            "SELECT * FROM proposals WHERE status = ? AND applied_at IS NULL ORDER BY id",
            (ProposalStatus.APPROVED.value,),
        ).fetchall()
        return [self._proposal_from_row(row) for row in rows]

    def mark_proposal_applied(self, proposal_id: int) -> None:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if proposal.applied_at is not None:
            raise ValueError(f"proposal {proposal_id} is already applied")
        self._connection.execute(
            "UPDATE proposals SET applied_at = ? WHERE id = ?",
            (self._clock().isoformat(), proposal_id),
        )
        self._connection.commit()
