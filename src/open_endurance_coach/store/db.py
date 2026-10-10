import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport

from .records import Message, MessageRole, Proposal, ProposalStatus

_BASE_SCHEMA = """
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
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    report_json TEXT
);
"""

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, str] = {
    1: _BASE_SCHEMA
    + """
DELETE FROM proposals WHERE status = 'rejected';
""",
    2: """
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_created_at ON proposals(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_seen_activities_seen_at ON seen_activities(seen_at);
""",
}


logger = logging.getLogger(__name__)
_SQL_VARIABLE_BATCH = 900
_VACUUM_MIN_ROWS = 100


def _vacuum_best_effort(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("VACUUM")
    except (sqlite3.Error, KeyboardInterrupt):
        logger.warning("VACUUM skipped")


class CoachStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._restrict_permissions()
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._migrate()
        except BaseException:
            self._connection.close()
            raise

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def _migrate(self) -> None:
        version = self.schema_version
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        for target in range(version + 1, SCHEMA_VERSION + 1):
            script = f"BEGIN;\n{MIGRATIONS[target]}\nPRAGMA user_version = {target};\nCOMMIT;"
            try:
                self._connection.executescript(script)
            except sqlite3.Error:
                self._connection.rollback()
                raise

    def delete_proposal(self, proposal_id: int) -> None:
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
            "messages": "DELETE FROM messages WHERE created_at < ?",
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

    def delete_all_local(self) -> dict[str, int]:
        statements = {
            "messages": "DELETE FROM messages",
            "proposals": "DELETE FROM proposals",
            "seen_activities": "DELETE FROM seen_activities",
        }
        counts: dict[str, int] = {}
        try:
            for name, statement in statements.items():
                counts[name] = self._connection.execute(statement).rowcount
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        if sum(counts.values()) >= _VACUUM_MIN_ROWS:
            _vacuum_best_effort(self._connection)
        return counts

    def prune_messages(self, cutoff: datetime | None) -> int:
        try:
            if cutoff is None:
                cursor = self._connection.execute("DELETE FROM messages")
            else:
                cursor = self._connection.execute(
                    "DELETE FROM messages WHERE created_at < ?", (cutoff.isoformat(),)
                )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        removed = cursor.rowcount
        if removed >= _VACUUM_MIN_ROWS:
            _vacuum_best_effort(self._connection)
        return removed

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

    def list_unapplied_proposals(self) -> list[Proposal]:
        rows = self._connection.execute(
            "SELECT * FROM proposals WHERE status = ? ORDER BY id",
            (ProposalStatus.UNAPPLIED.value,),
        ).fetchall()
        return [self._proposal_from_row(row) for row in rows]

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

    def add_message(
        self,
        role: MessageRole,
        content: str,
        *,
        report: DecisionReport | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO messages (created_at, role, content, report_json) VALUES (?, ?, ?, ?)",
            (
                self._clock().isoformat(),
                role.value,
                content,
                json.dumps(report.model_dump(mode="json")) if report is not None else None,
            ),
        )
        self._connection.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        return lastrowid

    def set_message_report(self, message_id: int, report: DecisionReport) -> None:
        self._connection.execute(
            "UPDATE messages SET report_json = ? WHERE id = ?",
            (json.dumps(report.model_dump(mode="json")), message_id),
        )
        self._connection.commit()

    def _message_from_row(self, row: sqlite3.Row) -> Message:
        report_json = row["report_json"]
        return Message(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            role=MessageRole(row["role"]),
            content=row["content"],
            report=(
                DecisionReport.model_validate(json.loads(report_json))
                if report_json is not None
                else None
            ),
        )

    def list_messages(self) -> list[Message]:
        rows = self._connection.execute("SELECT * FROM messages ORDER BY id").fetchall()
        return [self._message_from_row(row) for row in rows]

    def recent_messages(self, limit: int, *, max_age_days: int | None = None) -> list[Message]:
        if limit <= 0:
            return []
        query = "SELECT * FROM messages"
        params: tuple[object, ...] = ()
        if max_age_days is not None:
            cutoff = (self._clock() - timedelta(days=max_age_days)).isoformat()
            query += " WHERE created_at >= ?"
            params += (cutoff,)
        query += " ORDER BY id DESC LIMIT ?"
        params += (limit,)
        rows = self._connection.execute(query, params).fetchall()
        return [self._message_from_row(row) for row in rows]

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
            cursor = self._connection.execute(
                "UPDATE proposals SET status = ?, approved_json = ?, decided_at = ?"
                " WHERE id = ? AND status = ?",
                (
                    ProposalStatus.UNAPPLIED.value,
                    json.dumps(approved.model_dump(mode="json")),
                    decided_at.isoformat(),
                    proposal_id,
                    ProposalStatus.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"proposal {proposal_id} is no longer pending")
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
        self.delete_proposal(proposal_id)
