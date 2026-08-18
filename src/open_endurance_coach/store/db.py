import json
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport

from .records import Decision, Draft, DraftStatus, Feedback

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_activities (
    activity_id TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    focus TEXT NOT NULL,
    user_feedback TEXT,
    context_json TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL REFERENCES drafts(id),
    created_at TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL REFERENCES drafts(id),
    decided_at TEXT NOT NULL,
    report_json TEXT NOT NULL
);
"""


class CoachStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def mark_activity_seen(self, activity_id: str) -> bool:
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO seen_activities (activity_id, seen_at) VALUES (?, ?)",
            (activity_id, self._clock().isoformat()),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def is_activity_seen(self, activity_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM seen_activities WHERE activity_id = ?", (activity_id,)
        ).fetchone()
        return row is not None

    def unseen_activity_ids(self, activity_ids: Iterable[str]) -> set[str]:
        rows = self._connection.execute("SELECT activity_id FROM seen_activities").fetchall()
        seen = {row["activity_id"] for row in rows}
        return {item for item in activity_ids if item not in seen}

    def save_draft(
        self,
        *,
        focus: str,
        report: DecisionReport,
        context: CoachContext,
        user_feedback: str | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO drafts (created_at, status, focus, user_feedback, context_json,"
            " report_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._clock().isoformat(),
                DraftStatus.PENDING.value,
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

    def _draft_from_row(self, row: sqlite3.Row) -> Draft:
        return Draft(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=DraftStatus(row["status"]),
            focus=row["focus"],
            user_feedback=row["user_feedback"],
            context=CoachContext.model_validate(json.loads(row["context_json"])),
            report=DecisionReport.model_validate(json.loads(row["report_json"])),
        )

    def get_draft(self, draft_id: int) -> Draft | None:
        row = self._connection.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return self._draft_from_row(row) if row else None

    def list_drafts(self, status: DraftStatus | None = None) -> list[Draft]:
        if status is None:
            rows = self._connection.execute("SELECT * FROM drafts ORDER BY id DESC").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM drafts WHERE status = ? ORDER BY id DESC", (status.value,)
            ).fetchall()
        return [self._draft_from_row(row) for row in rows]

    def update_draft_report(
        self,
        draft_id: int,
        *,
        report: DecisionReport,
        user_feedback: str | None,
    ) -> None:
        draft = self.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.status != DraftStatus.PENDING:
            raise ValueError(
                f"draft {draft_id} is {draft.status.value}; only pending drafts can be updated"
            )
        self._connection.execute(
            "UPDATE drafts SET report_json = ?, user_feedback = ? WHERE id = ?",
            (json.dumps(report.model_dump(mode="json")), user_feedback, draft_id),
        )
        self._connection.commit()

    def add_feedback(self, draft_id: int, content: str) -> int:
        if self.get_draft(draft_id) is None:
            raise ValueError(f"draft not found: {draft_id}")
        cursor = self._connection.execute(
            "INSERT INTO feedback (draft_id, created_at, content) VALUES (?, ?, ?)",
            (draft_id, self._clock().isoformat(), content),
        )
        self._connection.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        return lastrowid

    def list_feedback(self, draft_id: int) -> list[Feedback]:
        rows = self._connection.execute(
            "SELECT * FROM feedback WHERE draft_id = ? ORDER BY id", (draft_id,)
        ).fetchall()
        return [
            Feedback(
                id=row["id"],
                draft_id=row["draft_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                content=row["content"],
            )
            for row in rows
        ]

    def approve_draft(self, draft_id: int) -> Decision:
        draft = self.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.status != DraftStatus.PENDING:
            raise ValueError(
                f"draft {draft_id} is {draft.status.value}; only pending drafts can be approved"
            )
        decided_at = self._clock()
        self._connection.execute(
            "UPDATE drafts SET status = ? WHERE id = ?", (DraftStatus.APPROVED.value, draft_id)
        )
        cursor = self._connection.execute(
            "INSERT INTO decisions (draft_id, decided_at, report_json) VALUES (?, ?, ?)",
            (draft_id, decided_at.isoformat(), json.dumps(draft.report.model_dump(mode="json"))),
        )
        self._connection.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        return Decision(
            id=lastrowid,
            draft_id=draft_id,
            decided_at=decided_at,
            report=draft.report,
        )

    def reject_draft(self, draft_id: int) -> None:
        draft = self.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.status != DraftStatus.PENDING:
            raise ValueError(
                f"draft {draft_id} is {draft.status.value}; only pending drafts can be rejected"
            )
        self._connection.execute(
            "UPDATE drafts SET status = ? WHERE id = ?", (DraftStatus.REJECTED.value, draft_id)
        )
        self._connection.commit()

    def list_decisions(self) -> list[Decision]:
        rows = self._connection.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        return [
            Decision(
                id=row["id"],
                draft_id=row["draft_id"],
                decided_at=datetime.fromisoformat(row["decided_at"]),
                report=DecisionReport.model_validate(json.loads(row["report_json"])),
            )
            for row in rows
        ]
