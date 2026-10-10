import sqlite3
from pathlib import Path

import pytest

from open_endurance_coach.store import db as store_db
from open_endurance_coach.store.db import SCHEMA_VERSION, CoachStore


def _user_version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def _set_user_version(path: Path, version: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    finally:
        connection.close()


def _table_names(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def _column_names(path: Path, table: str) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    finally:
        connection.close()


def test_fresh_database_records_the_current_version(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    store = CoachStore(path)
    assert store.schema_version == SCHEMA_VERSION
    store.close()
    assert _user_version(path) == SCHEMA_VERSION


def test_fresh_database_creates_the_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    assert {"proposals", "messages", "seen_activities"} <= _table_names(path)


def test_unversioned_database_is_stamped_to_the_current_version(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    _set_user_version(path, 0)

    reopened = CoachStore(path)
    assert reopened.schema_version == SCHEMA_VERSION
    reopened.close()


def test_migration_is_idempotent_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    first = CoachStore(path)
    assert first.schema_version == SCHEMA_VERSION
    first.close()
    second = CoachStore(path)
    assert second.schema_version == SCHEMA_VERSION
    second.close()


def test_existing_rows_survive_an_unversioned_migration(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO seen_activities (activity_id, seen_at) VALUES ('a', '2024-01-01')"
        )
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
    finally:
        connection.close()

    reopened = CoachStore(path)
    assert reopened.is_activity_seen("a") is True
    reopened.close()


def test_a_new_migration_reaches_an_existing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    assert _user_version(path) == SCHEMA_VERSION

    monkeypatch.setattr(store_db, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    monkeypatch.setitem(
        store_db.MIGRATIONS, SCHEMA_VERSION + 1, "ALTER TABLE proposals ADD COLUMN note TEXT;"
    )

    reopened = CoachStore(path)
    assert reopened.schema_version == SCHEMA_VERSION + 1
    reopened.close()
    assert "note" in _column_names(path, "proposals")


def test_ordered_migrations_run_step_by_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()

    monkeypatch.setattr(store_db, "SCHEMA_VERSION", SCHEMA_VERSION + 2)
    monkeypatch.setitem(
        store_db.MIGRATIONS, SCHEMA_VERSION + 1, "CREATE TABLE step_two (id INTEGER);"
    )
    monkeypatch.setitem(
        store_db.MIGRATIONS, SCHEMA_VERSION + 2, "CREATE TABLE step_three (id INTEGER);"
    )

    reopened = CoachStore(path)
    assert reopened.schema_version == SCHEMA_VERSION + 2
    reopened.close()
    assert {"step_two", "step_three"} <= _table_names(path)


def test_a_newer_database_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    _set_user_version(path, SCHEMA_VERSION + 1)

    with pytest.raises(RuntimeError, match="version"):
        CoachStore(path)


def test_legacy_rejected_rows_are_cleaned_on_migration(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO proposals (created_at, status, focus, context_json, report_json)"
            " VALUES ('2024-01-01T00:00:00+00:00', 'rejected', 'legacy', '{}', '{}')"
        )
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
    finally:
        connection.close()

    reopened = CoachStore(path)
    assert reopened.list_proposals() == []
    assert reopened.schema_version == SCHEMA_VERSION
    reopened.close()


def test_failed_migration_rolls_back_and_keeps_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path).close()

    monkeypatch.setattr(store_db, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    monkeypatch.setitem(
        store_db.MIGRATIONS,
        SCHEMA_VERSION + 1,
        "CREATE TABLE partial (id INTEGER); CREATE TABLE partial (id INTEGER);",
    )

    with pytest.raises(sqlite3.Error):
        CoachStore(path)
    assert _user_version(path) == SCHEMA_VERSION
    assert "partial" not in _table_names(path)
