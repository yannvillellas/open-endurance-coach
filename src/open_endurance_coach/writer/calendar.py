from datetime import date, timedelta
from typing import Any

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.protocols import IntervalsCalendarClient
from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DeleteWorkout,
    UpdateWorkout,
    WorkoutMutation,
)
from open_endurance_coach.store.records import Decision

from .records import MutationOutcome

WORKOUT_CATEGORY = "WORKOUT"


class WriterError(RuntimeError):
    pass


class CalendarWriter:
    def __init__(self, client: IntervalsCalendarClient) -> None:
        self._client = client

    async def apply_decision(self, decision: Decision) -> list[MutationOutcome]:
        outcomes = []
        for mutation in decision.report.mutations:
            outcomes.append(await self._apply_mutation(mutation))
        return outcomes

    async def _apply_mutation(self, mutation: WorkoutMutation) -> MutationOutcome:
        if isinstance(mutation, CreateWorkout):
            return await self._apply_create(mutation)
        if isinstance(mutation, UpdateWorkout):
            return await self._apply_update(mutation)
        return await self._apply_delete(mutation)

    @staticmethod
    def _date_string(day: date) -> str:
        return f"{day.isoformat()}T00:00:00"

    def _create_payload(self, mutation: CreateWorkout) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": WORKOUT_CATEGORY,
            "name": mutation.name,
            "start_date_local": self._date_string(mutation.start_date_local),
        }
        for field in ("description", "type", "moving_time", "icu_training_load"):
            value = getattr(mutation, field)
            if value is not None:
                payload[field] = value
        return payload

    async def _find_workout_by_name_and_date(self, name: str, day: date) -> dict[str, Any] | None:
        rows = await self._client.list_events(
            day.isoformat(), (day + timedelta(days=1)).isoformat(), category=WORKOUT_CATEGORY
        )
        for row in rows:
            if row.get("name") == name:
                return row
        return None

    async def _apply_create(self, mutation: CreateWorkout) -> MutationOutcome:
        payload = self._create_payload(mutation)
        existing = await self._find_workout_by_name_and_date(
            mutation.name, mutation.start_date_local
        )
        if existing is not None:
            await self._client.update_event(str(existing["id"]), payload)
            return MutationOutcome(
                action="create",
                target="updated",
                event_id=existing["id"],
                name=mutation.name,
            )
        created = await self._client.create_event(payload)
        return MutationOutcome(
            action="create", target="created", event_id=created.get("id"), name=mutation.name
        )

    async def _fetch_workout(self, event_id: int | str) -> dict[str, Any] | None:
        try:
            event = await self._client.get_event(str(event_id))
        except IntervalsApiError as exc:
            if exc.status_code == 404:
                return None
            raise
        if event.get("category") != WORKOUT_CATEGORY:
            raise WriterError(
                f"refusing to mutate non-WORKOUT event {event_id}"
                f" (category: {event.get('category')})"
            )
        return event

    async def _apply_update(self, mutation: UpdateWorkout) -> MutationOutcome:
        event = await self._fetch_workout(mutation.event_id)
        if event is None:
            raise WriterError(f"update target event not found: {mutation.event_id}")
        payload: dict[str, Any] = {}
        if mutation.name is not None:
            payload["name"] = mutation.name
        if mutation.start_date_local is not None:
            payload["start_date_local"] = self._date_string(mutation.start_date_local)
        for field in ("description", "type", "moving_time", "icu_training_load"):
            value = getattr(mutation, field)
            if value is not None:
                payload[field] = value
        await self._client.update_event(str(mutation.event_id), payload)
        return MutationOutcome(action="update", target="updated", event_id=mutation.event_id)

    async def _apply_delete(self, mutation: DeleteWorkout) -> MutationOutcome:
        event = await self._fetch_workout(mutation.event_id)
        if event is None:
            return MutationOutcome(action="delete", target="skipped", event_id=mutation.event_id)
        await self._client.delete_event(str(mutation.event_id))
        return MutationOutcome(action="delete", target="deleted", event_id=mutation.event_id)
