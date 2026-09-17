import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmClient, LlmMessage
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor
from open_endurance_coach.prompts.prompts import build_messages, system_prompt
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    Mutation,
    UpdateRace,
    UpdateWorkout,
)
from open_endurance_coach.schemas.intervals import Event
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import (
    Decision,
    Draft,
    DraftStatus,
    FeedbackWithReport,
)
from open_endurance_coach.tokens import INPUT_TOKEN_CEILING, estimate_text_tokens
from open_endurance_coach.writer.calendar import CalendarWriter
from open_endurance_coach.writer.records import AppliedDecision, ApplyReport

logger = logging.getLogger(__name__)


def _today(context: CoachContext, settings: Settings) -> date:
    if context.today is not None:
        return context.today
    return datetime.now(ZoneInfo(settings.app_timezone)).date()


def _context_event_dates(context: CoachContext) -> dict[str, date]:
    dates: dict[str, date] = {}
    for race in context.goal_races:
        if race.event_id is not None:
            dates[str(race.event_id)] = race.date
    for event in context.upcoming_events:
        if event.id is not None:
            dates[str(event.id)] = event.start_date_local.date()
    return dates


MAX_PLANNING_DAYS = 400


class StaleDecisionError(ValueError):
    """The decision's dated mutations are in the past."""


class PlaceholderMutationError(ValueError):
    """The decision still carries example placeholder values."""


def _is_stale(mutation: Any, *, today: date) -> bool:
    return (
        isinstance(mutation, (CreateWorkout, CreateRace, UpdateWorkout, UpdateRace))
        and mutation.start_date_local is not None
        and (
            mutation.start_date_local < today
            or (mutation.start_date_local - today).days > MAX_PLANNING_DAYS
        )
    )


def _bad_race_number(value: float | None, *, required: bool) -> bool:
    if value is None:
        return required
    return value <= 0


def _is_placeholder(mutation: Any) -> bool:
    if (
        isinstance(mutation, (CreateRace, UpdateRace))
        and mutation.distance is not None
        and mutation.distance <= 0
    ):
        return True
    if isinstance(mutation, CreateWorkout):
        return _bad_race_number(mutation.moving_time, required=True) or _bad_race_number(
            mutation.icu_training_load, required=False
        )
    if isinstance(mutation, UpdateWorkout):
        return _bad_race_number(mutation.moving_time, required=False) or _bad_race_number(
            mutation.icu_training_load, required=False
        )
    if isinstance(mutation, CreateRace):
        return _bad_race_number(mutation.moving_time, required=True) or _bad_race_number(
            mutation.icu_training_load, required=True
        )
    return isinstance(mutation, UpdateRace) and (
        _bad_race_number(mutation.moving_time, required=False)
        or _bad_race_number(mutation.icu_training_load, required=False)
    )


def _filter_valid_mutations(report: DecisionReport, *, today: date) -> tuple[list[Any], list[str]]:
    kept: list[Any] = []
    dropped: list[str] = []
    for mutation in report.mutations:
        if _is_stale(mutation, today=today):
            dropped.append("past-dated")
        elif _is_placeholder(mutation):
            dropped.append("placeholder")
        else:
            kept.append(mutation)
    return kept, dropped


def _reject_past_dates(report: DecisionReport, *, today: date) -> None:
    if any(_is_stale(mutation, today=today) for mutation in report.mutations):
        raise StaleDecisionError(
            "mutations must be dated between today and "
            f"{MAX_PLANNING_DAYS} days ahead; example dates are placeholders"
        )


def _reject_placeholders(report: DecisionReport) -> None:
    for mutation in report.mutations:
        if _is_placeholder(mutation):
            raise PlaceholderMutationError(
                "duration/load must be real values; ask the athlete instead of"
                " copying the example zeros"
            )


def _assert_valid_mutations(report: DecisionReport, *, today: date) -> None:
    _reject_placeholders(report)
    _reject_past_dates(report, today=today)


def _validate_report(payload: Any, *, today: date) -> DecisionReport:
    report = DecisionReport.model_validate(payload)
    _assert_valid_mutations(report, today=today)
    return report


class CoachEngine:
    def __init__(
        self,
        settings: Settings,
        store: CoachStore,
        read_client: IntervalsReadClient,
        llm_client: LlmClient,
        writer: CalendarWriter | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(ZoneInfo(settings.app_timezone)))
        self._store = store
        self._read_client = read_client
        self._llm_client = llm_client
        self._writer = writer

    async def _extract(
        self, focus: str, *, user_feedback: str | None, today: date | None
    ) -> CoachContext:
        deep_query = detect_deep_query(focus, today=today)
        if deep_query is not None:
            extractor = DeepHistoricalExtractor(self._settings, self._read_client)
            return await extractor.extract(
                focus, query=deep_query, user_feedback=user_feedback, today=today
            )
        return await StandardExtractor(self._settings, self._read_client).extract(
            focus, user_feedback=user_feedback, today=today
        )

    def _surface_unseen(self, context: CoachContext) -> CoachContext:
        unseen = self._store.unseen_activity_ids(
            activity.id for activity in context.recent_activities
        )
        if not unseen:
            return context
        new_activities = [
            activity for activity in context.recent_activities if activity.id in unseen
        ]
        listing = ", ".join(
            f"{activity.name} ({activity.start_date_local.date().isoformat()})"
            for activity in new_activities
        )
        try:
            return CoachContext.model_validate(
                {
                    **context.model_dump(),
                    "focus": f"{context.focus}\nNew activities since last review: {listing}",
                }
            )
        except ValidationError:
            return context

    @staticmethod
    def _fit_history(
        context: CoachContext,
        history: list[LlmMessage] | None,
        *,
        system_tokens: int,
    ) -> list[LlmMessage] | None:
        if not history:
            return history
        remaining = max(0, INPUT_TOKEN_CEILING - system_tokens - context.estimated_tokens())
        kept: list[LlmMessage] = []
        total = 0
        for turn in reversed(history):
            cost = estimate_text_tokens(turn.content)
            if total + cost > remaining:
                break
            kept.append(turn)
            total += cost
        ordered = list(reversed(kept))
        while ordered and ordered[0].role == "assistant":
            ordered.pop(0)
        if len(ordered) < len(history):
            logger.warning(
                "trimmed the conversation history from %d to %d turns for the context budget",
                len(history),
                len(ordered),
            )
        return ordered

    async def _run_llm(
        self, context: CoachContext, *, history: list[LlmMessage] | None = None
    ) -> DecisionReport:
        today = _today(context, self._settings)
        content = await self._llm_client.complete_json(
            build_messages(
                context,
                self._settings,
                self._fit_history(
                    context,
                    history,
                    system_tokens=estimate_text_tokens(system_prompt(self._settings)),
                ),
            ),
            validator=lambda payload: _validate_report(payload, today=today),
        )
        return DecisionReport.model_validate(json.loads(content))

    async def build_context(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        today: date | None = None,
    ) -> CoachContext:
        return self._surface_unseen(
            await self._extract(focus, user_feedback=user_feedback, today=today)
        )

    async def resolve_event_dates(
        self, context: CoachContext, mutations: Sequence[Mutation]
    ) -> dict[str, date]:
        """Map the event ids referenced by mutations to their calendar dates.

        Events fetched into the context resolve offline; ids outside the fetched
        window are looked up individually. Lookup failures are best effort and
        leave the mutation undated.
        """
        dates = _context_event_dates(context)
        missing: list[str] = []
        for mutation in mutations:
            event_id = getattr(mutation, "event_id", None)
            if event_id is None or getattr(mutation, "start_date_local", None) is not None:
                continue
            key = str(event_id)
            if key not in dates and key not in missing:
                missing.append(key)
        for event_id in missing:
            try:
                payload = await self._read_client.get_event(event_id)
                event = Event.model_validate(payload)
            except (IntervalsApiError, ValueError) as exc:
                # Expected: the API refused the id, or the payload did not validate
                # (pydantic's ValidationError is a ValueError). Leave it undated.
                logger.warning("could not resolve the date of event %s: %s", event_id, exc)
                continue
            except Exception:
                # A genuine defect must not stay invisible: log the traceback and
                # keep the proposal renderable rather than dropping it.
                logger.exception("unexpected error resolving the date of event %s", event_id)
                continue
            if event.id is not None:
                dates[str(event.id)] = event.start_date_local.date()
        return dates

    async def analyze(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        today: date | None = None,
        context: CoachContext | None = None,
        history: list[LlmMessage] | None = None,
    ) -> Draft:
        if context is None:
            context = await self.build_context(focus, user_feedback=user_feedback, today=today)
        report = await self._run_llm(context, history=history)
        draft_id = self._store.save_draft(
            focus=context.focus, report=report, context=context, user_feedback=user_feedback
        )
        self._store.mark_activities_seen(activity.id for activity in context.recent_activities)
        draft = self._store.get_draft(draft_id)
        assert draft is not None
        return draft

    def review(self, draft_id: int) -> Draft:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        return draft

    def today(self) -> date:
        return self._clock().date()

    def unapplied_decisions(self) -> list[Decision]:
        return self._store.list_unapplied_decisions()

    def llm_selection(self) -> tuple[str, str]:
        return (self._llm_client.provider_name, self._llm_client.model_name)

    def select_llm(
        self, *, provider: str | None = None, model: str | None = None
    ) -> tuple[str, str]:
        self._llm_client.select(provider=provider, model=model)
        return self.llm_selection()

    def prune_history(
        self, days: int | None = None, *, now: datetime | None = None
    ) -> dict[str, int]:
        cutoff = now or datetime.now(UTC)
        if days is not None:
            cutoff = cutoff - timedelta(days=days)
        return self._store.prune_before(cutoff)

    def recent_history(
        self, limit: int, *, max_age_days: int | None = None
    ) -> list[FeedbackWithReport]:
        return self._store.recent_feedback(limit, max_age_days=max_age_days)

    async def submit_feedback(
        self,
        draft_id: int,
        feedback: str,
        *,
        history: list[LlmMessage] | None = None,
    ) -> Draft:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.status != DraftStatus.PENDING:
            raise ValueError(
                f"draft {draft_id} is {draft.status.value}; only pending drafts accept feedback"
            )
        base = draft.context
        context = build_within_budget(
            base.focus,
            base.recent_activities,
            base.wellness,
            base.upcoming_events,
            base.sport_settings,
            goal_races=base.goal_races,
            training_rollup=base.training_rollup,
            current_proposal=draft.report,
            user_feedback=feedback,
            activity_detail=base.activity_detail,
            max_tokens=base.max_tokens,
            today=self.today(),
        )
        self._store.add_feedback(draft_id, feedback)
        report = await self._run_llm(context, history=history)
        self._store.update_draft_report(
            draft_id, report=report, user_feedback=feedback, context=context
        )
        updated = self._store.get_draft(draft_id)
        assert updated is not None
        return updated

    def _assert_current_dates(self, report: DecisionReport) -> None:
        _assert_valid_mutations(report, today=self.today())

    def discard_decision(self, decision_id: int) -> None:
        self._store.discard_decision(decision_id)

    def discard_stale_decisions(self) -> list[tuple[int, str]]:
        discarded: list[tuple[int, str]] = []
        today = self.today()
        for decision in self._store.list_unapplied_decisions():
            kept, dropped = _filter_valid_mutations(decision.report, today=today)
            if not dropped or kept:
                continue
            reason = (
                "placeholder values"
                if set(dropped) == {"placeholder"}
                else "dates that have passed"
            )
            self._store.discard_decision(decision.id)
            discarded.append((decision.id, reason))
        return discarded

    def approve(self, draft_id: int) -> Decision:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        self._assert_current_dates(draft.report)
        return self._store.approve_draft(draft_id)

    async def apply(self, decision_id: int | None = None) -> ApplyReport:
        """Apply approved decisions to the calendar.

        If a mutation fails, earlier mutations of the same decision stay applied while
        the decision remains unapplied; re-running is safe because mutations are
        idempotent (create resolves by name+date, update re-applies, delete skips).
        """
        if self._writer is None:
            raise RuntimeError("no calendar writer configured")
        if decision_id is not None:
            decision = self._store.get_decision(decision_id)
            if decision is None:
                raise ValueError(f"decision not found: {decision_id}")
            if decision.applied_at is not None:
                raise ValueError(f"decision {decision_id} is already applied")
            decisions = [decision]
        else:
            decisions = self._store.list_unapplied_decisions()
        applied: list[AppliedDecision] = []
        for decision in decisions:
            kept, dropped = _filter_valid_mutations(decision.report, today=self.today())
            if dropped and not kept:
                reason = ", ".join(sorted(set(dropped)))
                if set(dropped) == {"placeholder"}:
                    raise PlaceholderMutationError(
                        f"decision {decision.id} only contains placeholder mutations ({reason})"
                    )
                raise StaleDecisionError(
                    f"decision {decision.id} has no mutations left to apply ({reason})"
                )
            if dropped:
                logger.warning(
                    "skipping %d %s mutation(s) of decision %s",
                    len(dropped),
                    "/".join(sorted(set(dropped))),
                    decision.id,
                )
            outcomes = await self._writer.apply_decision(decision, mutations=kept)
            applied.append(
                AppliedDecision(decision_id=decision.id, outcomes=outcomes, skipped=dropped)
            )
            self._store.mark_decision_applied(decision.id)
        return ApplyReport(decisions=applied)
