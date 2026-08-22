import json
from dataclasses import dataclass
from datetime import date

from pydantic import ValidationError

from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor
from open_endurance_coach.prompts.chat import build_chat_messages
from open_endurance_coach.prompts.prompts import build_messages
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport, WorkoutMutation
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import (
    Decision,
    Draft,
    DraftStatus,
    FeedbackWithReport,
)
from open_endurance_coach.writer.calendar import CalendarWriter
from open_endurance_coach.writer.records import AppliedDecision, ApplyReport


@dataclass(frozen=True)
class ReviewView:
    draft: Draft
    requested_feedback: list[str]


class CoachEngine:
    def __init__(
        self,
        settings: Settings,
        store: CoachStore,
        read_client: IntervalsReadClient,
        llm_client: LlmClient,
        writer: CalendarWriter | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._read_client = read_client
        self._llm_client = llm_client
        self._writer = writer

    async def _extract(
        self, focus: str, *, user_feedback: str | None, today: date | None
    ) -> CoachContext:
        deep_query = detect_deep_query(focus)
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

    async def _run_llm(self, context: CoachContext) -> DecisionReport:
        content = await self._llm_client.complete_json(
            build_messages(context, self._settings),
            validator=lambda payload: DecisionReport.model_validate(payload),
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

    async def analyze(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        today: date | None = None,
    ) -> Draft:
        context = await self.build_context(focus, user_feedback=user_feedback, today=today)
        report = await self._run_llm(context)
        draft_id = self._store.save_draft(
            focus=context.focus, report=report, context=context, user_feedback=user_feedback
        )
        self._store.mark_activities_seen(activity.id for activity in context.recent_activities)
        draft = self._store.get_draft(draft_id)
        assert draft is not None
        return draft

    async def converse(
        self,
        text: str,
        *,
        history: list[LlmMessage] | None = None,
        context: CoachContext | None = None,
        today: date | None = None,
    ) -> str:
        if context is None:
            context = self._surface_unseen(
                await self._extract(text, user_feedback=None, today=today)
            )
        messages = build_chat_messages(context, self._settings, history=history, text=text)
        completion = await self._llm_client.complete(messages, json_mode=False)
        if not completion.content.strip():
            raise LlmError("empty content returned")
        return completion.content

    def review(self, draft_id: int) -> ReviewView:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        return ReviewView(draft=draft, requested_feedback=self._solicitations(draft.context))

    def pending_drafts(self) -> list[Draft]:
        return self._store.list_drafts(DraftStatus.PENDING)

    def recent_history(self, limit: int) -> list[FeedbackWithReport]:
        return self._store.recent_feedback(limit)

    @staticmethod
    def _solicitations(context: CoachContext) -> list[str]:
        missing = [
            activity
            for activity in context.recent_activities
            if activity.icu_rpe is None
            and activity.perceived_exertion is None
            and activity.session_rpe is None
        ]
        lines = [
            f"RPE missing for {activity.name} on {activity.start_date_local.date().isoformat()}:"
            " how hard did it feel (1-10)?"
            for activity in missing
        ]
        if missing:
            lines.append("Fueling: carb intake and hydration for the sessions above?")
        return lines

    async def submit_feedback(self, draft_id: int, feedback: str) -> Draft:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.status != DraftStatus.PENDING:
            raise ValueError(
                f"draft {draft_id} is {draft.status.value}; only pending drafts accept feedback"
            )
        context = CoachContext.model_validate(
            {
                **draft.context.model_dump(),
                "user_feedback": feedback,
                "current_proposal": draft.report,
            }
        )
        self._store.add_feedback(draft_id, feedback)
        report = await self._run_llm(context)
        self._store.update_draft_report(
            draft_id, report=report, user_feedback=feedback, context=context
        )
        updated = self._store.get_draft(draft_id)
        assert updated is not None
        return updated

    def approve(self, draft_id: int, *, mutations: list[WorkoutMutation] | None = None) -> Decision:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if mutations is not None:
            overridden = DecisionReport(
                summary="Mutations overridden by the athlete.",
                findings=[],
                questions=[],
                mutations=mutations,
            )
            self._store.update_draft_report(
                draft_id, report=overridden, user_feedback=draft.user_feedback
            )
        return self._store.approve_draft(draft_id)

    def reject(self, draft_id: int) -> None:
        self._store.reject_draft(draft_id)

    async def apply(self, decision_id: int | None = None, *, dry_run: bool = False) -> ApplyReport:
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
            outcomes = await self._writer.apply_decision(decision, dry_run=dry_run)
            applied.append(AppliedDecision(decision_id=decision.id, outcomes=outcomes))
            if not dry_run:
                self._store.mark_decision_applied(decision.id)
        return ApplyReport(decisions=applied)
