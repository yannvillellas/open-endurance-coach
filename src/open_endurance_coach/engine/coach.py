import json
from dataclasses import dataclass
from datetime import date

from open_endurance_coach.clients.llm import LlmClient
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor
from open_endurance_coach.prompts.prompts import build_messages
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport, WorkoutMutation
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import Decision, Draft, DraftStatus


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
    ) -> None:
        self._settings = settings
        self._store = store
        self._read_client = read_client
        self._llm_client = llm_client

    async def _extract(
        self, focus: str, *, user_feedback: str | None, today: date | None
    ) -> CoachContext:
        extractor: StandardExtractor | DeepHistoricalExtractor
        if detect_deep_query(focus) is not None:
            extractor = DeepHistoricalExtractor(self._settings, self._read_client)
        else:
            extractor = StandardExtractor(self._settings, self._read_client)
        return await extractor.extract(focus, user_feedback=user_feedback, today=today)

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
        return context.model_copy(
            update={"focus": f"{context.focus}\nNew activities since last review: {listing}"}
        )

    async def _run_llm(self, context: CoachContext) -> DecisionReport:
        content = await self._llm_client.complete_json(
            build_messages(context, self._settings),
            validator=lambda payload: DecisionReport.model_validate(payload),
        )
        return DecisionReport.model_validate(json.loads(content))

    async def analyze(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        today: date | None = None,
    ) -> Draft:
        context = self._surface_unseen(
            await self._extract(focus, user_feedback=user_feedback, today=today)
        )
        report = await self._run_llm(context)
        draft_id = self._store.save_draft(
            focus=context.focus, report=report, context=context, user_feedback=user_feedback
        )
        for activity in context.recent_activities:
            self._store.mark_activity_seen(activity.id)
        draft = self._store.get_draft(draft_id)
        assert draft is not None
        return draft

    def review(self, draft_id: int) -> ReviewView:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        return ReviewView(draft=draft, requested_feedback=self._solicitations(draft.context))

    def pending_drafts(self) -> list[Draft]:
        return self._store.list_drafts(DraftStatus.PENDING)

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
        self._store.add_feedback(draft_id, feedback)
        context = draft.context.model_copy(update={"user_feedback": feedback})
        report = await self._run_llm(context)
        self._store.update_draft_report(draft_id, report=report, user_feedback=feedback)
        updated = self._store.get_draft(draft_id)
        assert updated is not None
        return updated

    def approve(self, draft_id: int, *, mutations: list[WorkoutMutation] | None = None) -> Decision:
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if mutations is not None:
            overridden = DecisionReport(
                summary=draft.report.summary,
                findings=draft.report.findings,
                questions=[],
                mutations=mutations,
            )
            self._store.update_draft_report(
                draft_id, report=overridden, user_feedback=draft.user_feedback
            )
        return self._store.approve_draft(draft_id)

    def reject(self, draft_id: int) -> None:
        self._store.reject_draft(draft_id)
