import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from open_endurance_coach.chat.history import count_exchanges
from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings, effective_input_budget, resolved_input_room
from open_endurance_coach.errors import InternalError
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import DEFAULT_MAX_TOKENS, StandardExtractor
from open_endurance_coach.prompts.prompts import (
    CHAT_ONLY_FALLBACK,
    build_messages,
    estimate_user_message_tokens,
    system_prompt,
)
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
    FeedbackWithReport,
    Proposal,
    ProposalStatus,
)
from open_endurance_coach.tokens import CHARS_PER_TOKEN, estimate_text_tokens
from open_endurance_coach.writer.calendar import CalendarWriter
from open_endurance_coach.writer.records import AppliedProposal, ApplyReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackOutcome:
    """Result of a feedback turn.

    ``proposal`` is the persisted revision when the model returned a plan, or the
    unchanged pending proposal when the turn was conversational or still blocked on
    material questions; ``report`` always carries the turn's report for display.
    """

    proposal: Proposal
    report: DecisionReport


def _today(context: CoachContext, settings: Settings) -> date:
    if context.today is not None:
        return context.today
    return datetime.now(ZoneInfo(settings.app_timezone)).date()


def _context_event_dates(context: CoachContext) -> dict[str, date]:
    dates: dict[str, date] = {}
    for race in context.goal_races:
        if race.event_id is not None:
            dates[str(race.event_id)] = race.date
    for event in (*context.recent_events, *context.upcoming_events):
        if event.id is not None:
            dates[str(event.id)] = event.start_date_local.date()
    return dates


MAX_PLANNING_DAYS = 400

# Slack in the message cap for the prompt boilerplate around the data payload.
PROMPT_OVERHEAD_TOKENS = 256


class StaleProposalError(ValueError):
    """The proposal's dated mutations are in the past."""


class PlaceholderMutationError(ValueError):
    """The proposal still carries example placeholder values."""


class StateDriftError(ValueError):
    """Some of the proposal's mutations are no longer valid."""


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
    return not math.isfinite(value) or value <= 0


def _is_placeholder(mutation: Any) -> bool:
    if isinstance(mutation, CreateWorkout):
        return (
            _bad_race_number(mutation.moving_time, required=True)
            or _bad_race_number(mutation.distance, required=False)
            or _bad_race_number(mutation.icu_training_load, required=False)
        )
    if isinstance(mutation, UpdateWorkout):
        return (
            _bad_race_number(mutation.moving_time, required=False)
            or _bad_race_number(mutation.distance, required=False)
            or _bad_race_number(mutation.icu_training_load, required=False)
        )
    if isinstance(mutation, CreateRace):
        return (
            _bad_race_number(mutation.moving_time, required=True)
            or _bad_race_number(mutation.distance, required=False)
            or _bad_race_number(mutation.icu_training_load, required=True)
        )
    return isinstance(mutation, UpdateRace) and (
        _bad_race_number(mutation.moving_time, required=False)
        or _bad_race_number(mutation.distance, required=False)
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


def _drop_reason(dropped: list[str]) -> str:
    reasons = set(dropped)
    if reasons == {"placeholder"}:
        return "placeholder values"
    if reasons == {"past-dated"}:
        return "dates that have passed"
    return "mutations that no longer match the plan"


def _reject_past_dates(report: DecisionReport, *, today: date) -> None:
    if any(_is_stale(mutation, today=today) for mutation in report.mutations):
        raise StaleProposalError(
            "mutations must be dated between today and "
            f"{MAX_PLANNING_DAYS} days ahead; example dates are placeholders"
        )


def _reject_placeholders(report: DecisionReport) -> None:
    for mutation in report.mutations:
        if _is_placeholder(mutation):
            raise PlaceholderMutationError(
                "duration/load/distance must be real values; ask the athlete instead of"
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
        budget: int,
    ) -> list[LlmMessage] | None:
        if not history:
            return history
        remaining = max(
            0,
            budget
            - system_tokens
            - estimate_user_message_tokens(context)
            - estimate_text_tokens(CHAT_ONLY_FALLBACK),
        )
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
        before = count_exchanges(history)
        after = count_exchanges(ordered)
        if after < before:
            logger.info(
                "trimmed the conversation history from %d to %d exchanges for the context budget",
                before,
                after,
            )
        elif len(ordered) < len(history):
            logger.info("trimmed the oldest messages from the conversation for the context budget")
        return ordered

    def _assert_within_ceiling(self, messages: list[LlmMessage]) -> None:
        budget = effective_input_budget(self._settings)
        total = sum(estimate_text_tokens(message.content) for message in messages)
        if total > budget:
            soft = self._settings.llm_input_budget
            advice = (
                "raise LLM_INPUT_BUDGET"
                if soft is not None and soft < resolved_input_room(self._settings)
                else "select a model with a larger window"
            )
            raise ValueError(
                f"request too large: about {total} tokens; the budget is {budget} for"
                f" {self._settings.llm_model!r}; shorten the message or {advice}"
            )

    async def _run_llm(
        self, context: CoachContext, *, history: list[LlmMessage] | None = None
    ) -> DecisionReport:
        today = _today(context, self._settings)
        budget = effective_input_budget(self._settings)
        system_tokens = estimate_text_tokens(system_prompt(self._settings))
        messages = build_messages(
            context,
            self._settings,
            self._fit_history(context, history, system_tokens=system_tokens, budget=budget),
        )
        self._assert_within_ceiling(messages)
        validated: list[DecisionReport] = []

        def validate(payload: Any) -> None:
            validated.append(_validate_report(payload, today=today))

        try:
            await self._llm_client.complete_json(messages, validator=validate)
        except LlmError:
            return await self._chat_only_fallback(messages, today=today)
        return validated[0]

    async def _chat_only_fallback(
        self, messages: list[LlmMessage], *, today: date
    ) -> DecisionReport:
        fallback = [*messages, LlmMessage(role="user", content=CHAT_ONLY_FALLBACK)]
        self._assert_within_ceiling(fallback)
        validated: list[DecisionReport] = []

        def validate(payload: Any) -> None:
            report = _validate_report(payload, today=today)
            if report.mutations:
                raise ValueError("the fallback answer must not contain mutations")
            validated.append(report)

        await self._llm_client.complete_json(fallback, max_attempts=2, validator=validate)
        return validated[0]

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
    ) -> Proposal:
        self.check_focus(focus)
        if context is None:
            context = await self.build_context(focus, user_feedback=user_feedback, today=today)
        report = await self._run_llm(context, history=history)
        proposal_id = self._store.save_proposal(
            focus=context.focus, report=report, context=context, user_feedback=user_feedback
        )
        self._store.mark_activities_seen(activity.id for activity in context.recent_activities)
        proposal = self._store.get_proposal(proposal_id)
        assert proposal is not None
        return proposal

    def review(self, proposal_id: int) -> Proposal:
        proposal = self._store.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        return proposal

    def today(self) -> date:
        return self._clock().date()

    def input_room(self) -> int:
        """Tokens a request can spend on the conversation: message plus history.

        That is the derived input budget minus the parts that are never negotiable: the
        system prompt, the reserved athlete-data budget and the prompt boilerplate. Raises
        when the budget cannot cover them, rather than silently squeezing the athlete out.
        """
        return self._input_room_for(self._settings)

    def _input_room_for(self, settings: Settings) -> int:
        budget = effective_input_budget(settings)
        system_tokens = estimate_text_tokens(system_prompt(settings))
        room = budget - system_tokens - DEFAULT_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS
        if room < 1:
            raise ValueError(
                f"the input budget ({budget} tokens) is too small: the system prompt"
                f" (~{system_tokens} tokens), the {DEFAULT_MAX_TOKENS}-token athlete-data"
                f" reserve and {PROMPT_OVERHEAD_TOKENS} tokens of overhead leave no room"
                " for the conversation; raise LLM_INPUT_BUDGET, lower LLM_MAX_TOKENS, or"
                " select a model with a larger window"
            )
        return room

    def focus_limit(self) -> int:
        """Max tokens a single athlete message may occupy in a request."""
        return self.input_room()

    def history_budget(self) -> int:
        """Tokens the in-session conversation may occupy.

        The same room as the message: data and history share what is left after the system
        prompt and the reserved data budget. It assumes the full data reserve and no current
        message, so it is an upper bound: a very large message can make ``_fit_history`` trim
        slightly earlier than this cap implies. ``_fit_history`` still trims to the exact
        remainder per request, so this only bounds what the session keeps in memory.
        """
        return self.input_room()

    def check_focus(self, focus: str) -> None:
        """Reject a message too large to send, rather than truncating it silently."""
        limit = self.focus_limit()
        tokens = estimate_text_tokens(focus)
        if tokens > limit:
            raise ValueError(
                f"message too long: about {tokens} tokens; keep it under {limit} tokens"
                f" (roughly {limit * CHARS_PER_TOKEN} characters)"
            )

    def unapplied_proposals(self) -> list[Proposal]:
        return self._store.list_unapplied_proposals()

    def llm_selection(self) -> tuple[str, str]:
        return (self._llm_client.provider_name, self._llm_client.model_name)

    def select_llm(
        self, *, provider: str | None = None, model: str | None = None
    ) -> tuple[str, str]:
        candidate = self._llm_client.preview(provider=provider, model=model)
        self._input_room_for(candidate)
        self._settings = self._llm_client.select(provider=provider, model=model)
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

    def _context_around(
        self,
        base: CoachContext,
        *,
        focus: str,
        proposal: DecisionReport | None,
        user_feedback: str | None,
        today: date,
    ) -> CoachContext:
        return build_within_budget(
            focus,
            base.recent_activities,
            base.wellness,
            base.upcoming_events,
            base.sport_settings,
            goal_races=base.goal_races,
            training_rollup=base.training_rollup,
            recent_events=base.recent_events,
            current_proposal=proposal,
            user_feedback=user_feedback,
            activity_detail=base.activity_detail,
            activity_splits=base.activity_splits,
            max_tokens=base.max_tokens,
            today=today,
        )

    def refocus_context(
        self, base: CoachContext, focus: str, *, today: date | None = None
    ) -> CoachContext:
        """Reuse cached athlete data for a new message within the context budget.

        A raw copy would let a long message push the estimate over ``max_tokens``
        and break the proposal on reload, so the data is trimmed to fit.
        """
        return self._context_around(
            base,
            focus=focus,
            proposal=None,
            user_feedback=None,
            today=today or self.today(),
        )

    async def submit_feedback(
        self,
        proposal_id: int,
        feedback: str,
        *,
        focus: str | None = None,
        assume: bool = False,
        history: list[LlmMessage] | None = None,
    ) -> FeedbackOutcome:
        proposal = self._store.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id} is {proposal.status.value};"
                " only pending proposals accept feedback"
            )
        self.check_focus(focus if focus is not None else feedback)
        context = self._context_around(
            proposal.context,
            focus=focus or proposal.context.focus,
            proposal=proposal.report,
            # The message is the focus when the caller passes it, so do not also
            # charge it as user_feedback: that section counts against the data
            # budget and would evict real athlete data for long messages.
            user_feedback=None if focus is not None else feedback,
            today=self.today(),
        )
        feedback_id = self._store.add_feedback(proposal_id, feedback)
        report = await self._run_llm(context, history=history)
        self._store.set_feedback_report(feedback_id, report)
        if report.intent != "plan" or (report.needs_input and not assume):
            # Conversational turn or a plan still blocked on material questions:
            # keep the pending proposal and only carry the report for display.
            return FeedbackOutcome(proposal=proposal, report=report)
        self._store.update_proposal_report(
            proposal_id, report=report, user_feedback=feedback, context=context
        )
        updated = self._store.get_proposal(proposal_id)
        assert updated is not None
        return FeedbackOutcome(proposal=updated, report=report)

    def _assert_current_dates(self, report: DecisionReport) -> None:
        _assert_valid_mutations(report, today=self.today())

    def discard_proposal(self, proposal_id: int) -> None:
        self._store.discard_proposal(proposal_id)

    def discard_stale_proposals(self) -> list[tuple[int, str]]:
        discarded: list[tuple[int, str]] = []
        today = self.today()
        for proposal in self._store.list_unapplied_proposals():
            approved = proposal.approved_report or proposal.report
            _, dropped = _filter_valid_mutations(approved, today=today)
            if not dropped:
                continue
            reason = _drop_reason(dropped)
            self._store.discard_proposal(proposal.id)
            discarded.append((proposal.id, reason))
        return discarded

    def approve(self, proposal_id: int, *, keep: Sequence[int] | None = None) -> Proposal:
        """Approve a pending proposal; ``keep`` limits the proposal to those mutation indices."""
        proposal = self._store.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        report = proposal.report
        if keep is not None:
            report = report.model_copy(
                update={
                    "mutations": [
                        mutation for index, mutation in enumerate(report.mutations) if index in keep
                    ]
                }
            )
        self._assert_current_dates(report)
        return self._store.approve_proposal(proposal_id, report=report)

    def reject_proposal(self, proposal_id: int) -> None:
        self._store.reject_proposal(proposal_id)

    async def apply(self, proposal_id: int | None = None) -> ApplyReport:
        """Apply approved proposals to the calendar.

        If a mutation fails, earlier mutations of the same proposal stay applied while
        the proposal remains unapplied; re-running is safe because mutations are
        idempotent (create resolves by name+date, update re-applies, delete skips).

        An approval is atomic: a proposal with a stale or placeholder mutation is
        rejected with ``StateDriftError`` and nothing is written.
        """
        if self._writer is None:
            raise InternalError("no calendar writer configured")
        if proposal_id is not None:
            proposal = self._store.get_proposal(proposal_id)
            if proposal is None:
                raise ValueError(f"proposal not found: {proposal_id}")
            if proposal.applied_at is not None:
                raise ValueError(f"proposal {proposal_id} is already applied")
            proposals = [proposal]
        else:
            proposals = self._store.list_unapplied_proposals()
        applied: list[AppliedProposal] = []
        for proposal in proposals:
            approved = proposal.approved_report or proposal.report
            kept, dropped = _filter_valid_mutations(approved, today=self.today())
            if dropped:
                reason = _drop_reason(dropped)
                if not kept:
                    if set(dropped) == {"placeholder"}:
                        raise PlaceholderMutationError(
                            f"proposal {proposal.id} only contains placeholder mutations ({reason})"
                        )
                    raise StaleProposalError(
                        f"proposal {proposal.id} has no mutations left to apply ({reason})"
                    )
                logger.warning(
                    "rejecting proposal %s: %d %s mutation(s) dropped (%s); nothing written",
                    proposal.id,
                    len(dropped),
                    "/".join(sorted(set(dropped))),
                    reason,
                )
                raise StateDriftError(
                    f"proposal {proposal.id} was rejected: {len(dropped)} mutation(s)"
                    f" dropped ({reason}); nothing was written, ask for an updated plan"
                )
            outcomes = await self._writer.apply_proposal(proposal, mutations=kept)
            applied.append(AppliedProposal(proposal_id=proposal.id, outcomes=outcomes))
            self._store.mark_proposal_applied(proposal.id)
        return ApplyReport(proposals=applied)
