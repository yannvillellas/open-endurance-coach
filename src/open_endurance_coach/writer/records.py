from dataclasses import dataclass, field


@dataclass(frozen=True)
class MutationOutcome:
    action: str
    target: str
    event_id: int | str | None = None
    name: str | None = None


@dataclass(frozen=True)
class AppliedDecision:
    decision_id: int
    outcomes: list[MutationOutcome]


@dataclass(frozen=True)
class ApplyReport:
    decisions: list[AppliedDecision] = field(default_factory=list)
