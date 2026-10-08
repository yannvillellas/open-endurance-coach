from dataclasses import dataclass, field


@dataclass(frozen=True)
class MutationOutcome:
    action: str
    target: str
    event_id: int | str | None = None
    name: str | None = None
    drift: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppliedProposal:
    proposal_id: int
    outcomes: list[MutationOutcome]


@dataclass(frozen=True)
class ApplyReport:
    proposals: list[AppliedProposal] = field(default_factory=list)
