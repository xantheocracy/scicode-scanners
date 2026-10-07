"""Reports describe discovered evidence, not known benchmark defect labels."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Implementation = Literal["scicode", "scicode_verified"]
Direction = Literal["false_rejection", "false_acceptance"]
Status = Literal["demonstrated", "suspected", "not_found", "inconclusive"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Confidence(StrictModel):
    rating: Literal["low", "medium", "high"]
    rationale: str = Field(min_length=1)


class Finding(StrictModel):
    step_id: str
    direction: Direction
    status: Status
    claim: str = Field(min_length=1)
    defect_key: str = Field(min_length=1, description="Reuse for the same root cause.")
    category: str
    origin_steps: list[str]
    prompt_test_references: list[str]
    candidate_evidence: str | None = None
    semantic_evidence: list[str] = Field(default_factory=list)
    semantic_argument: str = ""
    plausibility_rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    semantic_confidence: Confidence
    grader_confidence: Confidence

    @model_validator(mode="after")
    def strong_evidence(self):
        if self.status == "demonstrated":
            if not self.candidate_evidence or not self.semantic_argument:
                raise ValueError(
                    "Demonstrated findings need candidate evidence and a semantic argument."
                )
            if not self.prompt_test_references or not self.origin_steps:
                raise ValueError(
                    "Demonstrated findings need specification/test references and origins."
                )
            if self.direction == "false_acceptance" and (
                not self.semantic_evidence or not self.plausibility_rationale
            ):
                raise ValueError(
                    "False acceptance needs an executable counterexample and plausibility rationale."
                )
        return self


class Report(StrictModel):
    implementation: Implementation
    problem_id: str
    summary: str
    findings: list[Finding]
    uncovered_steps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Limits(StrictModel):
    generated_tokens: int = Field(default=100_000, ge=512)
    finalization_reserve: int = Field(default=5_000, ge=256)
    per_call_tokens: int = Field(default=8_000, ge=256)
    tool_calls: int = Field(default=140, ge=1)
    model_calls: int = Field(default=150, ge=2)
    diagnostic_timeout: int = Field(default=120, ge=1)
    grading_timeout: int = Field(default=1800, ge=1)
    reasoning_effort: Literal["low", "high", "max"] = "high"

    @model_validator(mode="after")
    def reserve_fits(self):
        if self.finalization_reserve >= self.generated_tokens:
            raise ValueError(
                "Finalization reserve must be smaller than the generation budget."
            )
        return self
