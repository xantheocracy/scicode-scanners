"""Validated causal assessments and investigation limits."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["underspecified", "wrongly_specified", "model_error", "other"]


class Cause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category
    mechanism: str = Field(min_length=1)
    origin_type: Literal["current", "earlier", "author_provided", "external", "unknown"]
    origin_steps: list[str]
    dependency_path: list[str]
    evidence: list[str] = Field(
        min_length=1, description="Stable M/E/T references and quotations."
    )
    causal_contribution: str = Field(
        min_length=1,
        description="Why this defect causes this failure; counterfactual evidence if available.",
    )
    defect_key: str = Field(
        min_length=1,
        description="Specific defect, reused verbatim for the same cause in later steps.",
    )


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["resolved", "partially_resolved", "unresolved"]
    causes: list[Cause]
    explanation: str = Field(min_length=1)
    limitations: list[str]
    alternatives_considered: list[str]

    @model_validator(mode="after")
    def consistent(self):
        if self.status != "unresolved" and not self.causes:
            raise ValueError(
                "A resolved/partially resolved assessment needs supported causes."
            )
        if self.status == "unresolved" and self.causes:
            raise ValueError("Use partially_resolved when some causes are supported.")
        return self


class Limits(BaseModel):
    generated_token_budget: int = Field(default=8192, ge=2048)
    finalization_reserve: int = Field(default=1536, ge=512)
    per_call_tokens: int = Field(default=4096, ge=512)
    tool_rounds: int = Field(default=6, ge=1)
    reasoning_effort: Literal["low", "high", "max"] = "high"

    @model_validator(mode="after")
    def reserve_fits(self):
        if self.finalization_reserve >= self.generated_token_budget:
            raise ValueError(
                "Finalization reserve must be smaller than the total budget."
            )
        return self
