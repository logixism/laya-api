from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from laya_api.config import (
    MAX_INSTRUCTIONS_LENGTH,
    MAX_OPTION_COUNT,
    MAX_OPTION_DESCRIPTION_LENGTH,
    MAX_OPTION_LENGTH,
    MAX_QUESTIONS,
    MAX_QUESTION_ID_LENGTH,
    MAX_STATE_BYTES,
)


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str = Field(..., min_length=1, max_length=MAX_INSTRUCTIONS_LENGTH)
    criteria: list[str] | dict[str, str | None] = Field(
        description=(
            "Options as a plain list, or a mapping of option to optional "
            "description that gives the option semantic meaning."
        ),
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(
        cls,
        value: list[str] | dict[str, str | None],
    ) -> list[str] | dict[str, str | None]:
        if not value:
            raise ValueError("choice criteria must not be empty")

        if len(value) > MAX_OPTION_COUNT:
            raise ValueError(
                f"choice criteria cannot contain more than {MAX_OPTION_COUNT} options"
            )

        if isinstance(value, list):
            if len(set(value)) != len(value):
                raise ValueError("choice criteria must contain unique options")

            for option in value:
                if not option.strip():
                    raise ValueError("choice options cannot be empty")

                if len(option) > MAX_OPTION_LENGTH:
                    raise ValueError(
                        f"choice option cannot exceed {MAX_OPTION_LENGTH} characters"
                    )

        else:
            for option, description in value.items():
                if not option.strip():
                    raise ValueError("choice option names cannot be empty")

                if len(option) > MAX_OPTION_LENGTH:
                    raise ValueError(
                        f"choice option cannot exceed {MAX_OPTION_LENGTH} characters"
                    )

                if description is not None and len(description) > MAX_OPTION_DESCRIPTION_LENGTH:
                    raise ValueError(
                        f"choice option description cannot exceed "
                        f"{MAX_OPTION_DESCRIPTION_LENGTH} characters"
                    )

        return value


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: str = Field(..., min_length=1, max_length=MAX_INSTRUCTIONS_LENGTH)
    criteria: list[str] = Field(
        description="Ordered score levels from lowest to highest.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("score criteria must contain at least 2 levels")

        if len(value) > MAX_OPTION_COUNT:
            raise ValueError(
                f"score criteria cannot contain more than {MAX_OPTION_COUNT} levels"
            )

        for criterion in value:
            if not criterion.strip():
                raise ValueError("score criteria cannot contain empty levels")

            if len(criterion) > MAX_OPTION_DESCRIPTION_LENGTH:
                raise ValueError(
                    f"score criterion cannot exceed "
                    f"{MAX_OPTION_DESCRIPTION_LENGTH} characters"
                )

        return value


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str = Field(..., min_length=1, max_length=MAX_INSTRUCTIONS_LENGTH)
    criteria: dict[str, str | None] | list[str] | None = Field(
        default=None,
        description=(
            "Optional. Criteria may be useful for giving semantic meaning "
            "to true/false."
        ),
    )


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] = Field(
        description="Conversation state to evaluate; free text or a JSON object.",
    )
    questions: dict[
        Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_QUESTION_ID_LENGTH,
                pattern=r"^[A-Za-z0-9_.-]+$",
            ),
        ],
        Question,
    ] = Field(
        description="Questions to answer, keyed by question id.",
    )

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("state must not be empty")

            encoded = value.encode("utf-8")

        else:
            try:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("state must contain JSON-serializable data") from exc

        if len(encoded) > MAX_STATE_BYTES:
            raise ValueError(f"state cannot exceed {MAX_STATE_BYTES} bytes")

        return value

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, value: dict[str, Question]) -> dict[str, Question]:
        if not value:
            raise ValueError("questions must not be empty")

        if len(value) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions are allowed")

        return value
