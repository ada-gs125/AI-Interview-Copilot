from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


SchemaT = TypeVar("SchemaT")


@dataclass(frozen=True)
class PromptMessages:
    system: str
    user: str


@dataclass(frozen=True)
class SkillSpec:
    name: str
    version: str
    output_schema: str


@dataclass(frozen=True)
class LLMSkill(Generic[SchemaT]):
    name: str
    version: str
    output_schema: type[SchemaT]
    build_prompt: Callable[..., PromptMessages]

    def prompt(self, **kwargs: Any) -> PromptMessages:
        return self.build_prompt(**kwargs)

    def spec(self) -> SkillSpec:
        return SkillSpec(
            name=self.name,
            version=self.version,
            output_schema=self.output_schema.__name__,
        )
