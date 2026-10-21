"""Bounded API inputs and stable outputs."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
GenerationMode = Literal["fast", "deep"]
SearchDomain = Annotated[str, StringConstraints(strip_whitespace=True, max_length=120, pattern=r"^[^\x00-\x1f]*$")]
Category = Literal[
    "Balanced", "Foundational", "Survey", "Recent", "Implementation Focused",
    "Evaluation Focused", "Alternative Approach", "Contrarian View",
]


class SearchRequest(BaseModel):
    query: ShortText
    categories: list[Category] = Field(default_factory=list, max_length=8)
    domain: SearchDomain = ""


class Paper(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=1000)
    authors: list[Annotated[str, Field(max_length=300)]] = Field(max_length=100)
    abstract: str = Field(min_length=1, max_length=20000)
    link: str = Field(max_length=2000)
    date: str = Field(max_length=100)
    paper_id: str = Field(default="", max_length=2000)
    source: str = Field(default="arxiv", max_length=100)
    url: str = Field(default="", max_length=2000)
    published: str = Field(default="", max_length=100)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    generation_mode: GenerationMode = "fast"


class PapersRequest(BaseModel):
    papers: list[Paper] = Field(min_length=1, max_length=10)
    summaries: list[Annotated[str, Field(max_length=5000)]] = Field(default_factory=list, max_length=10)
    query: str = Field(default="", max_length=2000)
    domain: SearchDomain = ""
    generation_mode: GenerationMode = "fast"

    @model_validator(mode="after")
    def summaries_match_papers(self):
        if self.summaries and len(self.summaries) != len(self.papers):
            raise ValueError("Provide one summary per paper, or omit summaries")
        return self


class SearchResponse(BaseModel):
    papers: list[dict[str, Any]]
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)
    domain: str = ""
    search_query: str = ""


class GenerationResponse(BaseModel):
    generation: dict[str, Any] = Field(default_factory=dict)


class SummaryResponse(GenerationResponse):
    summary: str


class GuidanceResponse(GenerationResponse):
    guidance: str


class ResearchPlanResponse(GenerationResponse):
    research_plan: str
