"""Request/response models. They are the API contract (and what /docs shows), so the response
mirrors answer()'s dict exactly; FastAPI validates every response against them."""

from pydantic import BaseModel, Field

from hybrid_rag.retrieval.retriever import Mode


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=1000, examples=["What does section 148A provide?"]
    )
    mode: Mode = "hybrid"


class Citation(BaseModel):
    doc: str
    page: str = Field(examples=["p.12-13"])
    chunk_id: str
    snippet: str


class ChunkDebug(BaseModel):
    rank: int
    chunk_id: str
    vector_rank: int | None
    bm25_rank: int | None
    rrf_rank: int | None
    rerank_score: float | None
    score: float


class RetrievalDebug(BaseModel):
    mode: Mode
    chunks: list[ChunkDebug]


class Latency(BaseModel):
    retrieve: int
    generate: int
    total: int


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    invalid_citations: list[int] = Field(
        description="source numbers the LLM cited but was never given (dropped from the answer)"
    )
    uncited: bool = Field(description="answered without citing anything; treat as unverified")
    retrieval_debug: RetrievalDebug
    latency_ms: Latency


class Health(BaseModel):
    status: str
    chunks_indexed: int
    llm: str
    reranker: str
    ingest_running: bool


class IngestStatus(BaseModel):
    running: bool
    last_result: str | None
