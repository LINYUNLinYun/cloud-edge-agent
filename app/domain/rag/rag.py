"""RAG pipeline abstractions — chunking, embedding, retrieval, reranking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """A document or document chunk for RAG."""

    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


@dataclass
class RetrievalResult:
    """A retrieved document with relevance score."""

    document: Document
    score: float


class Chunker(ABC):
    """Split documents into chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """Split a document into chunks, each returned as a Document."""
        ...


class Embedder(ABC):
    """Compute embeddings for documents."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        ...


class Retriever(ABC):
    """Retrieve relevant documents from a vector store."""

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        """Retrieve top-k relevant documents for a query."""
        ...


class Reranker(ABC):
    """Rerank retrieved documents for better relevance."""

    @abstractmethod
    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Rerank and return top-k results."""
        ...
