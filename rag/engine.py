"""PDF ingest, Chroma / BM25 retrieval, and RAG orchestration."""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .ollama_client import OllamaClient

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "streamlit_pdf_chunks"
IndexMode = Literal["vector", "bm25", "hybrid"]


class _BM25Okapi:
    """Minimal BM25 Okapi scorer (no external dependency)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.doc_freqs: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in corpus:
            freqs: dict[str, int] = {}
            for word in doc:
                freqs[word] = freqs.get(word, 0) + 1
            self.tf.append(freqs)
            for word in freqs:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        self.N = len(corpus)

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        if self.N == 0 or self.avgdl == 0:
            return scores
        for q in query:
            df = self.doc_freqs.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, freqs in enumerate(self.tf):
                f = freqs.get(q, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / self.avgdl
                )
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class RAGEngine:
    def __init__(
        self,
        *,
        chroma_dir: str | Path,
        ollama: OllamaClient,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.ollama = ollama
        self.collection_name = collection_name
        self._embedder: SentenceTransformer | None = None
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: _BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_texts: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []
        self._preview: list[dict[str, Any]] = []
        self.index_modes: set[IndexMode] = set()
        self.retrieval_mode: IndexMode | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    @property
    def chunk_count(self) -> int:
        if self._preview:
            return len(self._preview)
        if self._bm25_texts:
            return len(self._bm25_texts)
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_texts = []
        self._bm25_metadatas = []
        self._preview = []
        self.index_modes = set()
        self.retrieval_mode = None

    def ingest_pdf_bytes(
        self,
        file_name: str,
        raw: bytes,
        *,
        chunk_size: int = 600,
        chunk_overlap: int = 200,
        modes: set[IndexMode] | None = None,
        clear_first: bool = False,
    ) -> dict[str, Any]:
        selected = modes or {"vector"}
        if clear_first:
            self.clear()
            self.index_modes = set(selected)
            self.retrieval_mode = _resolve_retrieval_mode(selected)

        suffix = Path(file_name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        try:
            loader = PyPDFLoader(str(tmp_path))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = file_name

            overlap = min(chunk_overlap, max(chunk_size - 1, 0))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            chunks = splitter.split_documents(documents)
            if not chunks:
                return {"pages": 0, "chunks": 0, "file_name": file_name}

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [_clean_metadata(chunk.metadata) for chunk in chunks]
            digest = hashlib.sha1(raw).hexdigest()[:10]
            ids = [f"{digest}-{i}" for i in range(len(chunks))]

            need_vectors = bool(selected & {"vector", "hybrid"})
            embeddings: list[list[float]] | None = None
            if need_vectors:
                encoded = self.embedder.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings = encoded.tolist()
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

            need_bm25 = bool(selected & {"bm25", "hybrid"})
            if need_bm25:
                self._bm25_ids.extend(ids)
                self._bm25_texts.extend(texts)
                self._bm25_metadatas.extend(metadatas)
                tokenized = [_tokenize(t) for t in self._bm25_texts]
                self._bm25 = _BM25Okapi(tokenized) if tokenized else None

            for i, (chunk_id, text, meta) in enumerate(zip(ids, texts, metadatas)):
                vector = embeddings[i] if embeddings is not None else None
                self._preview.append(
                    {
                        "id": chunk_id,
                        "text": text,
                        "metadata": meta,
                        "embedding": vector,
                        "embedding_dim": len(vector) if vector is not None else 0,
                    }
                )

            return {
                "pages": len(documents),
                "chunks": len(chunks),
                "file_name": file_name,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    def get_preview(self) -> list[dict[str, Any]]:
        return list(self._preview)

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if self.chunk_count == 0:
            return []

        mode = self.retrieval_mode or "vector"
        if mode == "vector":
            return self._retrieve_vector(query, top_k)
        if mode == "bm25":
            return self._retrieve_bm25(query, top_k)
        return self._retrieve_hybrid(query, top_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[dict[str, Any]]:
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self.embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].tolist()

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for doc_id, doc, meta, distance in zip(ids, docs, metas, distances):
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": distance,
                    "score": 1.0 / (1.0 + float(distance)),
                }
            )
        return hits

    def _retrieve_bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._bm25 or not self._bm25_texts:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_k]

        hits: list[dict[str, Any]] = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0:
                continue
            hits.append(
                {
                    "id": self._bm25_ids[idx],
                    "text": self._bm25_texts[idx],
                    "metadata": self._bm25_metadatas[idx],
                    "distance": None,
                    "score": score,
                }
            )
        return hits

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector_hits = self._retrieve_vector(query, top_k=max(top_k * 3, top_k))
        bm25_hits = self._retrieve_bm25(query, top_k=max(top_k * 3, top_k))

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(vector_hits):
            doc_id = hit["id"]
            by_id[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["id"]
            by_id[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)

        ordered = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        hits: list[dict[str, Any]] = []
        for doc_id, score in ordered[:top_k]:
            hit = dict(by_id[doc_id])
            hit["score"] = score
            hits.append(hit)
        return hits

    def ask(self, question: str, role: str, top_k: int = 4) -> dict[str, Any]:
        rewritten = self.ollama.rewrite_query(question)
        hits = self.retrieve(rewritten, top_k=top_k)
        if not hits:
            return {
                "rewritten_query": rewritten,
                "answer": "No indexed PDF content found. Upload and index a PDF first.",
                "sources": [],
            }

        context = "\n\n---\n\n".join(
            f"[source: {hit['metadata'].get('source', 'unknown')} | "
            f"page: {hit['metadata'].get('page', '?')}]\n{hit['text']}"
            for hit in hits
        )
        answer = self.ollama.answer(
            role=role,
            question=question,
            context=context,
        )
        return {
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": hits,
        }


def _resolve_retrieval_mode(modes: set[IndexMode]) -> IndexMode:
    if "hybrid" in modes:
        return "hybrid"
    if modes >= {"vector", "bm25"}:
        return "hybrid"
    if "bm25" in modes:
        return "bm25"
    return "vector"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
