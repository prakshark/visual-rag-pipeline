"""Thin Ollama connection helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import ollama
import requests


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "llama3.2:1b"

# Free ngrok serves an interstitial unless this header is present.
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


@dataclass(frozen=True)
class OllamaStatus:
    connected: bool
    host: str
    port: int
    base_url: str
    model: str
    model_ready: bool
    message: str


def _normalize_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        if base_url:
            self.base_url = _normalize_base_url(base_url)
            parsed = urlparse(self.base_url)
            self.host = parsed.hostname or DEFAULT_HOST
            default_port = 443 if parsed.scheme == "https" else 80
            self.port = parsed.port or default_port
        else:
            self.host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
            self.port = int(port)
            self.base_url = f"http://{self.host}:{self.port}"
        self._client = ollama.Client(host=self.base_url, headers=dict(_NGROK_HEADERS))

    @classmethod
    def from_base_url(cls, base_url: str, model: str = DEFAULT_MODEL) -> "OllamaClient":
        return cls(model=model, base_url=base_url)

    @classmethod
    def from_env_or_defaults(cls, model: str = DEFAULT_MODEL) -> "OllamaClient":
        """Prefer OLLAMA_BASE_URL, else OLLAMA_HOST + OLLAMA_PORT."""
        base = os.environ.get("OLLAMA_BASE_URL", "").strip()
        if base:
            return cls.from_base_url(base, model=model)
        host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        port = int(os.environ.get("OLLAMA_PORT", str(DEFAULT_PORT)))
        return cls(host=host, port=port, model=model)

    def check(self) -> OllamaStatus:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=8,
                headers=dict(_NGROK_HEADERS),
            )
            response.raise_for_status()
            models = {item.get("name", "") for item in response.json().get("models", [])}
            model_ready = self.model in models or any(
                name.startswith(f"{self.model}:") or name.startswith(self.model)
                for name in models
            )
            if not model_ready:
                return OllamaStatus(
                    connected=True,
                    host=self.host,
                    port=self.port,
                    base_url=self.base_url,
                    model=self.model,
                    model_ready=False,
                    message=f"Ollama is up, but '{self.model}' is not pulled yet.",
                )
            return OllamaStatus(
                connected=True,
                host=self.host,
                port=self.port,
                base_url=self.base_url,
                model=self.model,
                model_ready=True,
                message=f"Connected to {self.model}",
            )
        except requests.RequestException as exc:
            return OllamaStatus(
                connected=False,
                host=self.host,
                port=self.port,
                base_url=self.base_url,
                model=self.model,
                model_ready=False,
                message=f"Cannot reach Ollama at {self.base_url} ({exc.__class__.__name__})",
            )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        result = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
        )
        return (result.get("message") or {}).get("content", "").strip()

    def rewrite_query(self, question: str) -> str:
        prompt = (
            "Rewrite the user question into a clear, searchable retrieval query. "
            "Keep the same intent. Return only the rewritten query.\n\n"
            f"Question: {question}"
        )
        rewritten = self.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite questions for document search. Be concise.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return rewritten or question

    def answer(
        self,
        *,
        role: str,
        question: str,
        context: str,
    ) -> str:
        system_prompt = (
            role.strip()
            or "You are a careful assistant that answers only from the provided context."
        )
        user_prompt = (
            "Use the context below to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
