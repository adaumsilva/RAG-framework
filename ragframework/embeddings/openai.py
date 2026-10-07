"""OpenAI-powered text embeddings."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from ragframework.base import Embedder
from ragframework.exceptions import EmbedderError

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 5


def _raise_import_error(exc: ImportError, module: str = "openai") -> NoReturn:
    raise ImportError(
        f"{module} support requires 'ragframework[{module}]'. "
        f"Install it with: pip install ragframework[{module}]"
    ) from exc


class OpenAIEmbedder(Embedder):
    """Create embeddings using the OpenAI Embeddings API.

    Clients are created lazily and reused across calls. They are released only
    by ``close()`` / ``aclose()`` or by leaving a ``with`` / ``async with``
    block. Use one instance within a single event loop.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        max_concurrency: int = 4,
        batch_size: int = 256,  # max strings per request
        max_batch_tokens: int = 250_000,  # max total tokens per request
        max_input_tokens: int = 8191,  # max tokens in a single string
        on_overflow: Literal["error", "truncate"] = "error",
        timeout: float | None = None,
        max_retries: int | None = None,
        organization: str | None = None,
    ) -> None:
        # ---- validate config first: fail early ----
        if on_overflow not in ("error", "truncate"):
            raise ValueError("on_overflow must be 'error' or 'truncate'")
        if max_concurrency < 1 or batch_size < 1:
            raise ValueError("max_concurrency and batch_size must be >= 1")
        if max_input_tokens < 1 or max_batch_tokens < 1:
            raise ValueError("max_input_tokens and max_batch_tokens must be >= 1")
        if max_input_tokens > max_batch_tokens:
            raise ValueError("max_input_tokens must be <= max_batch_tokens")
        if timeout is not None and (isinstance(timeout, bool) or timeout <= 0):
            raise ValueError("timeout must be a positive number of seconds")
        if max_retries is not None and (
            isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0
        ):
            raise ValueError("max_retries must be an integer >= 0")

        try:
            import openai
        except ImportError as exc:
            _raise_import_error(exc)

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EmbedderError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key.")

        self._openai = openai
        self._api_key = key  # private
        self.model = model
        self.api_base = api_base
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_input_tokens = max_input_tokens
        self.on_overflow = on_overflow
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else _DEFAULT_MAX_RETRIES
        self.organization = organization

        self._client: OpenAI | None = None
        self._aclient: AsyncOpenAI | None = None
        self._encoder: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle: clients live until close()/aclose()/context exit
    # ------------------------------------------------------------------
    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "api_key": self._api_key,
            "organization": self.organization,
            "base_url": self.api_base,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def _get_client(self) -> OpenAI:
        if self._client is None:
            try:
                self._client = self._openai.OpenAI(**self._client_kwargs())
            except Exception as exc:
                raise EmbedderError(f"Could not initialize OpenAI client: {exc}") from exc
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            try:
                self._aclient = self._openai.AsyncOpenAI(**self._client_kwargs())
            except Exception as exc:
                raise EmbedderError(f"Could not initialize OpenAI client: {exc}") from exc
        return self._aclient

    def close(self) -> None:
        """Release the synchronous OpenAI client. Safe to call repeatedly."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Release the asynchronous OpenAI client. Safe to call repeatedly."""
        if self._aclient is not None:
            await self._aclient.close()
            self._aclient = None

    def __enter__(self) -> OpenAIEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> OpenAIEmbedder:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                import tiktoken
            except ImportError as exc:
                _raise_import_error(exc, "tokens")

            try:
                try:
                    self._encoder = tiktoken.encoding_for_model(self.model)
                except KeyError:  # unknown model name (OpenRouter, vLLM, ...)
                    self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:
                raise EmbedderError(f"Could not initialize tokenizer: {exc}") from exc
        return self._encoder

    @staticmethod
    def _validate_inputs(texts: list[str]) -> None:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError(f"Expected a list of strings, got {type(texts).__name__}")

    def _truncate(self, enc: Any, toks: list[int]) -> tuple[str, int] | None:
        """Cut to max_input_tokens at a valid character boundary.

        Returns (text, actual_token_count), or None if nothing usable fits.
        """
        limit = self.max_input_tokens
        cut = limit
        while cut > 0:
            # a token cut can land mid-character; drop the partial trailing bytes
            text = enc.decode_bytes(toks[:cut]).decode("utf-8", errors="ignore")
            if not text.strip():
                return None  # shorter prefixes would be whitespace-only too
            n = len(enc.encode_ordinary(text))
            if n <= limit:
                return text, n
            cut -= 1  # re-tokenizing gave more tokens than the limit (rare)
        return None

    def _prepare_batches(self, texts: list[str]) -> list[list[str]]:
        """Split validated texts into request-sized batches, keeping order."""
        try:
            empty = [i for i, t in enumerate(texts) if not t.strip()]
            if empty:
                raise EmbedderError(f"Empty/whitespace-only texts at indices {empty[:10]}")

            enc = self._get_encoder()
            # encode_ordinary: treats special-token text like "<|endoftext|>" as plain text
            tokens_list = enc.encode_ordinary_batch(texts)

            too_long = [i for i, t in enumerate(tokens_list) if len(t) > self.max_input_tokens]

            if too_long and self.on_overflow == "error":
                raise EmbedderError(
                    f"Texts at indices {too_long[:10]} exceed {self.max_input_tokens} tokens. "
                    "Chunk them smaller or set on_overflow='truncate'."
                )

            items: list[tuple[str, int]] = []
            unusable: list[int] = []

            for i, (text, toks) in enumerate(zip(texts, tokens_list, strict=True)):
                if len(toks) > self.max_input_tokens:  # only reachable when truncating
                    result = self._truncate(enc, toks)
                    if result is None:
                        unusable.append(i)
                        continue
                    text, n = result
                else:
                    n = len(toks)
                items.append((text, n))

            if unusable:
                raise EmbedderError(
                    f"Texts at indices {unusable[:10]} have no usable text within "
                    f"{self.max_input_tokens} tokens after truncation."
                )

            if too_long:
                logger.warning(
                    "Truncating %d text(s) to %d tokens", len(too_long), self.max_input_tokens
                )

            batches: list[list[str]] = []
            curr_batch: list[str] = []
            curr_tokens = 0

            for text, n in items:
                # close the batch only if adding this text would break a limit
                if curr_batch and (
                    len(curr_batch) >= self.batch_size or curr_tokens + n > self.max_batch_tokens
                ):
                    batches.append(curr_batch)
                    curr_batch, curr_tokens = [], 0

                curr_batch.append(text)
                curr_tokens += n

            if curr_batch:
                batches.append(curr_batch)
            return batches
        except (EmbedderError, ImportError):
            raise
        except Exception as exc:
            raise EmbedderError(f"Could not prepare embedding batches: {exc}") from exc

    def _params(self, batch: list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "input": batch}
        return params

    @staticmethod
    def _ordered(response: Any) -> list[list[float]]:
        # never assume response order matches input order
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]

    @staticmethod
    def _check_batch(out: list[list[float]], batch: list[str]) -> list[list[float]]:
        if len(out) != len(batch):
            raise EmbedderError(f"Expected {len(batch)} embeddings for batch, got {len(out)}")
        return out

    def _wrap(self, exc: Exception) -> EmbedderError:
        o = self._openai
        if isinstance(exc, o.AuthenticationError):
            msg = "Authentication failed (check API key)"
        elif isinstance(exc, o.RateLimitError):
            msg = "Rate limited or quota exhausted"
        elif isinstance(exc, o.BadRequestError):
            msg = "Invalid request (input too long or bad params?)"
        else:
            msg = "Could not create embeddings"
        return EmbedderError(f"{msg}: {exc}")

    @staticmethod
    def _check(out: list[list[float]], texts: list[str]) -> list[list[float]]:
        if len(out) != len(texts):
            raise EmbedderError(f"Expected {len(texts)} embeddings, got {len(out)}")
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text, in input order."""
        self._validate_inputs(texts)
        if not texts:
            return []

        batches = self._prepare_batches(texts)  # validate before creating a client

        out: list[list[float]] = []
        try:
            client = self._get_client()
            for batch in batches:
                batch_out = self._ordered(client.embeddings.create(**self._params(batch)))
                out.extend(self._check_batch(batch_out, batch))
        except self._openai.OpenAIError as exc:
            raise self._wrap(exc) from exc
        except EmbedderError:
            raise
        except Exception as exc:
            raise EmbedderError(f"Could not create embeddings: {exc}") from exc
        return self._check(out, texts)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings asynchronously, preserving input order."""
        self._validate_inputs(texts)
        if not texts:
            return []

        # tokenizing is CPU-bound: keep it off the event loop
        batches = await asyncio.to_thread(self._prepare_batches, texts)
        client = self._get_aclient()

        sem = asyncio.Semaphore(self.max_concurrency)  # per call: loop-safe

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with sem:
                response = await client.embeddings.create(**self._params(batch))
            return self._check_batch(self._ordered(response), batch)

        tasks = [asyncio.create_task(embed_batch(b)) for b in batches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException as exc:
            # stop sibling batches from running (and spending tokens) in the background
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(exc, self._openai.OpenAIError):
                raise self._wrap(exc) from exc
            if isinstance(exc, EmbedderError):
                raise
            if isinstance(exc, Exception):
                raise EmbedderError(f"Could not create embeddings: {exc}") from exc
            raise
        return self._check([v for r in results for v in r], texts)


__all__ = ["OpenAIEmbedder"]
