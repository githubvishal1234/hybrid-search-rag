"""One function, four providers. OpenAI, Gemini and Ollama all speak the OpenAI chat API, so they
share the openai SDK with a different base_url; Anthropic uses its own SDK."""

from functools import cache

import anthropic
import openai

from hybrid_rag.config import settings

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@cache
def _openai_compatible() -> openai.OpenAI:
    if settings.llm_provider == "ollama":
        return openai.OpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
    if settings.llm_provider == "gemini":
        return openai.OpenAI(base_url=GEMINI_OPENAI_URL, api_key=settings.gemini_api_key)
    return openai.OpenAI(api_key=settings.openai_api_key)


def complete(system: str, user: str) -> str:
    if settings.llm_provider == "anthropic":
        resp = anthropic.Anthropic(api_key=settings.anthropic_api_key).messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    resp = _openai_compatible().chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""
