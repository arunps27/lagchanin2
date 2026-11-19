class LLMError(RuntimeError):
    pass


class LLM:
    """Thin wrapper over the Groq chat completions API (default: openai/gpt-oss-120b)."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env.")
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        reasoning_effort: str = "low",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # network / auth / rate limit
            raise LLMError(f"LLM request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()
