import json
from anthropic import Anthropic


class LLM:
    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def stream(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ):
        """Yield text deltas as the model generates. Streamed via Anthropic's SDK."""
        with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        ) as s:
            for delta in s.text_stream:
                yield delta

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        text = self.complete(system=system, user=user, max_tokens=max_tokens)
        payload = _extract_first_json_blob(text)

        # Attempt 1: strict json.loads
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass

        # Attempt 2: escape raw control chars inside string regions
        cleaned = _escape_raw_control_chars_in_strings(payload)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 3: tolerant JSON5 parser (handles unquoted keys, trailing commas,
        # single quotes, embedded unescaped quotes in some cases).
        try:
            import json5
            return json5.loads(cleaned)
        except Exception:
            pass

        # Attempt 4: as a last resort, normalize smart quotes and retry strict JSON
        final = _normalize_smart_quotes(cleaned)
        return json.loads(final)

    def complete_with_schema(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        tool_name: str = "submit_response",
        tool_description: str = "Submit the structured response.",
        max_tokens: int = 2048,
    ) -> dict:
        """Get a Python dict back from the model with schema-validated structure.

        Uses Anthropic's tool-use API. The model is forced to call the named tool
        whose input_schema is the supplied JSON Schema; Anthropic validates and
        returns the tool input dict directly. No fragile JSON parsing.
        """
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            tools=[{
                "name": tool_name,
                "description": tool_description,
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in resp.content:
            # SDK exposes tool_use blocks with .input as a dict
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                return dict(block.input)
        # Fallback: if the model didn't call the tool, raise a clear error
        raise RuntimeError(
            f"Model did not call expected tool '{tool_name}'. Response blocks: "
            f"{[getattr(b, 'type', None) for b in resp.content]}"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize_smart_quotes(s: str) -> str:
    """Replace common curly/smart quotes with straight ASCII quotes."""
    return (
        s
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _escape_raw_control_chars_in_strings(s: str) -> str:
    """Escape literal newlines/tabs/CRs that appear inside JSON string literals.

    Some LLMs return JSON like '{"a": "line one
line two"}' — the raw newline is invalid JSON
    but easy to recover by escaping. Walks chars tracking string state."""
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def _extract_first_json_blob(text: str) -> str:
    """Find the first balanced { ... } or [ ... ] in text. Tolerates leading prose,
    fenced code blocks, and trailing commentary. Returns the JSON substring (no parsing)."""
    if not text:
        return text
    # Strip common markdown fences
    stripped = text.strip()
    for fence_open in ("```json", "```JSON", "```"):
        if stripped.startswith(fence_open):
            stripped = stripped[len(fence_open):].lstrip()
            # Strip trailing fence if present
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
            break

    # Find the first { or [ and scan forward to its matching close
    start_idx = -1
    open_ch = ""
    close_ch = ""
    for i, ch in enumerate(stripped):
        if ch in "{[":
            start_idx = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start_idx == -1:
        return stripped   # No JSON found; let json.loads raise

    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(stripped)):
        ch = stripped[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return stripped[start_idx:i + 1]
    # If we ran off the end, return what we have; json.loads will surface the real error
    return stripped[start_idx:]
