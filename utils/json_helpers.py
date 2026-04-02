import json
import re

def parse_llm_json(raw: str) -> dict:
    """
    Robustly parse JSON from LLM output.
    Handles markdown fences, trailing text, etc.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_start = cleaned.find('{')
    if brace_start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start:i+1])
                except json.JSONDecodeError:
                    raise ValueError("Found JSON boundaries but failed to parse")

    raise ValueError("Unbalanced braces in LLM output")
