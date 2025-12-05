import base64, os, time, random
from openai import OpenAI
from pydantic import ValidationError
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal


class ScreenshotSummary(BaseModel):
    topic: str = Field(..., description="Short phrase for what work/study/task the screenshot reflects")
    app_or_website: str = Field(..., description="App name or site title")
    url: Optional[str] = Field(None, description="Visible URL if present")
    work_type: Literal[
        "reading","note_taking","coding","messaging","browsing",
        "entertainment","design","spreadsheets","presentation","unknown"
    ] = Field(..., description="Type of work inferred from the screenshot")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list, description="Snippets or UI labels that justify the fields")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#///////////// HELPERS //////////

def _image_b64_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"

#////////////////////////////////


def analyze_screenshot_with_openai(path: str) -> ScreenshotSummary:
    """
    Calls GPT-4o-mini with structured output enforced by JSON schema.
    Returns a validated ScreenshotSummary (raises ValidationError on mismatch).
    """
    data_url = _image_b64_data_url(path)

    # Build a JSON schema the API will enforce
    json_schema = ScreenshotSummary.model_json_schema()
    json_schema["additionalProperties"] = False
    props = json_schema.get("properties", {})
    json_schema["required"] = list(props.keys())
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                  # or "gpt-4o" for higher quality
        temperature=0,                        # more deterministic
        seed=42,                              # repeatability (best-effort)
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ScreenshotSummary",
                "schema": json_schema,
                "strict": True
            }
        },
        messages=[
            {"role": "system", "content": (
                "You analyze desktop screenshots and return ONLY JSON that matches the schema. "
                "Never add extra keys or prose."
            )},
            {"role": "user", "content": [
                {"type": "text", "text":
                 "Extract fields according to the schema. Rules:\n"
                 "1) topic: what is this page about?\n"
                 "2) app_or_website: identify the application or website. If a browser, infer active site/tab (favicon/title). "
                 "3) url: if a browser URL bar is visible, read it; else infer from branding (or null).\n"
                 "4) Choose ONE work_type from exactly: "
                 '   ["reading","note_taking","coding","messaging","browsing","entertainment","design","spreadsheets","presentation","unknown"]\n'
                 "5) confidence: 0–1 based on how well you satisfied 1–4.\n"
                 "6) evidence: quote small visible on-screen text only (no guesses)."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
    )

    raw = resp.choices[0].message.content
    # The API returns a JSON string (already schema-constrained). Validate with Pydantic:
    return ScreenshotSummary.model_validate_json(raw)
