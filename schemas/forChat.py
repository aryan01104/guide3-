import base64, os, time, random
from openai import OpenAI
from pydantic import ValidationError
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal


from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal

class ScreenshotSummary(BaseModel):
    # 1–2 sentence task-level description
    semantic_summary: str = Field(
        ...,
        description=(
            "1–2 sentence description of what the user is trying to accomplish in this screenshot, "
            "including the concrete content (course/project/client) and how it fits into a broader task."
        )
    )

    workstream_label: str = Field(
    ...,
    description="Life-scale workstream or domain this belongs to (course, product, client, long-term study area)."
    )

    deliverable_label: str = Field(
        ...,
        description="Specific exam/assignment/report/feature/pitch the user is currently working toward."
    )

    app_or_website: str

    # Raw app/site name (titlebar/tab/app label)
    app_or_website: str = Field(
        ...,
        description="Application name or website title shown (e.g. 'Google Chrome – Canvas', 'VS Code', 'PowerPoint')."
    )

    # Normalized app bucket
    app_bucket: Literal[
        "browser",
        "ide",
        "pdf_viewer",
        "notes",
        "email",
        "terminal",
        "file_explorer",
        "messaging",
        "media_player",
        "other",
    ] = Field(
        ...,
        description=(
            "Coarse category of the active application: one of "
            "['browser','ide','pdf_viewer','notes','email','terminal','file_explorer','messaging','media_player','other']."
        )
    )

    # Micro work type (what kind of interaction)
    work_type: Literal[
        "reading",
        "note_taking",
        "coding",
        "messaging",
        "browsing",
        "entertainment",
        "design",
        "spreadsheets",
        "presentation",
        "unknown",
    ] = Field(
        ...,
        description="Type of work inferred from the screenshot."
    )

    # Telic vs atelic
    goal_type: Literal["telic", "atelic", "unknown"] = Field(
        ...,
        description=(
            "Telic if the activity is clearly directed at a concrete outcome/deliverable "
            "(assignment, pitch, feature, bugfix, report); "
            "atelic if it is open-ended exploration/consumption without a clear endpoint; "
            "unknown if not clear."
        )
    )


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
            {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text":
                "Extract fields according to the schema. Rules:\n"
                "\n"
                "1) semantic_summary: in 1–2 sentences, describe what the user is trying to accomplish in this screenshot, "
                "what concrete content they are working with (course, project, client, repo, document), and how this fits into "
                "a broader task (e.g. preparing an assignment, writing a speech, editing a pitch deck, implementing a feature).\n"
                "   - Focus on TASK-LEVEL meaning, not UI details.\n"
                "\n"
                "2) workstream_label: assign the life-scale workstream or domain this activity belongs to. This should be stable "
                "over weeks or months, not change every session.\n"
                "   - Examples: 'BIOG 1500 course', 'Global Development course', 'AI Mirror product', "
                "     'Internship applications', 'Personal reading: Buddhism'.\n"
                "   - If multiple screenshots on different days would belong to the same semester-long course or long-term project, "
                "     they should share the same workstream_label.\n"
                "\n"
                "3) deliverable_label: assign the specific deliverable or objective the user is currently working toward within the workstream.\n"
                "   - Examples: 'Study for BIOG Exam 2', 'Write Global Development midterm speech', "
                "     'Implement episodization coherence function', 'Prepare pitch deck v3', 'Finish lab report section 3'.\n"
                "   - This is a bounded objective that could be completed over one or several sessions.\n"
                "\n"
                "4) app_or_website: name the active application or website as shown in the window/tab/frame title "
                "(e.g. 'Google Chrome – Canvas', 'VS Code', 'Notion', 'YouTube').\n"
                "\n"
                "5) app_bucket: choose ONE from exactly:\n"
                "   ['browser','ide','pdf_viewer','notes','email','terminal','file_explorer','messaging','media_player','other'].\n"
                "   - 'browser' for Chrome/Firefox/Safari showing web content.\n"
                "   - 'ide' for code editors like VS Code, PyCharm, etc.\n"
                "   - 'pdf_viewer' for apps showing PDF documents.\n"
                "   - 'notes' for note-taking tools (Notion, Obsidian, Apple Notes, etc.).\n"
                "   - 'email' for email clients (Gmail web, Outlook, etc.).\n"
                "   - 'terminal' for command-line shells.\n"
                "   - 'file_explorer' for file managers.\n"
                "   - 'messaging' for chat apps.\n"
                "   - 'media_player' for dedicated media players.\n"
                "   - 'other' if none of the above fit.\n"
                "\n"
                "6) work_type: choose ONE from exactly:\n"
                "   [\"reading\",\"note_taking\",\"coding\",\"messaging\",\"browsing\",\"entertainment\",\"design\",\"spreadsheets\",\"presentation\",\"unknown\"].\n"
                "   - 'reading': mainly consuming text (articles, PDFs, textbooks, docs) without editing.\n"
                "   - 'note_taking': writing structured notes, outlines, or annotations.\n"
                "   - 'coding': working in code editors, terminals, or IDE tools to write/modify code.\n"
                "   - 'messaging': chat/email/slack/DM focused.\n"
                "   - 'browsing': general web surfing, search results, multiple unrelated tabs.\n"
                "   - 'entertainment': videos, games, social feeds primarily for leisure.\n"
                "   - 'design': Figma, slide design, visual layout work.\n"
                "   - 'spreadsheets': Excel/Sheets or similar grid-based work.\n"
                "   - 'presentation': editing slide decks (PowerPoint, Keynote, Google Slides).\n"
                "   - 'unknown': if unclear.\n"
                "\n"
                "7) goal_type: classify whether the activity is telic, atelic, or unknown.\n"
                "   - 'telic': clearly directed at a concrete outcome/deliverable (finish an assignment, complete a pitch deck, "
                "     implement a feature, fix a bug, submit a report, study for a specific exam).\n"
                "   - 'atelic': open-ended exploration or consumption (browsing, reading for curiosity, watching random videos) "
                "     without a clear endpoint.\n"
                "   - 'unknown': if you cannot reliably tell.\n"
                "\n"
        }
        ,
        {
            "type": "image_url",
            "image_url": {"url": data_url}
        }
    ]
}


        ],
    )

    raw = resp.choices[0].message.content
    # The API returns a JSON string (already schema-constrained). Validate with Pydantic:
    return ScreenshotSummary.model_validate_json(raw)
