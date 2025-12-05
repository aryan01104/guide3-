from pydantic import BaseModel, Field
from typing import Literal, List
from morphik import Morphik
from pathlib import Path
from uuid import uuid4

Morphik("morphik://guide3:eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMTgxYzZkOTEtMDAwNS00Y2E5LTkwN2MtMjg1ZTYwYjYxNGVkIiwiYXBwX2lkIjoidnYza2NxdzU3cGsiLCJuYW1lIjoiZ3VpZGUzIiwicGVybWlzc2lvbnMiOlsicmVhZCIsIndyaXRlIl0sImV4cCI6MTc5Mjg2MDQzNiwidHlwZSI6ImRldmVsb3BlciIsImVudGl0eV9pZCI6IjE4MWM2ZDkxLTAwMDUtNGNhOS05MDdjLTI4NWU2MGI2MTRlZCJ9.zjHuX3sBhjqRRaIjTLYq29TohVDpywxthxotCWs12Wg@api.morphik.ai")

class ScreenshotSummary(BaseModel):
    topic: str = Field(..., description="Short phrase for what work/study/task the screenshot reflects")
    app_or_website: str = Field(..., description="App name or site title")
    url: str = Field(None, description="Visible URL if present")
    work_type: Literal[
        'reading',"note_taking","coding","messaging","browsing",
        "entertainment","design","spreadsheets","presentation","unknown"
    ] = Field(..., description="Type of work inferred from the screenshot")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Snippets or UI labels that justify the fields")

def summarize_screenshot(doc_uid):
    prompt = (
    "You are analyzing a desktop screenshot. "
    "Extract fields according to the provided schema. "
    "Rules:\n"
    "1) topic: analyze the text on the page. what is the subject of the page?\n"
    "2) app_or_website:\n"
    " - analyze the image visually, does this match an application or a webpage you have seen?\n"
    " - if it is a browser, what is the active tab? look at the favicon, see if you recognize it\n" 
    " - if it is not a browser, you may see the app name in the top or bottom 10th horizontal slice of the picture\n"
    "3) If it is a browser, there is likely a search bar: what is the URL in it? what website is it?\n"
    " - if the search bar is not visible, the webpage usually has the website name in the upper part of the page\n"
    "4) Choose ONE work_type from EXACTLY this list (copy the token verbatim):\n"
    "  - [\"reading\",\"note_taking\",\"coding\",\"messaging\",\"browsing\",\"entertainment\",\"design\",\"spreadsheets\",\"presentation\",\"unknown\"]\n"
    "  - If unsure, use 'unknown'.\n"
    "5) confidence reflects overall certainty (0–1).\n"
    " - how sure are you of your reasoning in steps 1–4?\n"
    "6) evidence should quote small on-screen texts or labels.\n"
    " - you can only cite *TEXT DETECTED FROM IMAGE*\n"
    "Return ONLY the structured object — no extra prose.\n"
    )

    return morphik.query(
        query=prompt,
        schema=ScreenshotSummary,     # <-- Structured output
        
        # Scope retrieval strictly to one file:
        filters={"doc_uid": doc_uid}
    )



morphik = Morphik()

def ingest_screenshot(path: str):
    p = Path(path)
    doc_uid = f"{p.stem}-{uuid4().hex[:8]}"  # unique handle for this image
    doc = morphik.ingest_file(
        file=path,
        filename=p.name,               # top-level filename
        metadata={
            "filename": p.stem,        # ALSO copy into metadata for filtering
            "doc_uid": doc_uid,        # strong, unique key
            "source": "screenshot",
        },
        use_colpali=True,
    )
    doc.wait_for_completion()          # <- critical
    return doc.id, doc_uid



    
