# label_canonicalizer.py

from typing import Tuple
from supabase_client import supabase

def normalize_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    for ch in [":", "-", "–", "_", "|"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def _similarity(a: str, b: str) -> float:
    """
    Very simple token-overlap similarity between two normalized strings.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = max(len(ta), len(tb))
    return inter / denom


def canonicalize_workstream(raw_label: str) -> Tuple[int, str]:
    """
    Given a raw workstream_label from GPT, either match to an existing workstream
    (by canonical_label) or create a new one.

    Returns: (workstream_id, canonical_label)
    """
    raw_label = raw_label or "unknown workstream"
    norm = normalize_label(raw_label)

    # fetch existing workstreams
    resp = supabase.table("workstreams").select("id, canonical_label").execute()
    rows = resp.data or []

    best_id = None
    best_label = None
    best_score = 0.0

    for r in rows:
        existing = r["canonical_label"]
        norm_existing = normalize_label(existing)
        score = _similarity(norm, norm_existing)
        if score > best_score:
            best_score = score
            best_id = r["id"]
            best_label = existing

    # threshold: if it's similar enough, reuse it
    if best_id is not None and best_score >= 0.7:
        return best_id, best_label

    # otherwise create a new workstream
    ins = supabase.table("workstreams").insert(
        {"canonical_label": raw_label}
    ).execute()
    new_id = ins.data[0]["id"]
    return new_id, raw_label


def canonicalize_deliverable(workstream_id: int, raw_label: str) -> Tuple[int, str]:
    """
    Given a raw deliverable_label and a workstream_id, either match to an existing
    deliverable under that workstream or create a new one.

    Returns: (deliverable_id, canonical_label)
    """
    raw_label = raw_label or "unspecified deliverable"
    norm = normalize_label(raw_label)

    resp = supabase.table("deliverables").select("id, canonical_label").eq(
        "workstream_id", workstream_id
    ).execute()
    rows = resp.data or []

    best_id = None
    best_label = None
    best_score = 0.0

    for r in rows:
        existing = r["canonical_label"]
        norm_existing = normalize_label(existing)
        score = _similarity(norm, norm_existing)
        if score > best_score:
            best_score = score
            best_id = r["id"]
            best_label = existing

    if best_id is not None and best_score >= 0.7:
        return best_id, best_label

    ins = supabase.table("deliverables").insert(
        {"workstream_id": workstream_id, "canonical_label": raw_label}
    ).execute()
    new_id = ins.data[0]["id"]
    return new_id, raw_label
