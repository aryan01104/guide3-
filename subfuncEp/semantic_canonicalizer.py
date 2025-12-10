# subfuncEp/semantic_canonicalizer.py (embedding-based)
from typing import Tuple, List
from supabase_client import supabase
from embeddings import get_embedding, cosine_similarity

WORKSTREAM_SIM_THRESHOLD = 0.80
DELIVERABLE_SIM_THRESHOLD = 0.85

def _update_centroid(old_emb: List[float], old_n: int, new_emb: List[float]) -> List[float]:
    if not old_emb or old_n <= 0:
        return new_emb
    if len(old_emb) != len(new_emb):
        return new_emb
    n = float(old_n)
    return [(old_emb[i] * n + new_emb[i]) / (n + 1.0) for i in range(len(new_emb))]

def canonicalize_workstream(raw_label: str, semantic_summary: str) -> Tuple[int, str]:
    raw_label = raw_label or "unknown workstream"
    text_for_embed = f"{semantic_summary} | workstream: {raw_label}"
    emb = get_embedding(text_for_embed)

    resp = supabase.table("workstreams").select(
        "id, canonical_label, embedding, n_points"
    ).execute()
    rows = resp.data or []

    best_id = None
    best_label = None
    best_score = 0.0
    best_emb = None
    best_n = 0

    for r in rows:
        existing_emb = r.get("embedding") or []
        if not existing_emb:
            continue
        score = cosine_similarity(emb, existing_emb)
        if score > best_score:
            best_score = score
            best_id = r["id"]
            best_label = r["canonical_label"]
            best_emb = existing_emb
            best_n = r.get("n_points", 0) or 0

    if best_id is not None and best_score >= WORKSTREAM_SIM_THRESHOLD:
        new_centroid = _update_centroid(best_emb, best_n, emb)
        try:
            supabase.table("workstreams").update(
                {"embedding": new_centroid, "n_points": best_n + 1}
            ).eq("id", best_id).execute()
        except Exception as e:
            print(f"(WS.e) Failed to update workstream centroid: {e}")
        return best_id, best_label

    try:
        ins = supabase.table("workstreams").insert(
            {"canonical_label": raw_label, "embedding": emb, "n_points": 1}
        ).execute()
        new_id = ins.data[0]["id"]
        return new_id, raw_label
    except Exception as e:
        print(f"(WS.e) Failed to insert new workstream: {e}")
        return -1, raw_label

def canonicalize_deliverable(
    workstream_id: int,
    workstream_label: str,
    raw_label: str,
    semantic_summary: str,
) -> Tuple[int, str]:
    raw_label = raw_label or "unspecified deliverable"
    text_for_embed = (
        f"{semantic_summary} | workstream: {workstream_label} | deliverable: {raw_label}"
    )
    emb = get_embedding(text_for_embed)

    resp = (
        supabase.table("deliverables")
        .select("id, canonical_label, embedding, n_points")
        .eq("workstream_id", workstream_id)
        .execute()
    )
    rows = resp.data or []

    best_id = None
    best_label = None
    best_score = 0.0
    best_emb = None
    best_n = 0

    for r in rows:
        existing_emb = r.get("embedding") or []
        if not existing_emb:
            continue
        score = cosine_similarity(emb, existing_emb)
        if score > best_score:
            best_score = score
            best_id = r["id"]
            best_label = r["canonical_label"]
            best_emb = existing_emb
            best_n = r.get("n_points", 0) or 0

    if best_id is not None and best_score >= DELIVERABLE_SIM_THRESHOLD:
        new_centroid = _update_centroid(best_emb, best_n, emb)
        try:
            supabase.table("deliverables").update(
                {"embedding": new_centroid, "n_points": best_n + 1}
            ).eq("id", best_id).execute()
        except Exception as e:
            print(f"(DV.e) Failed to update deliverable centroid: {e}")
        return best_id, best_label

    try:
        ins = supabase.table("deliverables").insert(
            {
                "workstream_id": workstream_id,
                "canonical_label": raw_label,
                "embedding": emb,
                "n_points": 1,
            }
        ).execute()
        new_id = ins.data[0]["id"]
        return new_id, raw_label
    except Exception as e:
        print(f"(DV.e) Failed to insert deliverable: {e}")
        return -1, raw_label
