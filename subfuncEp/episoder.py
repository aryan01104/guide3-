# episoder.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import os
from openai import OpenAI


from supabase_client import supabase

# ---------- Config ----------------------------------------------------------

SAME_EPISODE_THRESHOLD = 0.7   # coherence score >= this → same episode
MIN_SWITCH_SCREENS     = 3     # need this many consecutive low-coherence screens to declare a new episode
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------- Helpers ---------------------------------------------------------

def _parse_timestamp(ts: str) -> datetime:
    """
    Parse your screenshot timestamp string: 'YYYY-MM-DD_HH-MM-SS'
    """
    return datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")

def _mode(xs: List[str]) -> str:
    xs = [x for x in xs if x and x.strip()]
    if not xs:
        return "unknown"
    counts = {}
    for x in xs:
        x = x.strip()
        counts[x] = counts.get(x, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _build_episode_descriptor(ep: EpisodeState, max_examples: int = 5) -> Dict[str, Any]:
    """
    Build a compact, textual descriptor of the ongoing episode to show GPT.
    This is the 'archetype' representation (whereas a kind of screenshot is a micro-archetype).
    """

    rows = ep.screenshot_rows
    if not rows:
        return {}

    # dominant categorical fields
    workstream_labels = [(r.get("workstream_label") or "unknown") for r in rows]
    deliverable_labels = [(r.get("deliverable_label") or "unknown") for r in rows]
    app_buckets    = [(r.get("app_bucket")    or "other")   for r in rows]
    work_types     = [(r.get("work_type")     or "unknown") for r in rows]
    goal_types     = [(r.get("goal_type")     or "unknown") for r in rows]

    dominant_workstream = _mode(workstream_labels)
    dominant_deliverable = _mode(deliverable_labels)
    dominant_app     = _mode(app_buckets)
    dominant_work    = _mode(work_types)
    dominant_goal    = _mode(goal_types)

    # example semantic summaries (most recent few)
    examples = []
    for r in rows[-max_examples:]:
        ss = r.get("semantic_summary") or r.get("topic") or ""
        if ss:
            examples.append(ss)

    return {
        "time_span": {
            "start": ep.start_time.isoformat(),
            "end": ep.end_time.isoformat(),
        },
        "screenshot_count": len(rows),
        "dominant_workstream_label": dominant_workstream,
        "dominant_deliverable_label": dominant_deliverable,
        "dominant_app_bucket": dominant_app,
        "dominant_work_type": dominant_work,
        "dominant_goal_type": dominant_goal,
        "example_summaries": examples,
    }


def _build_screenshot_descriptor(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact description of a single screenshot for GPT.
    """
    return {
        "timestamp": row.get("timestamp"),
        "app_or_website": row.get("app_or_website"),
        "app_bucket": row.get("app_bucket"),
        "url": row.get("url"),
        "topic": row.get("topic"),
        "semantic_summary": row.get("semantic_summary"),
        "workstream_label": row.get("workstream_label"),
        "deliverable_label": row.get("deliverable_label"),
        "work_type": row.get("work_type"),
        "goal_type": row.get("goal_type"),
    }


def _log_coherence_label(
    ep: EpisodeState,
    shot_row: Dict[str, Any],
    score: float,
    episode_descriptor: Dict[str, Any],
    screenshot_descriptor: Dict[str, Any],
) -> None:
    """
    Store the coherence judgment in 'coherence_labels' for training later.
    """
    row = {
        "screenshot_timestamp": shot_row.get("timestamp"),
        "episode_start_time": ep.start_time.isoformat(),
        "episode_end_time": ep.end_time.isoformat(),
        "coherence_score": float(score),
        "label_source": "gpt_v1",
        "episode_descriptor": episode_descriptor,
        "screenshot_descriptor": screenshot_descriptor,
    }
    try:
        resp = supabase.table("coherence_labels").insert(row).execute()
        print(f"(COH.✓) Logged coherence label")
    except Exception as e:
        print(f"(COH.e) Failed to insert coherence label: {e}")



@dataclass
class EpisodeState:
    """
    In-memory representation of an open episode.
    """
    start_time: datetime
    end_time: datetime
    screenshot_ids: List[Any] = field(default_factory=list)
    screenshot_rows: List[Dict[str, Any]] = field(default_factory=list)

    # optional aggregates
    workstream_labels: List[str] = field(default_factory=list)
    deliverable_labels: List[str] = field(default_factory=list)
    goal_types: List[str] = field(default_factory=list)
    work_types: List[str] = field(default_factory=list)
    apps: List[str] = field(default_factory=list)

    def add_screenshot(self, row: Dict[str, Any]) -> None:
        ts = _parse_timestamp(row["timestamp"])
        self.end_time = ts
        self.screenshot_rows.append(row)

        # primary key name may differ; adjust if needed (e.g. "id")
        if "id" in row:
            self.screenshot_ids.append(row["id"])

        self.workstream_labels.append((row.get("workstream_label") or "unknown").strip())
        self.deliverable_labels.append((row.get("deliverable_label") or "unknown").strip())
        self.goal_types.append((row.get("goal_type") or "unknown").strip())
        self.work_types.append((row.get("work_type") or "unknown").strip())
        self.apps.append((row.get("app_or_website") or "unknown").strip())

    def to_db_row_format(self) -> Dict[str, Any]:
        """
        Convert this in-memory episode into a single row for the 'episodes' table.
        You can later refine how you aggregate categorical fields.
        """
        def _mode(xs: List[str]) -> str:
            if not xs:
                return "unknown"
            counts = {}
            for x in xs:
                counts[x] = counts.get(x, 0) + 1
            # return most frequent
            return max(counts.items(), key=lambda kv: kv[1])[0]

        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "screenshot_count": len(self.screenshot_rows),
            "workstream_label": _mode(self.workstream_labels),
            "deliverable_label": _mode(self.deliverable_labels),
            "goal_type": _mode(self.goal_types),
            "work_band": _mode(self.work_types),
            "app_or_website": _mode(self.apps),
        }



# ---------- Global episodization state --------------------------------------

current_episode: Optional[EpisodeState] = None
pending_buffer: List[Dict[str, Any]] = []   # low-coherence screenshots waiting to see if they form a new episode


# ---------- Coherence function placeholder ----------------------------------

def coherence_with_episode(episode: EpisodeState, shot_row: Dict[str, Any]) -> float:
    """
    GPT-based teacher version.

    1. Build an 'episode descriptor' (archetype) from EpisodeState.
    2. Build a screenshot descriptor from the new screenshot.
    3. Ask GPT to judge coherence in [0,1].
    4. Log the pair + score to 'coherence_labels' for future training.
    """
    episode_desc = _build_episode_descriptor(episode)
    shot_desc = _build_screenshot_descriptor(shot_row)

    # JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "coherence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Coherence score between 0 and 1.",
            }
        },
        "required": ["coherence"],
        "additionalProperties": False,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            seed=42,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CoherenceJudgment",
                    "schema": json_schema,
                    "strict": True,
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You judge whether a new computer activity belongs to the same work session "
                        "as an ongoing episode.\n"
                        "A 'work session' means a sustained attempt to make progress on the same task or deliverable "
                        "(e.g. one assignment, one pitch deck, one feature, one bugfix, one report).\n"
                        "Return a coherence score in [0,1], where 1 means clearly the same work session/task, "
                        "and 0 means clearly unrelated. Intermediate values (e.g. 0.3, 0.7) reflect uncertainty.\n"
                        "Be conservative: if the new activity obviously pursues a different task/deliverable, "
                        "use a low score (< 0.3). If it clearly supports the same task (even via a different app), "
                        "use a high score (> 0.7)."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Ongoing episode (summary as JSON):\n"
                                f"{json.dumps(episode_desc, ensure_ascii=False, indent=2)}\n\n"
                                "New activity (screenshot summary as JSON):\n"
                                f"{json.dumps(shot_desc, ensure_ascii=False, indent=2)}\n\n"
                                "Question: On a scale from 0 to 1, how coherent is the new activity with the ongoing episode, "
                                "interpreting 'coherent' as 'part of the same work session or task'?\n"
                                "Return ONLY JSON of the form: {\"coherence\": number_between_0_and_1}."
                            ),
                        }
                    ],
                },
            ],
        )

        raw = resp.choices[0].message.content
        data = json.loads(raw)
        score = float(data["coherence"])

        # log for training
        _log_coherence_label(episode, shot_row, score, episode_desc, shot_desc)

        print(f"(COH) coherence_with_episode => {score:.3f}")
        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"(COH.e) GPT coherence judgment failed: {e}")
        # Fallback: treat as same episode to avoid oversplitting on errors
        return 1.0



# ---------- Core logic ------------------------------------------------------

def _flush_episode_to_db(ep: EpisodeState) -> None:
    """
    Insert a finished episode into the 'episodes' table.
    """
    row = ep.to_db_row_format()
    try:
        resp = supabase.table("episodes").insert(row).execute()
        print(f"(EPI.✓) Flushed episode: {resp}")
    except Exception as e:
        print(f"(EPI.e) Supabase insert to 'episodes' failed: {e}")


def _start_new_episode_from_rows(rows: List[Dict[str, Any]]) -> EpisodeState:
    """
    Initialize a new EpisodeState from a list of screenshot rows (non-empty).
    """
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    first_ts = _parse_timestamp(rows_sorted[0]["timestamp"])
    ep = EpisodeState(start_time=first_ts, end_time=first_ts)
    for r in rows_sorted:
        ep.add_screenshot(r)
    return ep


def advance_episoder(shot_row: Dict[str, Any]) -> None:
    """
    Main entrypoint. Call this once for each screenshot row you insert into 'screenshots'.

    Expected keys in shot_row:
        - 'timestamp': 'YYYY-MM-DD_HH-MM-SS'
        - plus anything else your coherence model later needs (project_label, goal_type, etc.).
    """
    global current_episode, pending_buffer

    ts_str = shot_row.get("timestamp")
    if not ts_str:
        print("(EPI.e) screenshot row missing 'timestamp'; skipping episoding.")
        return

    shot_time = _parse_timestamp(ts_str)

    # Case 1: no current episode yet
    if current_episode is None:
        current_episode = EpisodeState(start_time=shot_time, end_time=shot_time)
        current_episode.add_screenshot(shot_row)
        pending_buffer = []
        print(f"(EPI.1) Started first episode at {shot_time}")
        return

    # Case 2: we have an episode; ask coherence model
    try:
        score = coherence_with_episode(current_episode, shot_row)
    except NotImplementedError as e:
        # For now, just attach everything to one long episode so the rest of the app works.
        print(f"(EPI.warn) {e} – defaulting to single growing episode.")
        current_episode.add_screenshot(shot_row)
        return
    except Exception as e:
        print(f"(EPI.e) coherence_with_episode failed: {e}")
        # Conservative fallback: treat as same episode
        current_episode.add_screenshot(shot_row)
        return

    # High coherence: screenshot belongs to current episode
    if score >= SAME_EPISODE_THRESHOLD:
        if pending_buffer:
            # brief detour → treat buffered screenshots as noise inside same episode
            print(
                f"(EPI.2) Coherent again after detour; attaching {len(pending_buffer)} "
                "buffered screenshots to current episode."
            )
            for r in pending_buffer:
                current_episode.add_screenshot(r)
            pending_buffer = []

        current_episode.add_screenshot(shot_row)
        print(
            f"(EPI.3) Extended episode: start={current_episode.start_time}, "
            f"end={current_episode.end_time}, count={len(current_episode.screenshot_rows)}"
        )
        return

    # Low coherence: may be noise or start of a new task
    pending_buffer.append(shot_row)
    print(
        f"(EPI.4) Low coherence ({score:.2f}); buffered size={len(pending_buffer)} "
        f"(threshold={MIN_SWITCH_SCREENS})."
    )

    if len(pending_buffer) < MIN_SWITCH_SCREENS:
        # Not enough evidence to split; keep watching
        return

    # Enough consecutive low-coherence screenshots: declare a task switch
    print(
        "(EPI.5) Detected stable context change; closing current episode and "
        "starting a new one from buffered screenshots."
    )
    _flush_episode_to_db(current_episode)
    current_episode = _start_new_episode_from_rows(pending_buffer)
    pending_buffer = []
    print(
        f"(EPI.6) New episode started: start={current_episode.start_time}, "
        f"end={current_episode.end_time}, count={len(current_episode.screenshot_rows)}"
    )
