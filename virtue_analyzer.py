from datetime import datetime, timedelta
from typing import List, Dict, Any
import supabase
import os

# Assuming supabase_client.py contains the initialized Supabase client
from supabase_client import supabase

class Episode:
    def __init__(self, start_time: datetime, end_time: datetime, app_or_website: str, topic: str, work_type: str, screenshot_count: int):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = (end_time - start_time).total_seconds() / 60.0  # Duration in minutes
        self.app_or_website = app_or_website
        self.topic = topic
        self.work_type = work_type
        self.screenshot_count = screenshot_count

    def __repr__(self):
        return (f"Episode(start='{self.start_time}', end='{self.end_time}', duration={self.duration:.2f}m, "
                f"app='{self.app_or_website}', topic='{self.topic}', work_type='{self.work_type}', count={self.screenshot_count})")

def parse_timestamp(ts: str) -> datetime:
    """
    Parses a timestamp string that could be in one of two formats:
    1. A direct timestamp: '2025-10-25_16-29-50'
    2. A path containing a timestamp: 'raw/screenshots/2025-10-25_16-29-50'
    """
    try:
        return datetime.strptime(ts, '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        # Fallback for path-like timestamps
        stem = os.path.splitext(os.path.basename(ts))[0]
        return datetime.strptime(stem, '%Y-%m-%d_%H-%M-%S')

def fetch_screenshots(days_ago: int = 7) -> List[Dict[str, Any]]:
    """
    Fetches screenshot records from the Supabase 'screenshots' table from the last `days_ago` days.
    """
    print("Fetching screenshot data from Supabase...")
    # The 'timestamp' is a string like '2025-12-01_21-03-46'
    # We need to query based on this string format.
    start_date = datetime.now() - timedelta(days=days_ago)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    response = supabase.table('screenshots').select("*").gt('timestamp', start_date_str).order('timestamp', desc=False).execute()
    
    if response.data:
        print(f"Successfully fetched {len(response.data)} records.")
        return response.data
    else:
        print("No data found or there was an error.")
        return []

def group_into_episodes(screenshots: List[Dict[str, Any]], max_gap_minutes: int = 5) -> List[Episode]:
    """
    Groups a time-sorted list of screenshots into contiguous episodes of activity.
    """
    if not screenshots:
        return []

    episodes = []
    current_episode_screenshots = [screenshots[0]]

    for i in range(1, len(screenshots)):
        prev_shot = screenshots[i-1]
        current_shot = screenshots[i]

        prev_time = parse_timestamp(prev_shot['timestamp'])
        current_time = parse_timestamp(current_shot['timestamp'])
        
        time_gap = (current_time - prev_time).total_seconds() / 60.0

        # Define conditions for breaking an episode
        app_changed = prev_shot.get('app_or_website') != current_shot.get('app_or_website')
        gap_is_too_long = time_gap > max_gap_minutes

        if app_changed or gap_is_too_long:
            # Finalize the current episode
            start_time = parse_timestamp(current_episode_screenshots[0]['timestamp'])
            end_time = parse_timestamp(current_episode_screenshots[-1]['timestamp'])
            
            # For simplicity, we'll use the most common value for categorical attributes
            apps = [s.get('app_or_website', 'N/A') for s in current_episode_screenshots]
            topics = [s.get('topic', 'N/A') for s in current_episode_screenshots]
            work_types = [s.get('work_type', 'N/A') for s in current_episode_screenshots]

            dominant_app = max(set(apps), key=apps.count)
            dominant_topic = max(set(topics), key=topics.count)
            dominant_work_type = max(set(work_types), key=work_types.count)

            episode = Episode(
                start_time=start_time,
                end_time=end_time,
                app_or_website=dominant_app,
                topic=dominant_topic,
                work_type=dominant_work_type,
                screenshot_count=len(current_episode_screenshots)
            )
            episodes.append(episode)
            
            # Start a new episode
            current_episode_screenshots = [current_shot]
        else:
            # Continue the current episode
            current_episode_screenshots.append(current_shot)

    # Add the last episode
    if current_episode_screenshots:
        start_time = parse_timestamp(current_episode_screenshots[0]['timestamp'])
        end_time = parse_timestamp(current_episode_screenshots[-1]['timestamp'])
        apps = [s.get('app_or_website', 'N/A') for s in current_episode_screenshots]
        topics = [s.get('topic', 'N/A') for s in current_episode_screenshots]
        work_types = [s.get('work_type', 'N/A') for s in current_episode_screenshots]
        dominant_app = max(set(apps), key=apps.count)
        dominant_topic = max(set(topics), key=topics.count)
        dominant_work_type = max(set(work_types), key=work_types.count)
        episode = Episode(
            start_time=start_time,
            end_time=end_time,
            app_or_website=dominant_app,
            topic=dominant_topic,
            work_type=dominant_work_type,
            screenshot_count=len(current_episode_screenshots)
        )
        episodes.append(episode)

    print(f"Grouped into {len(episodes)} episodes.")
    return episodes


def calculate_discipline_score(episodes: List[Episode], today: datetime) -> float:
    """
    Calculates the Discipline score for a given day based on a list of episodes.
    """
    if not episodes:
        return 0.0

    # Signals for Discipline:
    # 1. Consistency of work across days. (Hard to measure with 1 day of data, will focus on daily metrics for now)
    # 2. Hitting a minimum deep-work threshold day after day.
    # 3. Low variance in “start work” time. (Needs more data)
    
    total_deep_work_minutes = 0
    work_start_times = []

    for e in episodes:
        if e.work_type == 'deep_work':
            total_deep_work_minutes += e.duration
            work_start_times.append(e.start_time)

    # Rule from user prompt: c_Discipline(e) = gamma1 * norm_deep_minutes_today + gamma2 * norm_streak_length
    # For now, we'll implement a simplified version for a single day.
    # Let's say the goal is 120 minutes of deep work per day.
    deep_work_goal_minutes = 120.0
    
    # Score based on deep work minutes. Capped at 1.0
    deep_work_score = min(total_deep_work_minutes / deep_work_goal_minutes, 1.0)
    
    # Let's add a penalty for starting work late. e.g. after 10 AM.
    late_start_penalty = 0.0
    if work_start_times:
        first_work_time = min(work_start_times)
        if first_work_time.hour > 10:
            late_start_penalty = 0.2 

    discipline_score = deep_work_score - late_start_penalty
    
    return max(0.0, discipline_score) # Ensure score is not negative


if __name__ == "__main__":
    print("Starting virtue analysis...")
    # 1. Fetch data
    screenshot_data = fetch_screenshots(days_ago=7)
    
    if screenshot_data:
        # 2. Group into episodes
        episodes = group_into_episodes(screenshot_data)
        
        for ep in episodes:
            print(ep)

        # 3. Calculate Discipline score
        today = datetime.now()
        discipline_score = calculate_discipline_score(episodes, today)
        print(f"Discipline Score for {today.date()}: {discipline_score:.2f}")

    print("Virtue analysis finished.")
