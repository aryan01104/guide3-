# screenshot.py
import os
from datetime import datetime
from PIL import ImageGrab

def capture_screenshot(folder_path="raw/screenshots"):
    """
    Captures a full-screen screenshot and saves it as a PNG file, returns fullpath of file

    Args:
        folder_path (str): Directory where the screenshot will be saved.

    Returns:
        str | None: The path of the saved image, or None if capture failed.
    """
    os.makedirs(folder_path, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(folder_path, f"{ts}.png")

    try:
        screenshot = ImageGrab.grab()
        screenshot.save(full_path, "PNG")
        print(f"Screenshot saved successfully as: {full_path}")
        return full_path
    except Exception as e:
        print(f"Failed to capture screenshot: {e}")
        return None


# Optional: allow running directly
if __name__ == "__main__":
    capture_screenshot()
