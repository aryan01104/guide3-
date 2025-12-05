# headshot.py
import os
from datetime import datetime
import time as pytime
import cv2 as cv

def capture_headshot(
    dir_path="raw/headshots", 
    camera_index=1, 
    warmup_seconds=1.0
):
    """
    Captures a single frame from the webcam and saves it as a PNG file.

    Args:
        dir_path (str): Directory where the image will be saved.
        camera_index (int): Index of the camera (0 or 1 usually).
        warmup_seconds (float): Time to warm up the camera before capture.

    Returns:
        str | None: The path of the saved image, or None if capture failed.
    """
    os.makedirs(dir_path, exist_ok=True)

    cam = cv.VideoCapture(camera_index, cv.CAP_AVFOUNDATION)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index}).")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(dir_path, f"{ts}.png")

    # warm up camera
    t0 = pytime.time()
    ret, frame = False, None
    while pytime.time() - t0 < warmup_seconds:
        ret, frame = cam.read()

    if ret and frame is not None:
        cv.imwrite(full_path, frame)
        print(f"Saved Headhost at: {full_path}")
        result = full_path
    else:
        print("Failed to capture Headshot.")
        result = None

    cv.destroyAllWindows()
    cam.release()
    return result


# Optional: allow running this file directly
if __name__ == "__main__":
    capture_headshot()
