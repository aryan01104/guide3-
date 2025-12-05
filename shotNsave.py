
"""
screenshots ->
save to folder

"""

from PIL import ImageGrab
from datetime import datetime
import os
import cv2 as cv

screenshots_path = "screenshots/"
camshots_path = "camshots/"
# Initialize webcam (0 = default camera)
cam = cv.VideoCapture(0)

time = datetime.now.strftime("%Y-%m-%d_%H-%M-%S")

# Capture one frame
ret, frame = cam.read()

if ret:
    cv.imshow("Captured", frame)         
    cv.imwrite("captured_image.png", frame)  
    cv.waitKey(0)                      
    cv.destroyWindow("Captured")       
else:
    print("Failed to capture image.")

cam.release() 




os.makedirs(folder_path, exist_ok=True)

full_path = os.path.join(folder_path, filename = f"{time}.png")

screenshot = ImageGrab.grab()

screenshot.save(full_path, 'PNG')

print(f"Image saved successfully as: {full_path}")