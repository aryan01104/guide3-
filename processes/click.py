import subprocess

p1 = subprocess.Popen(["python3", "subprocesses/headshot.py"])
p2 = subprocess.Popen(["python3", "subprocesses/screenshot.py"])

# both are running independently now
p1.wait()  # wait until webcam_capture.py ends
p2.wait()  # wait until screenshot_capture.py ends


