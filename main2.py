from subfuncsInput.headshot import capture_headshot
from subfuncsInput.screenshot import capture_screenshot
from subfuncsConditions.connected import is_connected
from morphik import Morphik
from schemas.forMorphik import ScreenshotSummary, summarize_screenshot, morphik
from supabase_client import supabase
import os



"""
WORK feature extraction

for every 10s:

    click
    if fail:
        notify dev, and return error
    else:
        if internet:
            pipeline output to llm/processor with template
            similarly for headhot
            if fail:
                notify dev, return error
                retain picture on user computer
            else:
                delete pic from user computer
                save features to supabase
        else:
            pass

"""

import subprocess, time
from morphik import Morphik

try:
    morphik.query(query="ping", k=0)  # or a lightweight call your SDK supports
    print("Morphik auth&endpoint OK")
except Exception as e:
    print("Morphik not reachable/authenticated:", e)
    # bail out or retry/backoff


INTERVAL = 10  # seconds between captures


#//// SCREENSHOT LOOP //////////////
while True:
    
    try:
        print("(1) proceeding to take screenshot")
        screenshot_location = capture_screenshot()
    except Exception as e:
        print(f"(a) error clicking screenshot: {e}")
    else:
        print("(2) checking for internet connection")
        if is_connected():
            try:
                print("(3) sending to interpreter for feature extraction")
                doc_uid = screenshot_location[0:-4]
                morphik.ingest_file(screenshot_location, metadata={"doc_uid": doc_uid})
                print(screenshot_location) # for sanity check
            except Exception as e:
                print(f"Screenshot loop/ error w/ llm ingesting file: {e}")
            else:
                try:
                    print("(4) collecting the response")
                    response = summarize_screenshot(doc_uid)
                except Exception as e:
                    print(f"Screenshot loop/ error w/ summarizing file: {e}")
                else:
                    # check against type, and send data to db
                    allowed_cols = {"topic", "app_or_website", "url", "work_type", "confidence"}
                    print("(5) matching output to pydantic scheme U db cols")
                    summary = ScreenshotSummary(**response.completion)
                    row = {k: v for k, v in summary.model_dump().items() if k in allowed_cols}
                    row["timestamp"] = screenshot_location[0:-4] # to remove .png
                    db_resp = (supabase.table("screenshots").insert(row).execute())
                    print(db_resp)
                #NOTE:delete picture from user/local location in production
                    # will keep the pictures now for reference
        else:
            print("no internet connection, so saved image")
            # we are saving all files for now
            # need method to denote unprocessed images
            # -> will prime it
            address = screenshot_location + "*"
            os.rename(screenshot_location, address)

    
    time.sleep(INTERVAL)
