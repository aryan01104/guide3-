from subfuncsInput.headshot import capture_headshot   # if/when you use it
from subfuncsInput.screenshot import capture_screenshot
from subfuncsChecks.connected import is_connected
from supabase_client import supabase
from schemas.forChat import analyze_screenshot_with_openai, ValidationError
from pathlib import Path
import subprocess, time, random, os, threading
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow/MediaPipe logs
from subfuncsProcessing.face_analysis import points_from_landmarks, eye_AR, mouth_AR, analyze_window, cv2, mp, FPS, EVAL_INTERVAL, FRAME_BATCH, BATCH_SEC
from datetime import datetime
from subfuncEp.episoder import advance_episoder
from subfuncEp.label_canonicalizer import canonicalize_workstream, canonicalize_deliverable




INTERVAL_1 = 10  # seconds between captures

def polite_sleep_backoff(i: int):
    time.sleep(min(30, 2**i + random.random()*0.5))


def screenshot_loop():
    while True:
        try:
            #print("(S.1) proceeding to take screenshot")
            screenshot_location = capture_screenshot()
        except Exception as e:
            print(f"(S.e(1)) error clicking screenshot: {e}")
        else:
            #print("(S.2) checking for internet connection")
            if is_connected():
                try:
                    #print("(S.3) collecting vision summary from OpenAI")
                    summary = analyze_screenshot_with_openai(screenshot_location)
                except ValidationError as ve:
                    print(f"(S.e(3))Schema validation failed: {ve}")
                except Exception as e:
                    print(f"(S.e(3))OpenAI vision error: {e}")
                else:
                    #print("(S.4) inserting into Supabase")
                    # 1) canonicalize workstream & deliverable
                    ws_id, ws_label = canonicalize_workstream(summary.workstream_label)
                    dv_id, dv_label = canonicalize_deliverable(ws_id, summary.deliverable_label)

                    # 2) build row for screenshots insert
                    allowed_cols = {
                        "topic",
                        "semantic_summary",
                        "workstream_label",
                        "deliverable_label",
                        "app_or_website",
                        "app_bucket",
                        "url",
                        "work_type",
                        "goal_type",
                        "confidence",
                    }

                    row = {k: v for k, v in summary.model_dump().items() if k in allowed_cols}

                    row["timestamp"] = Path(screenshot_location).stem
                    row["workstream_id"] = ws_id
                    row["deliverable_id"] = dv_id
                    row["workstream_label"] = ws_label      # canonical text
                    row["deliverable_label"] = dv_label     # canonical text
                    try:
                        db_resp = supabase.table("screenshots").insert(row).execute()
                        #print(db_resp)
                        # TODO: remove the above pound to see what gets inserted
                        # episoding: update in-memory session state and flush closed episodes
                        if db_resp.data:
                            try:
                                advance_episoder(db_resp.data[0])
                            except Exception as epi_e:
                                print(f"(EPI.e) Episoding failed: {epi_e}")
                        else:
                            print("(EPI.e) screenshots insert returned no data; skipping episoding.")
                    except Exception as e:
                        print(f"(S.e(4))Supabase insert failed: {e}")
            else:
                print("(S.e(2))no internet connection, keeping image for later")
                # optional: mark as pending
                try:
                    os.rename(screenshot_location, screenshot_location + ".pending")
                except Exception as e:
                    print(f"rename failed: {e}")

        time.sleep(INTERVAL_1)


# //////////////// FACE LOOP /////////////////////

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def headshot_batch_loop():
    while True:
        print("(F.1) HS loop active")
        cap = None
        face_mesh = None

        try:
            cap = cv2.VideoCapture(0)
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
            )

            metrics_list = []
            start = time.time()
            print("(F.2) Loaded vals and libraries to start data-collecting")
            while time.time() - start < BATCH_SEC:
                ret, frame = cap.read()
                if not ret:
                    print("(F.e1) frame not captured")
                    time.sleep(0.1)
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    pts = points_from_landmarks(res.multi_face_landmarks[0], w, h)
                    metrics_list.append(
                        {"EAR": eye_AR(pts), "MAR": mouth_AR(pts)}
                    )
                    
                time.sleep(1 / FPS)
            print("(F.3) done appending to metric list for 15s window")

        finally:
            # Make sure resources are released even if something blew up
            if cap is not None:
                cap.release()
            if face_mesh is not None:
                face_mesh.close()
            print("(F.4) data resource-objects released")


        if metrics_list:
            state = analyze_window(metrics_list)
            print("(F.5) received analysis result")
            try:
                db_resp = supabase.table("facevals").insert(state).execute()
                print(db_resp)
                print("(F.6) analysis posted")
            except Exception as e:
                print(f"Supabase headshot insert failed: {e}")
        else:
            print("No face metrics collected in this window; skipping DB insert.")

        # Wait before next 2-minute run
        print("(F.7) received analysis result")
        time.sleep(EVAL_INTERVAL)


# ---- main ----
if __name__ == "__main__":
    t1 = threading.Thread(target=screenshot_loop, daemon=True)
    #t2 = threading.Thread(target=headshot_batch_loop, daemon=True)

    t1.start()
    #t2.start()

    # keep main thread alive
    while True:
        time.sleep(1)
