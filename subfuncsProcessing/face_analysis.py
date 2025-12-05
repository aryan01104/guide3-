import cv2, mediapipe as mp, numpy as np, time, collections

FPS = 10
BATCH_SEC = 15
EVAL_INTERVAL = 120   # seconds between analyses
FRAME_BATCH = BATCH_SEC * FPS

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

def aspect_ratio(pts, a,b,c,d,e,f):
    vert = np.linalg.norm(pts[a]-pts[d]) + np.linalg.norm(pts[b]-pts[e])
    horz = np.linalg.norm(pts[c]-pts[f])
    return vert / (2.0 * horz + 1e-6)

LEFT_EYE = [159,145,33,133,160,144]
MOUTH    = [13,14,78,308,82,312]
def eye_AR(pts): a,b,c,f,d,e = LEFT_EYE; return aspect_ratio(pts,a,b,c,d,e,f)
def mouth_AR(pts): a,b,c,f,d,e = MOUTH; return aspect_ratio(pts,a,b,c,d,e,f)
def points_from_landmarks(landmarks,w,h):
    return np.array([(lm.x*w, lm.y*h) for lm in landmarks.landmark], dtype=np.float32)

def analyze_window(metrics_list):
    ears = np.array([m["EAR"] for m in metrics_list])
    mars = np.array([m["MAR"] for m in metrics_list])
    eye_closed = ears < 0.20
    perclos = eye_closed.mean()
    yawns = (mars > 0.70).mean()

    drowsy = np.clip(0.6*perclos + 0.4*min(1.0,2*yawns), 0, 1)
    engaged = np.clip(1.0 - drowsy - 0.2, 0, 1)
    if drowsy > 0.6: state = "fatigued"
    elif drowsy > 0.3: state = "neutral"
    else: state = "engaged"
    return {"state":state, "drowsy":float(drowsy), "engaged":float(engaged)}