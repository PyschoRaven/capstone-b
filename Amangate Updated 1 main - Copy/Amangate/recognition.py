import base64
import logging
import os
import threading
import time
from pathlib import Path

# oneDNN makes RetinaFace ~4x slower on this CPU (measured 18s -> 4.6s per frame).
# Must be set before TensorFlow is imported (via deepface below).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import retinaface.RetinaFace as _retinaface
from deepface import DeepFace

# Speed up RetinaFace. DeepFace pads each frame with a 50% border before detection
# (640x480 -> 1280x960) and RetinaFace then upscales it further, which took ~18s per
# frame on CPU. Webcam faces are large, so detect on a copy at most 640px wide
# without upscaling, then map the boxes and landmarks back to the full-size frame.
# The face crop used for ArcFace still comes from the full-resolution image.
_retinaface_detect = _retinaface.detect_faces
_MAX_DETECT_SIDE   = 640

def _inner_photo(img):
    """
    DeepFace (align=True) adds a black border of half the photo's size on every
    side before detection, so ~75% of the pixels RetinaFace searches are black.
    Return (photo, x_offset, y_offset) with that border cut off, or (img, 0, 0)
    when the border isn't there.
    """
    h, w = img.shape[:2]
    by, bx = h // 4, w // 4
    if by < 8 or bx < 8:
        return img, 0, 0
    border_is_black = not (img[:by].any() or img[h - by:].any() or img[:, :bx].any() or img[:, w - bx:].any())
    if not border_is_black:
        return img, 0, 0
    return img[by:h - by, bx:w - bx], bx, by


def _detect_faces_fast(img_path, threshold=0.9, model=None, allow_upscaling=True):
    if not isinstance(img_path, np.ndarray):
        return _retinaface_detect(img_path, threshold=threshold, model=model, allow_upscaling=False)

    h, w  = img_path.shape[:2]
    # Same detection scale as before (based on the full bordered size) — so the
    # face is searched at exactly the same resolution — but only over the real
    # photo, not the black border: ~4x fewer pixels, same result.
    scale = min(1.0, _MAX_DETECT_SIDE / max(h, w))
    photo, ox, oy = _inner_photo(img_path)
    ph, pw = photo.shape[:2]
    small = cv2.resize(photo, (max(1, int(pw * scale)), max(1, int(ph * scale)))) if scale < 1 else photo
    faces = _retinaface_detect(small, threshold=threshold, model=model, allow_upscaling=False)

    if isinstance(faces, dict):
        for face in faces.values():
            x1, y1, x2, y2 = face["facial_area"]
            face["facial_area"] = [int(round(x1 / scale + ox)), int(round(y1 / scale + oy)),
                                   int(round(x2 / scale + ox)), int(round(y2 / scale + oy))]
            face["landmarks"]   = {k: [float(pt[0]) / scale + ox, float(pt[1]) / scale + oy]
                                   for k, pt in face["landmarks"].items()}
    return faces

_retinaface.detect_faces = _detect_faces_fast


def _detect_faces_previous(img_path, threshold=0.9, model=None, allow_upscaling=True):
    """The previous detection (searched the black border too). Only used by
    compare_with_previous_detection() to confirm the speed-up changes nothing."""
    h, w  = img_path.shape[:2]
    scale = min(1.0, _MAX_DETECT_SIDE / max(h, w))
    small = cv2.resize(img_path, (int(w * scale), int(h * scale))) if scale < 1 else img_path
    faces = _retinaface_detect(small, threshold=threshold, model=model, allow_upscaling=False)
    if scale < 1 and isinstance(faces, dict):
        for face in faces.values():
            face["facial_area"] = [int(round(v / scale)) for v in face["facial_area"]]
            face["landmarks"]   = {k: [float(c) / scale for c in pt] for k, pt in face["landmarks"].items()}
    return faces


# Check: compare the new (faster) detection with the previous one on real scans
# and log the difference. Confirmed identical on real scans (21 Oct: embedding
# difference 0.000-0.016, same match %), so it is off — it competes with the next scan.
VERIFY_DETECTION_SPEEDUP = False
_detect_lock = threading.Lock()


MODEL = "ArcFace"
# Swap to try a faster backend (e.g. "yunet", "mediapipe") — re-validate match
# accuracy against your enrolled users before changing this in production.
DETECTOR_BACKEND = "retinaface"
CASCADE_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "haarcascade_frontalface_default.xml"
)

FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))

if FACE_CASCADE.empty():
    raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")
_MODEL_READY = False


def decode_image(image_data_url):
    """Decode a base64 image data URL into an OpenCV frame."""
    image_data = image_data_url.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def check_face_center(image_data_url, vehicle=False):
    """
    Quickly detect a face position with OpenCV's Haar Cascade.

    Used ONLY for the live on-screen guidance ("move left/right"). It never
    decides access: whether a face is there, whether it is one person, and who
    it is are all decided by RetinaFace + ArcFace in analyze_face().
    """
    try:
        frame = decode_image(image_data_url)
        if frame is None:
            return {
                "centered": False,
                "reason": "error",
                "message": "Position verification error",
            }

        img_h, img_w = frame.shape[:2]
        max_width = 480
        scale = min(1, max_width / img_w)
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # minSize 40 px on a 480 px frame (~8% of the width): still finds the
        # smaller faces of wide-angle iPad cameras
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            return {
                "centered": False,
                "reason": "no_face",
                "message": "No face detected - please look at the camera",
            }

        if len(faces) > 1 and not vehicle:
            return {
                "centered": False,
                "reason": "multiple_faces",
                "message": "Multiple faces detected - one person at a time",
            }

        # Vehicle gates: passengers may be visible — guide the driver (largest face)
        x, _y, w, _h = max(faces, key=lambda f: f[2] * f[3])
        if scale < 1:
            x = int(x / scale)
            w = int(w / scale)

        face_cx = x + (w / 2)
        messages = []

        # Camera frames are not mirrored: what is on the image's left is on the
        # person's right. Directions below are from the person's point of view
        # (the terminal shows a mirrored preview, so they match what they see).
        # A driver can't step sideways, so vehicle gates accept a wider area and a smaller face
        edge      = 0.20 if vehicle else 0.35
        min_ratio = GUIDE_MIN_FACE_RATIO_VEHICLE if vehicle else GUIDE_MIN_FACE_RATIO
        if face_cx < img_w * edge:
            messages.append("Move slightly to the left")
        elif face_cx > img_w * (1 - edge):
            messages.append("Move slightly to the right")
        elif w < img_w * min_ratio:
            messages.append("Move closer to the camera")

        if not messages:
            return {"centered": True, "reason": "centered", "message": "Face centered"}

        return {
            "centered": False,
            "reason": "position_error",
            "message": " | ".join(messages),
        }

    except Exception:
        logging.exception("Face-position check failed")
        return {
            "centered": False,
            "reason": "error",
            "message": "Position verification error — check server terminal",
        }


class MultipleFacesError(Exception):
    """RetinaFace found more than one person in the frame."""


class FaceCheckError(Exception):
    """The face was found but the photo is not good enough (blur, light, size)."""
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason  = reason
        self.message = message


# A second face counts as "another person" only if it is at least this big
# compared with the largest face — ignores tiny faces far in the background.
SECOND_FACE_MIN_RATIO = 0.2

# --- Photo quality (checked on the face before ArcFace's result is used) ---
# Enough face detail for ArcFace (its input is 112x112): judged in real pixels on
# the face, not as a share of the picture — wide-angle iPad cameras make the face
# a small share of the picture even at a normal distance.
MIN_FACE_PX          = 112
MIN_FACE_PX_VEHICLE  = 96    # drivers sit further from the camera
# Live "Move closer" hint (fast guide, on a 480 px preview frame)
GUIDE_MIN_FACE_RATIO         = 0.11
GUIDE_MIN_FACE_RATIO_VEHICLE = 0.08
MIN_SHARPNESS   = 15     # variance of the Laplacian on the face (lower = blurrier)
MIN_BRIGHTNESS  = 40     # mean grey level of the face, 0-255
MAX_BRIGHTNESS  = 225

# --- Head turn (yaw) from RetinaFace's eye + nose points ---
# 0 = facing the camera, + = turned to the person's left, - = to their right.
# Roughly: 0.15 ≈ 15°, 0.3 ≈ 30°, 0.45 ≈ 40°.
YAW_STRAIGHT    = 0.18   # "looking straight" for enrollment photo 1
YAW_SIDE_MIN    = 0.10   # enrollment side photos must be turned at least this much
YAW_SIDE_MAX    = 0.90   # ...and not more than this, about 45° (no full profiles)
YAW_SCAN_MAX    = 0.45   # gate scans: more turned than this -> ask to turn back


def estimate_yaw(facial_area):
    """How far the head is turned, from where the nose sits between the eyes."""
    left_eye, right_eye, nose = (facial_area.get(k) for k in ("left_eye", "right_eye", "nose"))
    if not (left_eye and right_eye and nose):
        return 0.0
    eye_mid  = (left_eye[0] + right_eye[0]) / 2
    eye_dist = abs(left_eye[0] - right_eye[0]) or 1
    # Frames are not mirrored, so the person's left is towards the image's right
    return round((nose[0] - eye_mid) / eye_dist, 3)


# --- Anti-spoofing (DeepFace's Silent-Face MiniFASNet, runs on PyTorch) ---
# Looks at the face crop and decides "real face" or "fake" (phone/tablet screen,
# printed photo). Modes:
#   "off"     - not run
#   "test"    - run and record the result in the log, but never block anyone
#   "enforce" - fakes are denied
# Enforced from 23 Oct: every real scan so far was judged "real" (96-100% sure).
ANTI_SPOOF_MODE = "enforce"
_spoof_model    = None
_spoof_lock     = threading.Lock()


class SpoofError(Exception):
    """Anti-spoofing judged the face to be fake (screen or printed photo)."""


def check_spoof(frame, facial_area):
    """Return (is_real, score) — score is the model's certainty in its answer, 0-1."""
    global _spoof_model
    if _spoof_model is None:
        with _spoof_lock:
            if _spoof_model is None:
                from deepface.modules import modeling
                _spoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")
    box = tuple(int(facial_area[k]) for k in ("x", "y", "w", "h"))
    is_real, score = _spoof_model.analyze(img=frame, facial_area=box)
    return bool(is_real), round(float(score), 3)


def turn_back_message(yaw):
    """Guidance for a head that is turned too far (person's point of view)."""
    return "Turn your head slightly to the right" if yaw > 0 else "Turn your head slightly to the left"


def check_quality(frame, facial_area, min_face_px=MIN_FACE_PX):
    """Return None when the face is usable, otherwise (reason, message)."""
    x, y, w, h = (int(facial_area[k]) for k in ("x", "y", "w", "h"))
    if w < min_face_px:
        return "too_small", "Move closer to the camera"

    face = frame[max(0, y):y + h, max(0, x):x + w]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(cv2.resize(face, (160, int(160 * h / max(w, 1)) or 160)), cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    if brightness < MIN_BRIGHTNESS:
        return "too_dark", "Too dark - please find better light"
    if brightness > MAX_BRIGHTNESS:
        return "too_bright", "Too bright - avoid strong light on your face"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        return "blurry", "Hold still - the photo is blurry"
    return None


def analyze_face(image_data_url, check=True, vehicle=False, run_spoof=True):
    """
    The AI pipeline for one photo:
      1. RetinaFace finds the face (+ eyes, nose, mouth) and straightens it,
      2. ArcFace turns the aligned face into a 512-number embedding.
    Returns {"embedding", "box", "yaw"} for the largest face.

    Raises ValueError when no face is found, MultipleFacesError when more than
    one person is in the frame, and FaceCheckError when the photo is too
    small, dark, bright or blurry (only when check=True).

    vehicle=True (vehicle gates): the largest face is the driver's, closest to
    the camera — passengers are ignored instead of failing with "one person
    only", and a smaller face is accepted.

    run_spoof=False skips the anti-spoofing model for this call. Used during a
    gate scan to defer the (relatively expensive) spoof check to the one frame
    that actually produced a match, instead of running it on every frame.
    """
    timing = {}
    started = time.perf_counter()
    frame = decode_image(image_data_url)
    results = DeepFace.represent(
        img_path=frame,
        model_name=MODEL,
        enforce_detection=True,
        align=True,
        detector_backend=DETECTOR_BACKEND,
    )
    timing["retinaface+arcface_ms"] = round((time.perf_counter() - started) * 1000)

    def area(r):
        return r["facial_area"]["w"] * r["facial_area"]["h"]

    results = sorted(results, key=area, reverse=True)
    if not vehicle and len(results) > 1 and area(results[1]) >= SECOND_FACE_MIN_RATIO * area(results[0]):
        raise MultipleFacesError()

    face = results[0]
    if check:
        problem = check_quality(frame, face["facial_area"],
                                MIN_FACE_PX_VEHICLE if vehicle else MIN_FACE_PX)
        if problem:
            raise FaceCheckError(*problem)

    spoof = None
    if run_spoof and ANTI_SPOOF_MODE != "off":
        spoof_started = time.perf_counter()
        try:
            spoof = check_spoof(frame, face["facial_area"])
        except Exception:
            logging.exception("Anti-spoofing check failed.")
        timing["antispoof_ms"] = round((time.perf_counter() - spoof_started) * 1000)
        if ANTI_SPOOF_MODE == "enforce" and spoof and not spoof[0]:
            raise SpoofError()
    timing["face_px"] = int(face["facial_area"]["w"])
    return {"embedding": face["embedding"], "box": face["facial_area"],
            "yaw": estimate_yaw(face["facial_area"]), "spoof": spoof, "frame": frame, "timing": timing}


def compare_with_previous_detection(frame, new_embedding, new_box):
    """
    Run the previous detection on the same frame and return
    (embedding_difference, previous_embedding, box_difference_px).
    Swaps the detector briefly, so it holds a lock and is meant for a
    background thread after the scan result has already been sent.
    """
    with _detect_lock:
        _retinaface.detect_faces = _detect_faces_previous
        try:
            results = DeepFace.represent(img_path=frame, model_name=MODEL, enforce_detection=True,
                                         align=True, detector_backend=DETECTOR_BACKEND)
        finally:
            _retinaface.detect_faces = _detect_faces_fast
    prev = max(results, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    dist, _ = calculate_match(new_embedding, prev["embedding"])
    box_diff = max(abs(prev["facial_area"][k] - new_box[k]) for k in ("x", "y", "w", "h"))
    return float(dist), prev["embedding"], box_diff


def stored_views(stored):
    """
    A user's stored face: either one embedding (older enrollments) or a list of
    embeddings — one per view (straight, slightly left, slightly right).
    """
    if len(stored) and isinstance(stored[0], (list, tuple, np.ndarray)):
        return list(stored)
    return [stored]


def calculate_match(current_embedding, stored_embedding):
    """
    Calculate cosine distance between two embeddings.
    Returns (distance, confidence_percentage).
    """
    dot = np.dot(current_embedding, stored_embedding)
    norm = np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
    dist = 1 - dot / norm
    confidence = round((1 - dist) * 100, 1)
    return dist, confidence

def calculate_match_multi(current_embeddings, stored_embedding):
    """
    Compare several candidate embeddings (e.g. from consecutive frames)
    against every stored view of a user, keep the best (lowest-distance) match.
    """
    best_dist = None
    best_confidence = None
    for view in stored_views(stored_embedding):
        for emb in current_embeddings:
            dist, confidence = calculate_match(emb, view)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_confidence = confidence
    return best_dist, best_confidence

def warm_up_model():
    """Load the DeepFace model once during startup to reduce first-scan delay."""
    global _MODEL_READY
    if _MODEL_READY:
        return

    try:
        # Run one blank frame through the full pipeline so the face detector and
        # ArcFace are both loaded before the first real scan or enrollment.
        DeepFace.represent(
            img_path=np.full((480, 640, 3), 128, np.uint8),
            model_name=MODEL,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )
        if ANTI_SPOOF_MODE != "off":
            check_spoof(np.full((480, 640, 3), 128, np.uint8), {"x": 220, "y": 140, "w": 200, "h": 200})
        _MODEL_READY = True
        logging.info("DeepFace %s model, RetinaFace detector and anti-spoofing (%s) warmed up.", MODEL, ANTI_SPOOF_MODE)
    except Exception:
        logging.exception("DeepFace model warmup failed.")
