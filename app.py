import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import threading
import winsound
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- LOAD MODEL ----------------

model  = load_model("face_shape_model.h5")
labels = ["Heart", "Oblong", "Oval", "Round", "Square"]


# ---------------- MEDIAPIPE ----------------

mp_face_mesh = mp.solutions.face_mesh

# static_image_mode=True  → for uploaded photos
face_mesh_image = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)

# static_image_mode=False → for live webcam
face_mesh_video = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Face detection for video scanning
mp_face_det    = mp.solutions.face_detection
face_detection = mp_face_det.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4
)


# ---------------- GLOBAL VARIABLES ----------------

reference_encoding = None
alert_played       = False
THRESHOLD          = 0.7

# Stores result of full video scan
scan_result = {
    "status":       "idle",
    "progress":     0,
    "total_frames": 0,
    "found":        False,
    "match_count":  0,
    "best_frame":   -1,
    "accuracy":     0.0,
    "message":      ""
}


# ---------------- HAIRSTYLES ----------------

def get_hairstyles(shape):
    styles = {
        "Heart":  ["Side Swept Fringe", "Textured Layers", "Medium Length Cut"],
        "Oblong": ["Fringe Cut", "Layered Hair", "Medium Volume"],
        "Oval":   ["Pompadour", "Quiff", "Layered Cut"],
        "Round":  ["High Fade", "Spiky Hair", "Undercut"],
        "Square": ["Crew Cut", "Side Part", "Textured Crop"]
    }
    return styles.get(shape, [])


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- FACE SHAPE PREDICTION ----------------

def predict_face_shape(face):
    try:
        if face is None or face.size == 0:
            return "Unknown", 0

        img = cv2.resize(face, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        prediction = model.predict(img, verbose=0)
        index      = np.argmax(prediction)
        confidence = float(prediction[0][index]) * 100

        print(f"[DEBUG] Shape: {labels[index]} | Confidence: {confidence:.2f}%")
        return labels[index], confidence

    except Exception as e:
        print(f"[ERROR] predict_face_shape: {e}")
        return "Unknown", 0


# ---------------- IMAGE DETECTION ----------------

@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print(f"[DEBUG] Image saved: {path}")

        image = cv2.imread(path)
        if image is None:
            return jsonify({"error": "Image could not be loaded"})

        print(f"[DEBUG] Image shape: {image.shape}")

        # Use static mode FaceMesh for photos
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_image.process(rgb)

        if not results.multi_face_landmarks:
            print("[ERROR] No face detected")
            return jsonify({"error": "No face detected — use a clear frontal photo"})

        print("[DEBUG] Face landmarks found")

        landmarks = results.multi_face_landmarks[0]
        h, w, _   = image.shape
        points    = []

        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        points = np.array(points)
        x_min  = max(0, int(np.min(points[:, 0])) - 30)
        y_min  = max(0, int(np.min(points[:, 1])) - 30)
        x_max  = min(w, int(np.max(points[:, 0])) + 30)
        y_max  = min(h, int(np.max(points[:, 1])) + 30)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        face_crop = image[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return jsonify({"error": "Face crop failed"})

        shape, confidence = predict_face_shape(face_crop)
        if shape == "Unknown":
            return jsonify({"error": "Could not predict face shape — try another image"})

        cv2.putText(image,
                    f"{shape} {confidence:.2f}%",
                    (x_min, max(y_min - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        hairstyles  = get_hairstyles(shape)
        output_path = "static/output.jpg"
        cv2.imwrite(output_path, image)

        print(f"[DEBUG] Output saved: {output_path}")

        return jsonify({
            "face_shape": shape,
            "confidence": round(confidence, 2),
            "hairstyles": hairstyles,
            "image_url":  "/static/output.jpg"
        })

    except Exception as e:
        print(f"[ERROR] /detect: {e}")
        return jsonify({"error": str(e)})


# ---------------- WEBCAM STREAM ----------------

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame   = cv2.resize(frame, (640, 480))
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_video.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _   = frame.shape
            points    = []

            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            points = np.array(points)
            x_min  = max(0, np.min(points[:, 0]) - 20)
            y_min  = max(0, np.min(points[:, 1]) - 20)
            x_max  = min(w, np.max(points[:, 0]) + 20)
            y_max  = min(h, np.max(points[:, 1]) + 20)

            face_crop         = frame[y_min:y_max, x_min:x_max]
            shape, confidence = predict_face_shape(face_crop)

            cv2.putText(frame,
                        f"{shape} ({confidence:.1f}%)",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()


# ---------------- REFERENCE IMAGE ----------------

@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    global reference_encoding

    try:
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        image     = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            return jsonify({"error": "No face found in reference image"})

        reference_encoding = encodings[0]
        print("[DEBUG] Reference encoding saved")
        return jsonify({"message": "Reference uploaded successfully"})

    except Exception as e:
        print(f"[ERROR] upload_reference: {e}")
        return jsonify({"error": str(e)})


# ---------------- UPLOAD VIDEO ----------------

@app.route("/upload_video", methods=["POST"])
def upload_video():
    try:
        file = request.files.get("video")
        if not file:
            return jsonify({"error": "No video provided"}), 400

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        print(f"[DEBUG] Video saved: {save_path}")
        return jsonify({"message": "Video uploaded", "video_path": save_path})

    except Exception as e:
        print(f"[ERROR] upload_video: {e}")
        return jsonify({"error": str(e)})


# ================================================================
#  DRAW BLUE TICK
# ================================================================

def draw_blue_tick(frame, left, top, right, bottom):
    cx  = (left + right) // 2
    ty  = max(top - 30, 15)
    pts = np.array([
        [cx - 12, ty + 6],
        [cx - 3,  ty + 15],
        [cx + 14, ty - 2]
    ], np.int32)
    cv2.polylines(frame, [pts], False, (255, 0, 0), 3, cv2.LINE_AA)


# ================================================================
#  GET FACE BOXES USING MEDIAPIPE
# ================================================================

def get_face_boxes(frame):
    h, w, _ = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result   = face_detection.process(rgb)
    boxes    = []

    if result.detections:
        for det in result.detections:
            bb  = det.location_data.relative_bounding_box
            x1  = max(0, int(bb.xmin * w))
            y1  = max(0, int(bb.ymin * h))
            x2  = min(w, int((bb.xmin + bb.width)  * w))
            y2  = min(h, int((bb.ymin + bb.height) * h))
            # face_recognition format: (top, right, bottom, left)
            boxes.append((y1, x2, y2, x1))

    return boxes, rgb


# ================================================================
#  SCAN ENTIRE VIDEO IN BACKGROUND THREAD
#
#  FIX 1: Scans WHOLE video at once — not frame by frame streaming
#  FIX 2: Only marks the REFERENCE person — ignores all other faces
# ================================================================

def scan_video_thread(video_path):
    global scan_result, alert_played

    scan_result = {
        "status":       "processing",
        "progress":     0,
        "total_frames": 0,
        "found":        False,
        "match_count":  0,
        "best_frame":   -1,
        "accuracy":     0.0,
        "message":      "Scanning video..."
    }
    alert_played = False

    cap          = cv2.VideoCapture(video_path)
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count  = 0
    best_dist    = 1.0
    best_frame_img = None

    print(f"[DEBUG] Total frames in video: {total}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        # Update progress
        scan_result["progress"]     = int(frame_count / max(total, 1) * 100)
        scan_result["total_frames"] = frame_count

        # Skip every other frame for speed
        if frame_count % 2 != 0:
            continue

        if reference_encoding is None:
            continue

        # Detect ALL faces in this frame
        boxes, rgb = get_face_boxes(frame)
        if not boxes:
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):

            distance = face_recognition.face_distance(
                [reference_encoding], encoding
            )[0]

            print(f"[DEBUG] Frame {frame_count} | Distance: {distance:.4f}")

            # FIX 2: ONLY process THIS face if it matches reference
            # All other faces are completely IGNORED (no box drawn)
            if distance < THRESHOLD:

                accuracy = max(0.0, (1.0 - distance / THRESHOLD)) * 100
                accuracy = min(round(accuracy, 1), 100.0)

                scan_result["found"]       = True
                scan_result["match_count"] += 1

                # Keep the best matching frame (lowest distance)
                if distance < best_dist:
                    best_dist = distance

                    # Draw on a copy of the frame
                    marked = frame.copy()

                    # Blue box 3px — ONLY on reference person
                    cv2.rectangle(marked, (left, top), (right, bottom),
                                  (255, 0, 0), 3)

                    # Blue tick ✓
                    draw_blue_tick(marked, left, top, right, bottom)

                    # Accuracy label
                    cv2.putText(marked,
                                f"FOUND  {accuracy}%",
                                (left, bottom + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2, cv2.LINE_AA)

                    # Frame info
                    cv2.putText(marked,
                                f"Frame: {frame_count}  Accuracy: {accuracy}%",
                                (8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)

                    best_frame_img          = marked
                    scan_result["best_frame"] = frame_count
                    scan_result["accuracy"]   = accuracy

    cap.release()

    # Save best detected frame
    if best_frame_img is not None:
        cv2.imwrite("static/detected.jpg", best_frame_img)
        print(f"[DEBUG] Best frame saved: frame {scan_result['best_frame']}")

        # Play alert sound
        threading.Thread(target=winsound.Beep, args=(1500, 700)).start()

        scan_result["status"]  = "done"
        scan_result["message"] = f"Person found in {scan_result['match_count']} frames!"

    else:
        scan_result["status"]  = "not_found"
        scan_result["message"] = "Person not found in video"

    print(f"[DEBUG] Scan complete: {scan_result['message']}")


# ================================================================
#  START VIDEO SCAN
# ================================================================

@app.route("/start_scan", methods=["POST"])
def start_scan():
    global scan_result

    if reference_encoding is None:
        return jsonify({"error": "Please upload reference image first"})

    data       = request.get_json()
    video_path = data.get("video_path")

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"})

    # Reset result
    scan_result["status"] = "processing"

    # Run scan in background thread so page doesn't freeze
    t = threading.Thread(target=scan_video_thread, args=(video_path,))
    t.daemon = True
    t.start()

    return jsonify({"message": "Scan started"})


# ================================================================
#  CHECK SCAN PROGRESS (JS polls this every second)
# ================================================================

@app.route("/scan_status")
def scan_status():
    return jsonify(scan_result)


# ================================================================
#  VIDEO DETECTION STREAM (kept for compatibility)
# ================================================================

@app.route("/video_detection")
def video_detection():
    video_path = request.args.get("video")
    return Response(
        stream_video(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stream_video(video_path):
    """Simple stream — no detection, just shows video while scan runs in background."""
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        cv2.putText(frame,
                    "Scanning entire video... please wait",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)