"""
RTSP Live Detection App
- People counter (line crossing IN/OUT)
- Mobile phone detection (person_with_phone alerts)
- Live stream from RTSP camera
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import os
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
import json

app = Flask(__name__)

RTSP_URL = "rtsp://admin:Thinkneural%2312@192.168.150.10:554/Streaming/Channels/101"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
VIOLATIONS_DIR = os.path.join(os.path.dirname(__file__), "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# ── Centroid tracker ────────────────────────────────────────────────────────

class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_id = 0
        self.objects = {}        # id -> centroid
        self.disappeared = {}    # id -> frames missing
        self.crossed = {}        # id -> {'start_side': str}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.crossed[self.next_id] = {'start_side': None}
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        self.crossed.pop(oid, None)

    def update(self, boxes):
        if len(boxes) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        centroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes],
            dtype="int"
        )

        if not self.objects:
            for c in centroids:
                self.register(c)
        else:
            oids = list(self.objects.keys())
            existing = list(self.objects.values())
            D = dist.cdist(np.array(existing), centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c:
                    continue
                self.objects[oids[r]] = centroids[c]
                self.disappeared[oids[r]] = 0
                used_r.add(r); used_c.add(c)
            for r in set(range(D.shape[0])) - used_r:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1])) - used_c:
                self.register(centroids[c])
        return self.objects


# ── Shared state ─────────────────────────────────────────────────────────────

class StreamState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.stats = {
            "in_count": 0,
            "out_count": 0,
            "current_persons": 0,
            "mobile_violations": 0,
            "fps": 0.0,
            "status": "connecting",
            "roi_y": None,        # set after first frame
        }
        self.violations = []
        self.roi_y = None         # fraction 0.0-1.0 of frame height
        self.running = False

state = StreamState()


# ── Detection thread ──────────────────────────────────────────────────────────

def detection_loop():
    model = YOLO(MODEL_PATH)
    tracker = CentroidTracker(max_disappeared=40)
    counted_ids = set()
    in_count = out_count = mobile_violations = 0
    last_alert_time = 0
    mobile_streak = 0

    cap = None

    def open_stream():
        c = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        return c

    state.stats["status"] = "connecting"
    cap = open_stream()
    frame_times = []

    while state.running:
        if not cap.isOpened():
            state.stats["status"] = "reconnecting"
            time.sleep(3)
            cap = open_stream()
            continue

        ret, frame = cap.read()
        if not ret:
            state.stats["status"] = "reconnecting"
            cap.release()
            time.sleep(2)
            cap = open_stream()
            continue

        state.stats["status"] = "live"
        h, w = frame.shape[:2]

        # Set default ROI line to 55% down the frame on first frame
        if state.roi_y is None:
            state.roi_y = 0.55
        roi_px = int(state.roi_y * h)

        t0 = time.time()

        # Run YOLO detection
        results = model(frame, conf=0.35, iou=0.45, verbose=False)[0]

        person_boxes = []
        mobile_this_frame = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            if name in ("person", "person_with_phone", "person_without_phone"):
                person_boxes.append((x1, y1, x2, y2))
            if name == "person_with_phone":
                mobile_this_frame = True

        # Mobile streak alerting
        if mobile_this_frame:
            mobile_streak += 1
        else:
            mobile_streak = 0

        if mobile_streak >= 3:
            now = time.time()
            if now - last_alert_time >= 5:
                last_alert_time = now
                mobile_violations += 1
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"violation_{ts}.jpg"
                cv2.imwrite(os.path.join(VIOLATIONS_DIR, fname), frame)
                state.violations.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": fname,
                })

        # Centroid tracking
        objects = tracker.update(person_boxes)

        for oid, (cx, cy) in objects.items():
            info = tracker.crossed[oid]
            side = "top" if cy < roi_px else "bottom"
            if info["start_side"] is None:
                info["start_side"] = side
            if oid not in counted_ids:
                if info["start_side"] == "top" and side == "bottom":
                    out_count += 1
                    counted_ids.add(oid)
                elif info["start_side"] == "bottom" and side == "top":
                    in_count += 1
                    counted_ids.add(oid)

        # ── Draw annotations ──────────────────────────────────────────────
        annotated = results.plot()

        # ROI line
        cv2.line(annotated, (0, roi_px), (w, roi_px), (0, 220, 255), 2)
        cv2.putText(annotated, "COUNT LINE", (w - 180, roi_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Tracker dots + IDs
        for oid, (cx, cy) in objects.items():
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 80), -1)
            cv2.putText(annotated, f"#{oid}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

        # Stats overlay (top-left panel)
        panel_h, panel_w = 130, 320
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        lines = [
            (f"IN  : {in_count}", (80, 220, 80)),
            (f"OUT : {out_count}", (80, 160, 255)),
            (f"NOW : {len(objects)} person(s)", (220, 220, 80)),
            (f"MOBILE ALERTS : {mobile_violations}", (80, 80, 255)),
        ]
        for i, (txt, color) in enumerate(lines):
            cv2.putText(annotated, txt, (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # FPS
        frame_times.append(time.time() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        cv2.putText(annotated, f"FPS {fps:.1f}", (w - 110, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Push frame
        with state.lock:
            state.frame = annotated.copy()
            state.stats.update({
                "in_count": in_count,
                "out_count": out_count,
                "current_persons": len(objects),
                "mobile_violations": mobile_violations,
                "fps": round(fps, 1),
                "roi_y": state.roi_y,
            })

    if cap:
        cap.release()


# ── MJPEG stream generator ────────────────────────────────────────────────────

def generate_frames():
    while True:
        with state.lock:
            frame = state.frame

        if frame is None:
            # Placeholder while connecting
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            msg = state.stats.get("status", "connecting").upper()
            cv2.putText(placeholder, msg, (280, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            frame = placeholder

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.033)   # ~30 fps to browser


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("rtsp_dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with state.lock:
        return jsonify(state.stats)


@app.route("/violations")
def violations():
    return jsonify({"violations": state.violations, "total": len(state.violations)})


@app.route("/set_roi", methods=["POST"])
def set_roi():
    data = request.get_json()
    frac = float(data.get("y_fraction", 0.55))
    frac = max(0.1, min(0.9, frac))
    state.roi_y = frac
    return jsonify({"ok": True, "roi_y": frac})


@app.route("/reset_counts", methods=["POST"])
def reset_counts():
    # Reset is reflected on next tracker cycle via state
    with state.lock:
        state.stats["in_count"] = 0
        state.stats["out_count"] = 0
        state.stats["mobile_violations"] = 0
        state.violations.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    state.running = True
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print("  RTSP LIVE DETECTION")
    print("  Open: http://localhost:5050")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
