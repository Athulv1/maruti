"""
Flask Web Application for Live Object Detection with ROI Counting
RTSP stream or local video with live detection + IN/OUT counting
"""

from flask import Flask, render_template, request, Response, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import cv2
import os
import json
import threading
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from inference import MobileOutDetector, CentroidTracker
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("⚠️ pygame not installed — audio alerts disabled")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('violations', exist_ok=True)

# Initialize pygame mixer for audio alerts
if HAS_PYGAME:
    try:
        pygame.mixer.init()
    except:
        HAS_PYGAME = False
        print("⚠️ Audio not available")
ALERT_SOUND_PATH = 'violation_alert.wav'

# PostgreSQL config
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'dbname': os.environ.get('DB_NAME', 'sakshiai_maruti'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'Postgres123'),
}

def _get_db():
    return psycopg2.connect(**DB_CONFIG)

def _init_db():
    with _get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS people_counts (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    in_count INTEGER NOT NULL DEFAULT 0,
                    out_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    CHECK (id = 1)
                )
            """)
            cur.execute("""
                INSERT INTO people_counts (id, in_count, out_count)
                VALUES (1, 0, 0)
                ON CONFLICT (id) DO NOTHING
            """)
            # Hourly logs table for historical reports
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hourly_logs (
                    id SERIAL PRIMARY KEY,
                    logged_at TIMESTAMP NOT NULL,
                    hour INTEGER NOT NULL,
                    in_count INTEGER NOT NULL DEFAULT 0,
                    out_count INTEGER NOT NULL DEFAULT 0,
                    violations INTEGER NOT NULL DEFAULT 0
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hourly_logs_date ON hourly_logs (logged_at)")
            # Seed sample data for past 7 days if table is empty
            cur.execute("SELECT COUNT(*) FROM hourly_logs")
            if cur.fetchone()[0] == 0:
                _seed_sample_data(cur)


def _seed_sample_data(cur):
    """Seed 7 days of realistic hourly data for demo purposes"""
    now = datetime.now()
    for day_offset in range(7, 0, -1):
        day = now - timedelta(days=day_offset)
        day_base = day.replace(hour=0, minute=0, second=0, microsecond=0)
        # Realistic hourly traffic pattern (busier 9AM-6PM)
        hourly_pattern = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 3, 7: 8,
            8: 15, 9: 25, 10: 30, 11: 28, 12: 20, 13: 22, 14: 27,
            15: 32, 16: 26, 17: 18, 18: 12, 19: 6, 20: 3, 21: 1, 22: 0, 23: 0
        }
        cumulative_in = 0
        cumulative_out = 0
        for hour in range(24):
            base = hourly_pattern[hour]
            # Add randomness
            in_delta = max(0, base + random.randint(-3, 5))
            out_delta = max(0, in_delta + random.randint(-4, 2))
            cumulative_in += in_delta
            cumulative_out += out_delta
            violations = random.randint(0, 2) if 8 <= hour <= 18 and random.random() > 0.6 else 0
            ts = day_base + timedelta(hours=hour)
            cur.execute(
                "INSERT INTO hourly_logs (logged_at, hour, in_count, out_count, violations) VALUES (%s, %s, %s, %s, %s)",
                (ts, hour, cumulative_in, cumulative_out, violations)
            )
    print("  ✅ Seeded 7 days of sample report data")

def _load_counts():
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT in_count, out_count, updated_at FROM people_counts WHERE id = 1")
                row = cur.fetchone()
                if row:
                    saved_date = row[2].date() if row[2] else None
                    if saved_date == datetime.now().date():
                        return (row[0], row[1])
                return (0, 0)
    except Exception as e:
        print(f"⚠️ DB load error: {e}")
        return 0, 0

def _save_counts(in_count, out_count):
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE people_counts
                    SET in_count = %s, out_count = %s, updated_at = NOW()
                    WHERE id = 1
                """, (in_count, out_count))
    except Exception as e:
        print(f"⚠️ DB save error: {e}")

try:
    _init_db()
    print("✅ PostgreSQL connected")
except Exception as e:
    print(f"⚠️ PostgreSQL unavailable: {e}")

# Video source configuration
USE_RTSP = os.environ.get('USE_RTSP', 'true').lower() == 'true'
RTSP_URL = 'rtsp://admin:Thinkneural%2312@192.168.150.11:554/Streaming/Channels/102'
TEST_VIDEO = 'videos/test.mp4'
MODEL_PATH = 'best.pt'  # Specialized phone detection model
ROI_CONFIG = 'roi_config.json'

# Global variables for live streaming
current_frame = None
processing_active = False
_saved_in, _saved_out = _load_counts()
processing_stats = {
    'frame_count': 0,
    'total_frames': 0,
    'in_count': _saved_in,
    'out_count': _saved_out,
    'fps': 0,
    'status': 'idle',
    'reset_requested': False,
}
violations_list = []  # Store mobile violation screenshots
frame_lock = threading.Lock()
from collections import deque
frame_buffer = deque(maxlen=150)  # Rolling buffer for GIF creation (~10 sec at 15fps)
frame_buffer_lock = threading.Lock()


def _load_violations_from_disk():
    """Scan violations/ folder and populate violations_list from existing files."""
    import re
    # pattern 1: mobile_violation_20260518_114814.jpg
    pat_auto = re.compile(r'.*_(\d{8})_(\d{6})\.(jpg|png)$')
    # pattern 2: Screenshot 2026-05-19 235123.png
    pat_screenshot = re.compile(r'Screenshot (\d{4}-\d{2}-\d{2}) (\d{6})\.(jpg|png)$', re.IGNORECASE)
    # pattern 3: 2026-05-20 172737.png (bare date-time filename)
    pat_bare = re.compile(r'^(\d{4}-\d{2}-\d{2}) (\d{6})\.(jpg|png)$', re.IGNORECASE)

    loaded = []
    vdir = Path('violations')
    if not vdir.exists():
        return

    all_files = sorted(vdir.glob('*.jpg')) + sorted(vdir.glob('*.png'))
    seen = set()

    for f in sorted(all_files, key=lambda x: x.name):
        if f.name in seen:
            continue
        seen.add(f.name)
        ts = None

        m = pat_auto.match(f.name)
        if m:
            try:
                ts = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            except ValueError:
                pass

        if ts is None:
            m = pat_screenshot.match(f.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1) + ' ' + m.group(2), '%Y-%m-%d %H%M%S')
                except ValueError:
                    pass

        if ts is None:
            m = pat_bare.match(f.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1) + ' ' + m.group(2), '%Y-%m-%d %H%M%S')
                except ValueError:
                    pass

        if ts is None:
            # Use file modification time as fallback
            ts = datetime.fromtimestamp(f.stat().st_mtime)

        entry = {
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'date': ts.strftime('%Y-%m-%d'),
            'frame_number': '--',
            'filename': f.name,
            'path': str(f),
        }
        gif_name = f.stem + '.gif'
        if (vdir / gif_name).exists():
            entry['gif'] = gif_name
        loaded.append(entry)

    violations_list.extend(loaded)
    print(f"✅ Loaded {len(loaded)} violations from disk")


_load_violations_from_disk()


def _midnight_reset_scheduler():
    """Background thread: triggers a count reset every day at midnight."""
    import datetime
    while True:
        now = datetime.datetime.now()
        next_midnight = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sleep_seconds = (next_midnight - now).total_seconds()
        time.sleep(sleep_seconds)
        processing_stats['reset_requested'] = True
        processing_stats['in_count'] = 0
        processing_stats['out_count'] = 0
        _save_counts(0, 0)
        print("  🕛 Midnight auto-reset triggered")


threading.Thread(target=_midnight_reset_scheduler, daemon=True).start()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_video_live(video_path, model_path, roi_config_file=None, conf_threshold=0.8):
    """Process video and generate frames for live streaming"""
    global current_frame, processing_active, processing_stats
    
    processing_active = True
    processing_stats['status'] = 'processing'
    last_alert_time = 0  # Track last alert time locally
    
    try:
        # Initialize detector
        detector = MobileOutDetector(model_path, conf_threshold=conf_threshold)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            processing_stats['status'] = 'error'
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize RTSP latency
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25  # RTSP streams often return 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        processing_stats['total_frames'] = total_frames
        
        # Load ROI configuration
        upper_poly = None
        lower_poly = None
        phone_poly = None
        line_p1 = None  # tilted line endpoints
        line_p2 = None

        if roi_config_file and os.path.exists(roi_config_file):
            with open(roi_config_file, 'r') as f:
                roi_config = json.load(f)
                config_type = roi_config.get('type', 'zones')

                if config_type == 'zones':
                    ub = roi_config.get('upper_box')
                    lb = roi_config.get('lower_box')
                    lp = roi_config.get('line_points')
                    pr = roi_config.get('phone_roi')
                    if ub and lb:
                        upper_poly = np.array(ub, np.int32)
                        lower_poly = np.array(lb, np.int32)
                        print(f"  Upper zone: {len(ub)} points")
                        print(f"  Lower zone: {len(lb)} points")
                    if lp and len(lp) == 2:
                        line_p1 = lp[0]
                        line_p2 = lp[1]
                        print(f"  Line: {line_p1} → {line_p2}")
                    else:
                        # Fallback horizontal line
                        line_p1 = [0, height // 2]
                        line_p2 = [width, height // 2]
                    if pr and len(pr) >= 3:
                        phone_poly = np.array(pr, np.int32)
                        print(f"  Phone zone: {len(pr)} points")

        # Default line if nothing loaded
        if line_p1 is None:
            line_p1 = [0, height // 2]
            line_p2 = [width, height // 2]
            print(f"  Using default horizontal line at y={height // 2}")

        def line_side(cx, cy):
            """Which side of the tilted line is the point on? Uses cross product.
            Returns 'upper' or 'lower' relative to the physical frame (upper = smaller y)."""
            # Normalize direction left-to-right so cross product sign is consistent
            # regardless of which endpoint was stored as p1 vs p2
            p1, p2 = (line_p1, line_p2) if line_p1[0] <= line_p2[0] else (line_p2, line_p1)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            cross = dx * (cy - p1[1]) - dy * (cx - p1[0])
            return 'upper' if cross < 0 else 'lower'

        def is_in_roi(cx, cy):
            """Check if point is in either zone polygon"""
            if upper_poly is not None and lower_poly is not None:
                if cv2.pointPolygonTest(upper_poly, (float(cx), float(cy)), False) >= 0:
                    return True
                if cv2.pointPolygonTest(lower_poly, (float(cx), float(cy)), False) >= 0:
                    return True
                return False
            return True

        def is_in_phone_roi(x1, y1, x2, y2):
            """Check if bbox center is within the phone detection zone. No zone = detect everywhere."""
            if phone_poly is None:
                return True
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return cv2.pointPolygonTest(phone_poly, (float(cx), float(cy)), False) >= 0

        # Initialize tracker
        tracker = CentroidTracker(max_disappeared=30)
        in_count, out_count = _load_counts()
        counted_ids = set()
        person_sides = {}  # Track which side of the LINE each person is on
        current_loop_date = datetime.now().strftime('%Y-%m-%d')

        # Output video writer
        source_label = 'live_' + time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'processed_{source_label}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        # Mobile violation tracking — consecutive frames
        PHONE_CONSEC_THRESHOLD = 5  # consecutive inference frames with phone → violation
        phone_consec_count = 0
        last_alert_time = 0
        gif_pending = None  # {'violation':..., 'pre_frames':[...], 'post_frames':[], 'target':N}

        # Performance: run inference every N frames, reuse last result in between
        DETECT_EVERY = 2
        last_results = None


        # Inference resolution (width); keep aspect ratio
        INFER_WIDTH = 640
        infer_scale = INFER_WIDTH / width if width > INFER_WIDTH else 1.0
        infer_size = (INFER_WIDTH, int(height * infer_scale)) if infer_scale < 1.0 else (width, height)

        processing_active = True
        processing_stats['status'] = 'processing'

        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                if USE_RTSP:
                    print("⚠️ RTSP read failed — attempting reconnect...")
                    cap.release()
                    reconnected = False
                    for attempt in range(10):
                        if not processing_active:
                            break
                        time.sleep(3)
                        cap = cv2.VideoCapture(str(video_path))
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if cap.isOpened():
                            print(f"✅ RTSP reconnected (attempt {attempt + 1})")
                            reconnected = True
                            break
                        print(f"  Reconnect {attempt + 1}/10 failed, retrying...")
                    if not reconnected:
                        break
                    continue
                break

            frame_count += 1
            processing_stats['frame_count'] = frame_count

            with frame_buffer_lock:
                frame_buffer.append(frame.copy())

            # Collect post-detection frames for deferred GIF
            if gif_pending is not None:
                gif_pending['post_frames'].append(frame.copy())
                if len(gif_pending['post_frames']) >= gif_pending['target']:
                    try:
                        from PIL import Image as PILImage
                        all_frames = gif_pending['pre_frames'] + gif_pending['post_frames']
                        sampled = all_frames[::2]
                        pf = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in sampled]
                        gif_filename = gif_pending['violation']['filename'].replace('.jpg', '.gif')
                        gif_path = os.path.join('violations', gif_filename)
                        pf[0].save(gif_path, save_all=True, append_images=pf[1:],
                                   duration=130, loop=0, optimize=False)
                        gif_pending['violation']['gif'] = gif_filename
                        print(f"📱 GIF saved: {gif_filename}")
                    except Exception as e:
                        print(f"⚠️ GIF save failed: {e}")
                    gif_pending = None

            # Run detection on every DETECT_EVERY frame, reuse otherwise
            if frame_count % DETECT_EVERY == 0 or last_results is None:
                small = cv2.resize(frame, infer_size) if infer_scale < 1.0 else frame
                results = detector.model(
                    small,
                    conf=conf_threshold,
                    iou=0.45,
                    verbose=False,
                    imgsz=INFER_WIDTH,
                    device=detector.device
                )[0]
                last_results = results
            else:
                results = last_results
            
            # Collect detections
            detections_for_tracking = []
            mobile_detected_this_frame = False
            phone_boxes = []

            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = detector.model.names.get(class_id, f'class_{class_id}')

                if class_name == 'person_with_phone':
                    mobile_detected_this_frame = True
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)
                    phone_boxes.append((bbox, float(box.conf[0])))
                elif class_name in ('person', 'person_without_phone'):
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)

            # Scale bounding boxes back to original resolution so ROI/line coords match
            if infer_scale < 1.0 and detections_for_tracking:
                detections_for_tracking = [
                    [x1 / infer_scale, y1 / infer_scale, x2 / infer_scale, y2 / infer_scale]
                    for x1, y1, x2, y2 in detections_for_tracking
                ]
            if infer_scale < 1.0 and phone_boxes:
                phone_boxes = [
                    ([x1 / infer_scale, y1 / infer_scale, x2 / infer_scale, y2 / infer_scale], c)
                    for (x1, y1, x2, y2), c in phone_boxes
                ]

            # Filter phone detections to phone ROI zone (if configured)
            phone_boxes = [(bbox, c) for bbox, c in phone_boxes if is_in_phone_roi(*bbox)]
            mobile_detected_this_frame = len(phone_boxes) > 0

            # Mobile violation tracking — consecutive frames
            if mobile_detected_this_frame:
                phone_consec_count += 1
            else:
                phone_consec_count = 0

            pending_violation = None
            if phone_consec_count >= PHONE_CONSEC_THRESHOLD:
                current_time = time.time()
                if current_time - last_alert_time >= 120:
                    last_alert_time = current_time
                    phone_consec_count = 0  # reset so next event needs fresh streak
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    pending_violation = {
                        'filename': f'mobile_violation_{timestamp}.jpg',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'date': time.strftime('%Y-%m-%d'),
                        'frame_number': frame_count,
                    }

                    # Audio alert disabled
                    if False and HAS_PYGAME:
                        try:
                            pygame.mixer.music.load(ALERT_SOUND_PATH)
                            pygame.mixer.music.play()
                        except Exception as e:
                            print(f"⚠️ Audio alert failed: {e}")
            
            # Update tracker
            objects = tracker.update(detections_for_tracking)
            
            # LINE-BASED counting (cross-product determines side)
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                # Only count people inside ROI zones
                if not is_in_roi(cx, cy):
                    continue
                
                current_side = line_side(cx, cy)
                
                # Initialize for new objects
                if object_id not in person_sides:
                    person_sides[object_id] = current_side
                    continue
                
                prev_side = person_sides[object_id]
                
                # Count when crossing the line
                if object_id not in counted_ids and prev_side != current_side:
                    if prev_side == 'upper' and current_side == 'lower':
                        in_count += 1
                        counted_ids.add(object_id)
                        _save_counts(in_count, out_count)
                        print(f"  ✅ IN+1: ID {object_id} crossed line at ({cx},{cy})")
                    elif prev_side == 'lower' and current_side == 'upper':
                        out_count += 1
                        counted_ids.add(object_id)
                        _save_counts(in_count, out_count)
                        print(f"  ✅ OUT+1: ID {object_id} crossed line at ({cx},{cy})")

                person_sides[object_id] = current_side

            # Clean up
            active_ids = set(objects.keys())
            for old_id in list(person_sides.keys()):
                if old_id not in active_ids:
                    del person_sides[old_id]

            # Auto-reset at midnight by detecting date change
            today_str = datetime.now().strftime('%Y-%m-%d')
            if today_str != current_loop_date:
                current_loop_date = today_str
                in_count = 0
                out_count = 0
                counted_ids = set()
                person_sides = {}
                _save_counts(0, 0)
                processing_stats['reset_requested'] = False
                print(f"  🕛 Midnight auto-reset triggered (date changed to {today_str})")

            # Handle reset request from UI
            elif processing_stats.get('reset_requested'):
                in_count = 0
                out_count = 0
                counted_ids = set()
                person_sides = {}
                _save_counts(0, 0)
                processing_stats['reset_requested'] = False

            processing_stats['in_count'] = in_count
            processing_stats['out_count'] = out_count
            processing_stats['current_heads'] = len(objects)
            
            # Annotate frame (scale back to original resolution if inference was downscaled)
            annotated = results.plot()
            if infer_scale < 1.0:
                annotated = cv2.resize(annotated, (width, height))
            
            # Draw upper zone (green polygon)
            if upper_poly is not None:
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [upper_poly.reshape((-1, 1, 2))], (0, 255, 0))
                cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
                cv2.polylines(annotated, [upper_poly.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            
            # Draw lower zone (red polygon)
            if lower_poly is not None:
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [lower_poly.reshape((-1, 1, 2))], (0, 0, 255))
                cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
                cv2.polylines(annotated, [lower_poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2)
            
            # Draw counting line (tilted)
            cv2.line(annotated, tuple(line_p1), tuple(line_p2), (0, 255, 255), 3)
            cv2.circle(annotated, tuple(line_p1), 6, (0, 255, 255), -1)
            cv2.circle(annotated, tuple(line_p2), 6, (0, 255, 255), -1)
            
            # Draw tracked objects with line-side info
            for object_id, centroid in objects.items():
                cx, cy = centroid
                side = person_sides.get(object_id, '?')
                color = (0, 255, 0) if side == 'upper' else (0, 0, 255) if side == 'lower' else (200, 200, 200)
                cv2.circle(annotated, (cx, cy), 5, color, -1)
                cv2.putText(annotated, f"ID:{object_id}", (cx - 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save violation screenshot with only person_with_phone bounding boxes
            if pending_violation:
                vpath = os.path.join('violations', pending_violation['filename'])
                vframe = frame.copy()
                for (x1, y1, x2, y2), conf_val in phone_boxes:
                    cv2.rectangle(vframe, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.imwrite(vpath, vframe)
                pending_violation['path'] = vpath

                # Snapshot pre-detection frames; post-frames collected in per-frame loop above
                if gif_pending is None:
                    with frame_buffer_lock:
                        pre_frames = list(frame_buffer)[-100:]  # last ~4 sec at 25fps
                    gif_pending = {
                        'violation': pending_violation,
                        'pre_frames': pre_frames,
                        'post_frames': [],
                        'target': 100,  # ~4 sec post at 25fps
                    }

                global violations_list
                violations_list.append(pending_violation)
                print(f"📱 Mobile violation! Screenshot: {pending_violation['filename']}")

            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            processing_stats['fps'] = current_fps
            
            # Update current frame for streaming
            with frame_lock:
                current_frame = annotated.copy()
            
            # Write to output video
            out.write(annotated)
            
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_stats['status'] = 'completed'
        
    except Exception as e:
        import traceback
        print(f"Error processing video: {e}")
        traceback.print_exc()
        processing_stats['status'] = 'error'
    
    finally:
        processing_active = False


def generate_frames():
    """Generator function for streaming frames"""
    global current_frame
    
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                # Create a blank frame with message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for video...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS streaming


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')




def get_video_source():
    """Determine video source: RTSP if enabled and available, else test video"""
    if USE_RTSP:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            cap.release()
            return RTSP_URL
        cap.release()
        print(f"⚠️ RTSP not reachable, falling back to: {TEST_VIDEO}")
    return TEST_VIDEO


@app.route('/start')
def start_stream():
    """Start processing from RTSP or test video"""
    global processing_active
    if processing_active:
        return jsonify({'success': False, 'message': 'Already processing'})
    
    source = get_video_source()
    conf = float(request.args.get('confidence', 0.25))
    
    thread = threading.Thread(
        target=process_video_live,
        args=(source, MODEL_PATH, ROI_CONFIG, conf)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'source': source})


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Get current processing statistics"""
    return jsonify(processing_stats)


@app.route('/stop')
def stop_processing():
    """Stop current processing"""
    global processing_active
    processing_active = False
    return jsonify({'success': True, 'message': 'Processing stopped'})


@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    """Reset IN/OUT counts to zero in DB and memory"""
    processing_stats['reset_requested'] = True
    processing_stats['in_count'] = 0
    processing_stats['out_count'] = 0
    _save_counts(0, 0)
    return jsonify({'success': True, 'message': 'Counts reset to 0'})


@app.route('/outputs/<filename>')
def download_file(filename):
    """Download processed video"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/report/<date_str>')
def client_report(date_str):
    """Render a print-ready client report for the given date"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD", 400

    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT ON (hour) hour, in_count, out_count, violations
                    FROM hourly_logs WHERE logged_at::date = %s
                    ORDER BY hour, logged_at DESC
                """, (target_date,))
                hourly_rows = cur.fetchall()
    except Exception as e:
        return f"Database error: {e}", 500

    total_in = max((r['in_count'] for r in hourly_rows), default=0)
    total_out = max((r['out_count'] for r in hourly_rows), default=0)
    net = max(total_in - total_out, 0)

    hourly = [{'label': f"{r['hour']}:00", 'in_count': r['in_count'],
               'out_count': r['out_count'], 'violations': r['violations']}
              for r in hourly_rows]

    viols = [v for v in violations_list if v.get('date') == date_str]
    total_viol = len(viols)  # count from actual files, not DB snapshots

    return render_template('client_report.html',
        date=date_str,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        total_in=total_in,
        total_out=total_out,
        net=net,
        total_viol=total_viol,
        hourly=hourly,
        hourly_json=json.dumps(hourly),
        violations=viols,
    )


@app.route('/violations')
def get_violations():
    """Get list of mobile violations, optionally filtered by ?date=YYYY-MM-DD"""
    date_filter = request.args.get('date')
    if date_filter:
        filtered = [v for v in violations_list if v.get('date') == date_filter]
    else:
        filtered = violations_list
    return jsonify({
        'violations': filtered,
        'total': len(filtered)
    })


@app.route('/violations/<filename>')
def get_violation_image(filename):
    """Serve violation screenshot"""
    return send_from_directory('violations', filename)


@app.route('/api/roi/frame')
def get_roi_frame():
    """Return current camera frame as JPEG for ROI drawing"""
    import io
    with frame_lock:
        frame = current_frame.copy() if current_frame is not None else None
    if frame is None:
        return jsonify({'error': 'No frame available — start the stream first'}), 404
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return jsonify({'error': 'Could not encode frame'}), 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')


@app.route('/api/roi', methods=['GET'])
def get_roi_config():
    """Return current ROI config"""
    try:
        if os.path.exists(ROI_CONFIG):
            with open(ROI_CONFIG, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/roi', methods=['POST'])
def save_roi_config():
    """Save new ROI config and stop processing so it restarts with new config"""
    global processing_active
    try:
        data = request.get_json()
        with open(ROI_CONFIG, 'w') as f:
            json.dump(data, f, indent=2)
        if processing_active:
            processing_active = False
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/violations/<filename>', methods=['DELETE'])
def delete_violation(filename):
    """Delete a violation image and remove it from the in-memory list"""
    global violations_list
    safe_name = os.path.basename(filename)
    file_path = os.path.join('violations', safe_name)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        violations_list = [v for v in violations_list if v.get('filename') != safe_name]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reports/<date_str>')
def get_daily_report(date_str):
    """Get hourly report data for a given date (YYYY-MM-DD)"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT hour, in_count, out_count, violations
                    FROM hourly_logs
                    WHERE logged_at::date = %s
                    ORDER BY hour
                """, (target_date,))
                rows = cur.fetchall()

                # Build full 24-hour data (fill missing hours with 0)
                hourly = {}
                for r in rows:
                    hourly[r['hour']] = r

                result = []
                for h in range(24):
                    if h in hourly:
                        result.append({
                            'hour': h,
                            'label': f"{h}:00",
                            'in_count': hourly[h]['in_count'],
                            'out_count': hourly[h]['out_count'],
                            'violations': hourly[h]['violations']
                        })
                    else:
                        result.append({'hour': h, 'label': f"{h}:00", 'in_count': 0, 'out_count': 0, 'violations': 0})

                # Summary
                total_in = max((r['in_count'] for r in rows), default=0) if rows else 0
                total_out = max((r['out_count'] for r in rows), default=0) if rows else 0
                total_viol = max((r['violations'] for r in rows), default=0) if rows else 0

                return jsonify({
                    'date': date_str,
                    'hourly': result,
                    'summary': {
                        'total_in': total_in,
                        'total_out': total_out,
                        'total_violations': total_viol,
                        'peak_hour': max(result, key=lambda x: x['in_count'])['label'] if result else '--'
                    }
                })
    except Exception as e:
        print(f"⚠️ Report query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/week')
def get_weekly_report():
    """Get daily totals for the last 7 days"""
    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT logged_at::date as day,
                           MAX(in_count) as total_in,
                           MAX(out_count) as total_out,
                           MAX(violations) as total_violations
                    FROM hourly_logs
                    WHERE logged_at >= NOW() - INTERVAL '7 days'
                    GROUP BY logged_at::date
                    ORDER BY day
                """)
                rows = cur.fetchall()
                result = []
                for r in rows:
                    result.append({
                        'date': r['day'].strftime('%Y-%m-%d'),
                        'day_name': r['day'].strftime('%a'),
                        'total_in': r['total_in'],
                        'total_out': r['total_out'],
                        'total_violations': r['total_violations']
                    })
                return jsonify({'days': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/weekly_pdf')
def weekly_pdf():
    """Generate a PDF weekly report with summary table and violation screenshots"""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    W, H = A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle('title',  fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#1E293B'), spaceAfter=2)
    sub_style    = ParagraphStyle('sub',    fontName='Helvetica',      fontSize=10, textColor=colors.HexColor('#6B7280'), spaceAfter=12)
    section_style= ParagraphStyle('section',fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#2563EB'), spaceBefore=14, spaceAfter=6)
    cell_style   = ParagraphStyle('cell',   fontName='Helvetica',      fontSize=9,  textColor=colors.HexColor('#1E293B'))
    caption_style= ParagraphStyle('cap',    fontName='Helvetica',      fontSize=8,  textColor=colors.HexColor('#6B7280'), alignment=TA_CENTER, spaceBefore=3)

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    logo_maruti = os.path.join('static', 'Maruti.png')
    logo_sakshi = os.path.join('static', 'Sakshi.png')
    logo_cells = []
    if os.path.exists(logo_maruti):
        logo_cells.append(RLImage(logo_maruti, width=40*mm, height=14*mm))
    if os.path.exists(logo_sakshi):
        logo_cells.append(RLImage(logo_sakshi, width=36*mm, height=14*mm))
    if logo_cells:
        logo_table = Table([logo_cells], colWidths=[50*mm] * len(logo_cells))
        logo_table.setStyle(TableStyle([
            ('ALIGN',  (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(logo_table)
        story.append(Spacer(1, 4*mm))

    today = datetime.now()
    week_start = (today - timedelta(days=6)).strftime('%d %b %Y')
    week_end   = today.strftime('%d %b %Y')

    story.append(Paragraph('Weekly Monitoring Report', title_style))
    story.append(Paragraph(f'Period: {week_start} — {week_end}  &nbsp;|&nbsp;  Generated: {today.strftime("%d %b %Y, %I:%M %p")}', sub_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#E5E7EB'), spaceAfter=10))

    # ── Weekly summary table ─────────────────────────────────────────────────
    story.append(Paragraph('Weekly Summary', section_style))

    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT logged_at::date as day,
                           MAX(in_count) as total_in,
                           MAX(out_count) as total_out,
                           MAX(violations) as total_violations
                    FROM hourly_logs
                    WHERE logged_at >= NOW() - INTERVAL '7 days'
                    GROUP BY logged_at::date
                    ORDER BY day
                """)
                db_days = cur.fetchall()
    except Exception:
        db_days = []

    tdata = [['Date', 'Day', 'Total IN', 'Total OUT', 'Net Inside', 'Violations']]
    sum_in = sum_out = sum_viol = 0
    for r in db_days:
        net = max(r['total_in'] - r['total_out'], 0)
        tdata.append([r['day'].strftime('%Y-%m-%d'), r['day'].strftime('%A'),
                      r['total_in'], r['total_out'], net, r['total_violations']])
        sum_in   += r['total_in']
        sum_out  += r['total_out']
        sum_viol += r['total_violations']
    tdata.append(['TOTAL', '', sum_in, sum_out, max(sum_in - sum_out, 0), sum_viol])

    col_w = [30*mm, 28*mm, 26*mm, 26*mm, 26*mm, 26*mm]
    t = Table(tdata, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#2563EB')),
        ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,0),  9),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-2), [colors.HexColor('#F8FAFC'), colors.white]),
        ('FONTNAME',     (0,1), (-1,-2), 'Helvetica'),
        ('FONTSIZE',     (0,1), (-1,-2), 9),
        ('BACKGROUND',   (0,-1),(-1,-1), colors.HexColor('#F1F5F9')),
        ('FONTNAME',     (0,-1),(-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,-1),(-1,-1), 9),
        ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#E5E7EB')),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]))
    story.append(t)

    # ── Charts ───────────────────────────────────────────────────────────────
    if db_days:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        labels     = [r['day'].strftime('%a\n%d %b') for r in db_days]
        in_vals    = [r['total_in']         for r in db_days]
        out_vals   = [r['total_out']        for r in db_days]
        viol_vals  = [r['total_violations'] for r in db_days]
        x = np.arange(len(labels))

        CHART_W, CHART_H = 7.2, 2.8   # inches — fits A4 content width

        def chart_to_image(fig):
            cb = io.BytesIO()
            fig.savefig(cb, format='png', dpi=130, bbox_inches='tight')
            plt.close(fig)
            cb.seek(0)
            return cb

        story.append(Spacer(1, 5*mm))
        story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#E5E7EB')))
        story.append(Paragraph('Weekly Traffic Overview', section_style))

        # Chart 1 — Daily IN vs OUT grouped bar
        fig1, ax1 = plt.subplots(figsize=(CHART_W, CHART_H))
        w = 0.35
        bars_in  = ax1.bar(x - w/2, in_vals,  w, color='#2563EB', label='IN',  zorder=3)
        bars_out = ax1.bar(x + w/2, out_vals, w, color='#EF4444', label='OUT', zorder=3)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_ylabel('People', fontsize=8); ax1.tick_params(axis='y', labelsize=8)
        ax1.set_title('Daily People IN vs OUT', fontsize=10, fontweight='bold', pad=8)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax1.set_axisbelow(True); ax1.spines[['top','right']].set_visible(False)
        for bar in list(bars_in) + list(bars_out):
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                         ha='center', va='bottom', fontsize=7)
        fig1.patch.set_facecolor('white')
        cb1 = chart_to_image(fig1)
        story.append(RLImage(cb1, width=162*mm, height=63*mm))

        story.append(Spacer(1, 4*mm))
        story.append(Paragraph('Daily Phone Violations', section_style))

        # Chart 2 — Daily violations bar
        fig2, ax2 = plt.subplots(figsize=(CHART_W, CHART_H))
        bars_v = ax2.bar(x, viol_vals, 0.5, color='#EA580C', zorder=3)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel('Violations', fontsize=8); ax2.tick_params(axis='y', labelsize=8)
        ax2.set_title('Phone Violations per Day', fontsize=10, fontweight='bold', pad=8)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax2.set_axisbelow(True); ax2.spines[['top','right']].set_visible(False)
        for bar in bars_v:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                         ha='center', va='bottom', fontsize=7)
        fig2.patch.set_facecolor('white')
        cb2 = chart_to_image(fig2)
        story.append(RLImage(cb2, width=162*mm, height=63*mm))

    # ── Violations section ───────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#E5E7EB')))
    story.append(Paragraph('Phone Violation Evidence', section_style))

    # Collect violations for the last 7 days
    cutoff = (today - timedelta(days=6)).strftime('%Y-%m-%d')
    week_viols = [v for v in violations_list if v.get('date', '') >= cutoff]

    if not week_viols:
        story.append(Paragraph('No violations recorded this week.', sub_style))
    else:
        # 3 images per row
        IMG_W = 54*mm
        IMG_H = 40*mm
        COLS  = 3
        gap   = 3*mm

        for i in range(0, len(week_viols), COLS):
            row_viols = week_viols[i:i+COLS]
            img_cells = []
            for v in row_viols:
                vpath = v.get('path') or os.path.join('violations', v.get('filename', ''))
                if os.path.exists(vpath):
                    try:
                        img = RLImage(vpath, width=IMG_W, height=IMG_H)
                        cap = Paragraph(v.get('timestamp', '--'), caption_style)
                        img_cells.append([img, cap])
                    except Exception:
                        img_cells.append([Paragraph('(image error)', caption_style)])
                else:
                    img_cells.append([Paragraph('(missing)', caption_style)])
            # Pad to COLS
            while len(img_cells) < COLS:
                img_cells.append([''])

            row_table = Table(
                [[cell[0] for cell in img_cells],
                 [cell[1] if len(cell) > 1 else '' for cell in img_cells]],
                colWidths=[IMG_W + gap] * COLS
            )
            row_table.setStyle(TableStyle([
                ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(row_table)
            story.append(Spacer(1, 3*mm))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E5E7EB')))
    story.append(Paragraph('Powered by SakshiAI &nbsp;·&nbsp; Thinkneuralai',
                            ParagraphStyle('footer', fontName='Helvetica', fontSize=8,
                                           textColor=colors.HexColor('#9CA3AF'), alignment=TA_CENTER, spaceBefore=6)))

    doc.build(story)
    buf.seek(0)
    filename = f'Weekly_Report_{today.strftime("%Y-%m-%d")}.pdf'
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name=filename)


def _log_hourly_snapshot():
    """Background task: log current counts to hourly_logs every 10 minutes"""
    while True:
        time.sleep(600)  # Every 10 minutes
        try:
            now = datetime.now()
            with _get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO hourly_logs (logged_at, hour, in_count, out_count, violations) VALUES (%s, %s, %s, %s, %s)",
                        (now, now.hour, processing_stats.get('in_count', 0),
                         processing_stats.get('out_count', 0),
                         len([v for v in violations_list if v.get('date') == now.strftime('%Y-%m-%d')]))
                    )
        except Exception as e:
            print(f"⚠️ Hourly log error: {e}")


def auto_start():
    """Auto-start video processing after server starts"""
    time.sleep(2)  # Wait for Flask to fully start
    source = get_video_source()
    print(f"▶️ Auto-starting detection from: {source}")
    process_video_live(source, MODEL_PATH, ROI_CONFIG, 0.15)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SAKSHI AI — Intelligent Security Monitoring")
    print("="*70)
    print(f"📺 Dashboard: http://localhost:5000")
    print(f"📹 RTSP: {RTSP_URL}")
    print(f"🎬 Fallback: {TEST_VIDEO}")
    print(f"🤖 Model: {MODEL_PATH}")
    print("="*70 + "\n")
    
    # Start background hourly logger
    log_thread = threading.Thread(target=_log_hourly_snapshot, daemon=True)
    log_thread.start()
    
    # Auto-start processing in background
    auto_thread = threading.Thread(target=auto_start, daemon=True)
    auto_thread.start()
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
