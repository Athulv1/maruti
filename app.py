"""
Flask Web Application for Live Object Detection with ROI Counting
RTSP stream or local video with live detection + IN/OUT counting
"""

from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import os
import json
import threading
import time
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

def _load_counts():
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT in_count, out_count FROM people_counts WHERE id = 1")
                row = cur.fetchone()
                return (row[0], row[1]) if row else (0, 0)
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
RTSP_URL = 'rtsp://admin:Thinkneural%2312@192.168.150.10:554/Streaming/Channels/101'
TEST_VIDEO = 'videos/test.mp4'
MODEL_PATH = 'best.pt'
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
    'status': 'idle'
}
violations_list = []  # Store mobile violation screenshots
frame_lock = threading.Lock()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_video_live(video_path, model_path, roi_config_file=None, conf_threshold=0.25):
    """Process video and generate frames for live streaming"""
    global current_frame, processing_active, processing_stats
    
    processing_active = True
    processing_stats['status'] = 'processing'
    last_alert_time = 0  # Track last alert time locally
    mobile_detection_frames = 0  # Track consecutive mobile detections
    MOBILE_FRAME_THRESHOLD = 2  # Minimum frames needed to trigger alert
    mobile_detection_frames = 0  # Track consecutive mobile detections
    MOBILE_FRAME_THRESHOLD = 2  # Minimum frames needed to trigger alert
    
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

        # Default line if nothing loaded
        if line_p1 is None:
            line_p1 = [0, height // 2]
            line_p2 = [width, height // 2]
            print(f"  Using default horizontal line at y={height // 2}")

        def line_side(cx, cy):
            """Which side of the tilted line is the point on? Uses cross product.
            Returns 'upper' or 'lower' relative to the line direction."""
            dx = line_p2[0] - line_p1[0]
            dy = line_p2[1] - line_p1[1]
            cross = dx * (cy - line_p1[1]) - dy * (cx - line_p1[0])
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

        # Initialize tracker
        tracker = CentroidTracker(max_disappeared=30)
        in_count, out_count = _load_counts()
        counted_ids = set()
        person_sides = {}  # Track which side of the LINE each person is on

        # Output video writer
        source_label = 'live_' + time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'processed_{source_label}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        # Mobile violation tracking
        mobile_detection_frames = 0
        MOBILE_FRAME_THRESHOLD = 3
        last_alert_time = 0

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
                break

            frame_count += 1
            processing_stats['frame_count'] = frame_count

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

            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = detector.model.names.get(class_id, f'class_{class_id}')

                if class_name == 'person_with_phone':
                    mobile_detected_this_frame = True
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)
                elif class_name in ('person', 'person_without_phone'):
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)

            # Scale bounding boxes back to original resolution so ROI/line coords match
            if infer_scale < 1.0 and detections_for_tracking:
                detections_for_tracking = [
                    [x1 / infer_scale, y1 / infer_scale, x2 / infer_scale, y2 / infer_scale]
                    for x1, y1, x2, y2 in detections_for_tracking
                ]
            
            # Mobile violation tracking
            if mobile_detected_this_frame:
                mobile_detection_frames += 1
            else:
                mobile_detection_frames = 0
            
            if mobile_detection_frames >= MOBILE_FRAME_THRESHOLD:
                current_time = time.time()
                if current_time - last_alert_time >= 5:
                    try:
                        pygame.mixer.music.load(ALERT_SOUND_PATH)
                        pygame.mixer.music.play()
                        last_alert_time = current_time
                        
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        violation_filename = f'mobile_violation_{timestamp}.jpg'
                        violation_path = os.path.join('violations', violation_filename)
                        cv2.imwrite(violation_path, frame)
                        
                        global violations_list
                        violations_list.append({
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'frame_number': frame_count,
                            'filename': violation_filename,
                            'path': violation_path
                        })
                        print(f"🔊 Mobile violation alert! Screenshot: {violation_filename}")
                    except Exception as e:
                        print(f"Error playing alert: {e}")
            
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
                        out_count += 1
                        counted_ids.add(object_id)
                        _save_counts(in_count, out_count)
                        print(f"  ✅ OUT+1: ID {object_id} crossed line at ({cx},{cy})")
                    elif prev_side == 'lower' and current_side == 'upper':
                        in_count += 1
                        counted_ids.add(object_id)
                        _save_counts(in_count, out_count)
                        print(f"  ✅ IN+1: ID {object_id} crossed line at ({cx},{cy})")
                
                person_sides[object_id] = current_side
            
            # Clean up
            active_ids = set(objects.keys())
            for old_id in list(person_sides.keys()):
                if old_id not in active_ids:
                    del person_sides[old_id]
            
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
            
            # Add stats overlay
            cv2.rectangle(annotated, (10, 10), (400, 55), (0, 0, 0), -1)
            text = f"IN: {in_count} | OUT: {out_count} | Heads: {len(objects)}"
            cv2.putText(annotated, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
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
    _save_counts(0, 0)
    processing_stats['in_count'] = 0
    processing_stats['out_count'] = 0
    return jsonify({'success': True, 'message': 'Counts reset to 0'})


@app.route('/outputs/<filename>')
def download_file(filename):
    """Download processed video"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/violations')
def get_violations():
    """Get list of mobile violations"""
    return jsonify({
        'violations': violations_list,
        'total': len(violations_list)
    })


@app.route('/violations/<filename>')
def get_violation_image(filename):
    """Serve violation screenshot"""
    return send_from_directory('violations', filename)


def auto_start():
    """Auto-start video processing after server starts"""
    time.sleep(2)  # Wait for Flask to fully start
    source = get_video_source()
    print(f"▶️ Auto-starting detection from: {source}")
    process_video_live(source, MODEL_PATH, ROI_CONFIG, 0.25)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SAKSHI AI — Intelligent Security Monitoring")
    print("="*70)
    print(f"📺 Dashboard: http://localhost:5000")
    print(f"📹 RTSP: {RTSP_URL}")
    print(f"🎬 Fallback: {TEST_VIDEO}")
    print(f"🤖 Model: {MODEL_PATH}")
    print("="*70 + "\n")
    
    # Auto-start processing in background
    auto_thread = threading.Thread(target=auto_start, daemon=True)
    auto_thread.start()
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
