"""
Flask Web Application for Live Object Detection with ROI Counting
Upload video and watch live detection with IN/OUT counting
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for live streaming
current_frame = None
processing_active = False
processing_stats = {
    'frame_count': 0,
    'total_frames': 0,
    'in_count': 0,
    'out_count': 0,
    'fps': 0,
    'status': 'idle'
}
frame_lock = threading.Lock()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_video_live(video_path, model_path, roi_config_file=None, conf_threshold=0.25):
    """Process video and generate frames for live streaming"""
    global current_frame, processing_active, processing_stats
    
    processing_active = True
    processing_stats['status'] = 'processing'
    
    try:
        # Initialize detector
        detector = MobileOutDetector(model_path, conf_threshold=conf_threshold)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            processing_stats['status'] = 'error'
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        processing_stats['total_frames'] = total_frames
        
        # Load ROI configuration
        roi_line = None
        is_custom_line = False
        is_horizontal = False
        line_pos = None
        line_p1, line_p2 = None, None
        
        if roi_config_file and os.path.exists(roi_config_file):
            with open(roi_config_file, 'r') as f:
                roi_config = json.load(f)
                config_type = roi_config.get('type', 'custom')
                
                if config_type == 'horizontal':
                    roi_line = {'y': roi_config.get('y')}
                elif config_type == 'vertical':
                    roi_line = {'x': roi_config.get('x')}
                elif config_type == 'custom' or 'line_points' in roi_config:
                    line_points = roi_config.get('line_points')
                    if line_points:
                        roi_line = {'line_points': line_points}
        
        # Default ROI line if not configured
        if roi_line is None:
            roi_line = {'y': height // 2}
        
        # Determine line type
        if 'line_points' in roi_line:
            is_custom_line = True
            line_p1, line_p2 = roi_line['line_points']
        else:
            is_horizontal = 'y' in roi_line
            line_pos = roi_line.get('y') if is_horizontal else roi_line.get('x')
        
        # Initialize tracker
        tracker = CentroidTracker(max_disappeared=30)
        in_count = 0
        out_count = 0
        counted_ids = set()
        
        # Output video writer
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            processing_stats['frame_count'] = frame_count
            
            # Run detection (NO FRAME SKIPPING - process every frame)
            results = detector.model(
                frame,
                conf=conf_threshold,
                iou=0.45,
                verbose=False
            )[0]
            
            # Collect detections for tracking
            detections_for_tracking = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = detector.class_names[class_id]
                
                if class_name == 'OUT':
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)
            
            # Update tracker
            objects = tracker.update(detections_for_tracking)
            
            # Check line crossings
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                if object_id not in tracker.crossed:
                    tracker.crossed[object_id] = {'crossed': False, 'direction': None, 'start_side': None}
                
                # Determine side
                if is_custom_line:
                    v1 = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
                    v2 = (cx - line_p1[0], cy - line_p1[1])
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    current_side = 'left' if cross > 0 else 'right'
                elif is_horizontal:
                    current_side = 'top' if cy < line_pos else 'bottom'
                else:
                    current_side = 'left' if cx < line_pos else 'right'
                
                if tracker.crossed[object_id]['start_side'] is None:
                    tracker.crossed[object_id]['start_side'] = current_side
                
                # Detect crossing
                if object_id not in counted_ids:
                    start_side = tracker.crossed[object_id]['start_side']
                    
                    if is_custom_line or not is_horizontal:
                        if start_side == 'left' and current_side == 'right':
                            out_count += 1
                            counted_ids.add(object_id)
                        elif start_side == 'right' and current_side == 'left':
                            in_count += 1
                            counted_ids.add(object_id)
                    else:
                        if start_side == 'top' and current_side == 'bottom':
                            out_count += 1
                            counted_ids.add(object_id)
                        elif start_side == 'bottom' and current_side == 'top':
                            in_count += 1
                            counted_ids.add(object_id)
            
            processing_stats['in_count'] = in_count
            processing_stats['out_count'] = out_count
            
            # Annotate frame
            annotated = results.plot()
            
            # Draw ROI line
            if is_custom_line:
                cv2.line(annotated, tuple(line_p1), tuple(line_p2), (0, 255, 255), 3)
                cv2.circle(annotated, tuple(line_p1), 8, (0, 255, 0), -1)
                cv2.circle(annotated, tuple(line_p2), 8, (0, 0, 255), -1)
            elif is_horizontal:
                cv2.line(annotated, (0, line_pos), (width, line_pos), (0, 255, 255), 3)
            else:
                cv2.line(annotated, (line_pos, 0), (line_pos, height), (0, 255, 255), 3)
            
            # Draw tracked objects
            for object_id, centroid in objects.items():
                cx, cy = centroid
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(annotated, f"ID:{object_id}", (cx - 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add stats overlay
            cv2.rectangle(annotated, (10, 10), (700, 50), (0, 0, 0), -1)
            text = f"Frame {frame_count}/{total_frames} | IN: {in_count} | OUT: {out_count}"
            cv2.putText(annotated, text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            processing_stats['fps'] = current_fps
            
            # Update current frame for streaming
            with frame_lock:
                current_frame = annotated.copy()
            
            # Write to output video
            out.write(annotated)
            
            # Small delay to control streaming speed
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_stats['status'] = 'completed'
        
    except Exception as e:
        print(f"Error processing video: {e}")
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


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    global processing_active, current_frame, processing_stats
    
    if processing_active:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Get parameters
    conf_threshold = float(request.form.get('confidence', 0.25))
    roi_config = request.form.get('roi_config', 'roi_config.json')
    
    # Reset stats
    current_frame = None
    processing_stats = {
        'frame_count': 0,
        'total_frames': 0,
        'in_count': 0,
        'out_count': 0,
        'fps': 0,
        'status': 'processing'
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_live,
        args=(video_path, 'weights/best.pt', roi_config, conf_threshold)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Video uploaded and processing started',
        'filename': filename
    })


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


@app.route('/outputs/<filename>')
def download_file(filename):
    """Download processed video"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 LIVE DETECTION WEB APP")
    print("="*70)
    print("📺 Open your browser and go to: http://localhost:5000")
    print("📤 Upload a video and watch LIVE detection!")
    print("🎯 All frames processed - No frame skipping")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
