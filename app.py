from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import threading
import pygame
import base64

app = Flask(__name__)

# Configuration
VIDEOS_FOLDER = 'videos'
KNOWN_FACES_FOLDER = 'known_faces'
VIOLATIONS_FOLDER = 'violations'
ALERT_SOUND = 'violation_alert.wav'

os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs(VIOLATIONS_FOLDER, exist_ok=True)

# Global state
class AppState:
    def __init__(self):
        self.face_detection_active = False
        self.violation_detection_active = False
        self.face_camera = None
        self.violation_camera = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.yolo_model = None
        self.violation_count = 0
        self.last_violation_time = None
        self.alert_cooldown = 3
        self.face_stats = {'in_count': 0, 'out_count': 0, 'net_count': 0}
        
        # Initialize
        self.load_known_faces()
        self.load_yolo_model()
        
        # Audio
        try:
            pygame.mixer.init()
            self.audio_available = True
        except:
            self.audio_available = False
    
    def load_known_faces(self):
        """Load known faces from folder"""
        if not os.path.exists(KNOWN_FACES_FOLDER):
            return
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(KNOWN_FACES_FOLDER):
            file_path = os.path.join(KNOWN_FACES_FOLDER, filename)
            
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_formats:
                    try:
                        image = face_recognition.load_image_file(file_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            name = os.path.splitext(filename)[0].replace('_', ' ')
                            self.known_face_names.append(name)
                            print(f"✓ Loaded face: {name}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        
        print(f"✓ Total known faces loaded: {len(self.known_face_encodings)}")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
    
    def play_alert(self):
        """Play alert sound"""
        try:
            if self.audio_available and os.path.exists(ALERT_SOUND):
                pygame.mixer.music.load(ALERT_SOUND)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"Alert error: {e}")

state = AppState()

@app.route('/')
def index():
    """Serve dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/videos')
def get_videos():
    """Get list of available videos"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    for filename in os.listdir(VIDEOS_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            videos.append(filename)
    
    return jsonify({'videos': videos})

@app.route('/api/face-detection/start', methods=['POST'])
def start_face_detection():
    """Start face detection"""
    data = request.get_json()
    source_type = data.get('source_type', 'video')
    source = data.get('source')
    
    if state.face_detection_active:
        return jsonify({'success': False, 'error': 'Already running'})
    
    if source_type == 'video':
        video_path = os.path.join(VIDEOS_FOLDER, source)
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video not found'})
        state.face_camera = cv2.VideoCapture(video_path)
    else:
        # Webcam
        camera_index = int(source) if source else 0
        state.face_camera = cv2.VideoCapture(camera_index)
    
    if not state.face_camera.isOpened():
        return jsonify({'success': False, 'error': 'Could not open source'})
    
    state.face_detection_active = True
    return jsonify({'success': True})

@app.route('/api/face-detection/stop', methods=['POST'])
def stop_face_detection():
    """Stop face detection"""
    state.face_detection_active = False
    if state.face_camera:
        state.face_camera.release()
        state.face_camera = None
    return jsonify({'success': True})

@app.route('/api/violation-detection/start', methods=['POST'])
def start_violation_detection():
    """Start violation detection"""
    data = request.get_json()
    source_type = data.get('source_type', 'video')
    source = data.get('source')
    
    if state.violation_detection_active:
        return jsonify({'success': False, 'error': 'Already running'})
    
    if source_type == 'video':
        video_path = os.path.join(VIDEOS_FOLDER, source)
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video not found'})
        state.violation_camera = cv2.VideoCapture(video_path)
    else:
        camera_index = int(source) if source else 0
        state.violation_camera = cv2.VideoCapture(camera_index)
    
    if not state.violation_camera.isOpened():
        return jsonify({'success': False, 'error': 'Could not open source'})
    
    state.violation_detection_active = True
    state.violation_count = 0
    return jsonify({'success': True})

@app.route('/api/violation-detection/stop', methods=['POST'])
def stop_violation_detection():
    """Stop violation detection"""
    state.violation_detection_active = False
    if state.violation_camera:
        state.violation_camera.release()
        state.violation_camera = None
    return jsonify({'success': True})

@app.route('/api/violation-detection/stats')
def get_violation_stats():
    """Get violation statistics"""
    return jsonify({
        'violation_count': state.violation_count
    })

def generate_face_frames():
    """Generate face detection frames"""
    frame_count = 0
    frame_skip = 2
    
    while state.face_detection_active:
        if not state.face_camera:
            break
            
        ret, frame = state.face_camera.read()
        
        if not ret:
            # Loop video
            if state.face_camera:
                state.face_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        if frame_count % frame_skip == 0:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                
                if state.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        state.known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(
                        state.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = state.known_face_names[best_match_index]
                
                face_names.append(name)
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                          cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Add info overlay
        cv2.putText(frame, f"Face Detection Active", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_violation_frames():
    """Generate violation detection frames"""
    frame_count = 0
    process_every_n_frames = 2
    cell_phone_class = 67
    
    while state.violation_detection_active:
        if not state.violation_camera:
            break
            
        ret, frame = state.violation_camera.read()
        
        if not ret:
            # Loop video
            if state.violation_camera:
                state.violation_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        if frame_count % process_every_n_frames == 0 and state.yolo_model:
            # Run YOLO detection
            results = state.yolo_model(frame, conf=0.3, verbose=False)
            
            violation_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == cell_phone_class:
                        violation_detected = True
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        label = f"VIOLATION! {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 255), -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if violation_detected:
                current_time = datetime.now()
                
                if (state.last_violation_time is None or 
                    (current_time - state.last_violation_time).total_seconds() >= state.alert_cooldown):
                    
                    state.violation_count += 1
                    state.last_violation_time = current_time
                    
                    # Play alert in background thread
                    threading.Thread(target=state.play_alert, daemon=True).start()
        
        # Add overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Violations: {state.violation_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/face')
def face_video_feed():
    """Video streaming route for face detection"""
    return Response(generate_face_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/violation')
def violation_video_feed():
    """Video streaming route for violation detection"""
    return Response(generate_violation_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("=" * 60)
    print("🔐 MARUTHI SECURITY DASHBOARD")
    print("=" * 60)
    print(f"✓ Known faces loaded: {len(state.known_face_encodings)}")
    print(f"✓ YOLO model: {'Ready' if state.yolo_model else 'Not loaded'}")
    print(f"✓ Audio alerts: {'Enabled' if state.audio_available else 'Disabled'}")
    print("=" * 60)
    print("🌐 Starting Flask server...")
    print("📱 Open browser: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
