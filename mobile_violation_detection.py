import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
from ultralytics import YOLO
import pygame
from datetime import datetime


class MobileViolationDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Mobile Phone Violation Detection System")
        self.root.geometry("900x650")
        
        # Paths
        self.videos_folder = "videos"
        self.violations_folder = "violations"
        self.alert_sound = "violation_alert.wav"
        
        # Create folders
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.violations_folder, exist_ok=True)
        
        # Initialize pygame mixer for audio
        self.audio_available = False
        try:
            pygame.mixer.init()
            self.audio_available = True
        except pygame.error as e:
            print(f"Warning: Audio not available ({e}). Detection will work without sound.")
        
        # Load YOLO model (using YOLOv8)
        try:
            self.model = YOLO('yolov8n.pt')  # nano model for speed
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        
        # Violation tracking
        self.violation_count = 0
        self.last_violation_time = None
        self.alert_cooldown = 3  # seconds between alerts
        
        # Setup UI
        self.setup_ui()
        
        # Create alert sound if it doesn't exist
        self.create_alert_sound()
        
    def create_alert_sound(self):
        """Create a simple beep alert sound if not exists"""
        if not os.path.exists(self.alert_sound):
            try:
                import numpy as np
                from scipy.io import wavfile
                
                # Generate a beep sound
                sample_rate = 44100
                duration = 0.5  # seconds
                frequency = 1000  # Hz
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                # Create beep with fade out
                beep = np.sin(2 * np.pi * frequency * t) * (1 - t / duration)
                beep = (beep * 32767).astype(np.int16)
                
                wavfile.write(self.alert_sound, sample_rate, beep)
                print(f"✓ Created alert sound: {self.alert_sound}")
            except Exception as e:
                print(f"Could not create alert sound: {e}")
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Mobile Phone Violation Detection", 
                              font=("Arial", 20, "bold"), pady=20, fg="#d32f2f")
        title_label.pack()
        
        # Source selection frame
        source_frame = tk.LabelFrame(self.root, text="Video Source", 
                                     font=("Arial", 12, "bold"),
                                     pady=10, padx=10)
        source_frame.pack(fill="x", padx=20)
        
        # Radio buttons for source type
        self.source_var = tk.StringVar(value="file")
        
        radio_frame = tk.Frame(source_frame)
        radio_frame.pack(fill="x", pady=5)
        
        tk.Radiobutton(radio_frame, text="Video File", 
                      variable=self.source_var, value="file",
                      font=("Arial", 10), command=self.toggle_source).pack(side="left", padx=10)
        
        tk.Radiobutton(radio_frame, text="Live Webcam", 
                      variable=self.source_var, value="webcam",
                      font=("Arial", 10), command=self.toggle_source).pack(side="left", padx=10)
        
        # Video file selection (shown by default)
        self.file_frame = tk.Frame(source_frame)
        self.file_frame.pack(fill="x", pady=5)
        
        tk.Label(self.file_frame, text="Select Video:", 
                font=("Arial", 10)).pack(side="left", padx=5)
        
        self.video_var = tk.StringVar()
        self.video_dropdown = ttk.Combobox(self.file_frame, 
                                          textvariable=self.video_var,
                                          state="readonly",
                                          width=35)
        self.video_dropdown.pack(side="left", padx=5)
        
        refresh_btn = tk.Button(self.file_frame, text="Refresh", 
                               command=self.refresh_videos,
                               bg="#4CAF50", fg="white", padx=10)
        refresh_btn.pack(side="left", padx=5)
        
        # Webcam selection (hidden by default)
        self.webcam_frame = tk.Frame(source_frame)
        
        tk.Label(self.webcam_frame, text="Camera Index:", 
                font=("Arial", 10)).pack(side="left", padx=5)
        
        self.webcam_var = tk.IntVar(value=0)
        tk.Spinbox(self.webcam_frame, from_=0, to=5,
                  textvariable=self.webcam_var, width=10).pack(side="left", padx=5)
        
        tk.Label(self.webcam_frame, text="(Usually 0 for default camera)", 
                font=("Arial", 9), fg="gray").pack(side="left", padx=5)
        
        # Controls frame
        controls_frame = tk.Frame(self.root, pady=10)
        controls_frame.pack(fill="x", padx=20)
        
        start_btn = tk.Button(controls_frame, text="Start Detection", 
                             command=self.start_detection,
                             bg="#2196F3", fg="white", 
                             font=("Arial", 12, "bold"),
                             padx=20, pady=10)
        start_btn.pack(side="left", padx=5)
        
        # Settings frame
        settings_frame = tk.LabelFrame(self.root, text="Detection Settings", 
                                       font=("Arial", 11, "bold"),
                                       pady=10, padx=10)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame)
        conf_frame.pack(fill="x", pady=5)
        
        tk.Label(conf_frame, text="Confidence Threshold:", 
                font=("Arial", 10)).pack(side="left", padx=5)
        
        self.confidence_var = tk.DoubleVar(value=0.3)
        confidence_scale = tk.Scale(conf_frame, from_=0.1, to=0.9, 
                                   resolution=0.1, orient="horizontal",
                                   variable=self.confidence_var,
                                   length=200)
        confidence_scale.pack(side="left", padx=5)
        
        tk.Label(conf_frame, textvariable=self.confidence_var,
                font=("Arial", 10)).pack(side="left", padx=5)
        
        tk.Label(conf_frame, text="(Lower = More Sensitive)", 
                font=("Arial", 9), fg="gray").pack(side="left", padx=5)
        
        # Alert sound toggle
        alert_frame = tk.Frame(settings_frame)
        alert_frame.pack(fill="x", pady=5)
        
        self.sound_enabled = tk.BooleanVar(value=True)
        sound_check = tk.Checkbutton(alert_frame, text="Enable Audio Alert",
                                    variable=self.sound_enabled,
                                    font=("Arial", 10))
        sound_check.pack(side="left", padx=5)
        
        test_sound_btn = tk.Button(alert_frame, text="Test Alert Sound",
                                  command=self.play_alert,
                                  bg="#FF9800", fg="white", padx=10)
        test_sound_btn.pack(side="left", padx=5)
        
        # Stats frame
        stats_frame = tk.LabelFrame(self.root, text="Detection Statistics", 
                                    font=("Arial", 11, "bold"),
                                    pady=10, padx=10)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        self.violation_label = tk.Label(stats_frame, 
                                       text="Violations Detected: 0",
                                       font=("Arial", 12, "bold"),
                                       fg="#d32f2f")
        self.violation_label.pack(pady=5)
        
        # Info frame
        info_frame = tk.Frame(self.root, pady=10)
        info_frame.pack(fill="both", expand=True, padx=20)
        
        tk.Label(info_frame, text="Instructions:", 
                font=("Arial", 12, "bold")).pack(anchor="w")
        
        instructions = [
            "1. Place your video files in the 'videos' folder",
            "2. Click 'Refresh' to update the video list",
            "3. Adjust confidence threshold (higher = more strict)",
            "4. Select a video from the dropdown",
            "5. Click 'Start Detection' to begin",
            "",
            "Detection Features:",
            "• Detects people using mobile phones/cell phones",
            "• Audio alert when violation is detected",
            "• Real-time violation counter",
            "• Red boxes around detected violations",
            "",
            "Controls during playback:",
            "- Press 'q' to quit",
            "- Press 'p' to pause/resume",
            "- Press 's' to take screenshot"
        ]
        
        for instruction in instructions:
            tk.Label(info_frame, text=instruction, 
                    font=("Arial", 9), anchor="w").pack(anchor="w", pady=1)
        
        # Status label
        self.status_label = tk.Label(self.root, text="", 
                                     font=("Arial", 10), 
                                     fg="blue")
        self.status_label.pack(pady=10)
        
        # Load videos initially
        self.refresh_videos()
    
    def toggle_source(self):
        """Toggle between video file and webcam source"""
        if self.source_var.get() == "file":
            self.file_frame.pack(fill="x", pady=5)
            self.webcam_frame.pack_forget()
        else:
            self.file_frame.pack_forget()
            self.webcam_frame.pack(fill="x", pady=5)
    
    def refresh_videos(self):
        """Refresh the list of available videos"""
        if not os.path.exists(self.videos_folder):
            os.makedirs(self.videos_folder)
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        videos = []
        
        for filename in os.listdir(self.videos_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                videos.append(filename)
        
        self.video_dropdown['values'] = videos
        
        if videos:
            self.video_dropdown.current(0)
            self.status_label.config(text=f"Found {len(videos)} video(s)", fg="green")
        else:
            self.status_label.config(text="No videos found. Add videos to 'videos' folder.", 
                                   fg="orange")
    
    def play_alert(self):
        """Play violation alert sound"""
        if self.sound_enabled.get():
            try:
                if self.audio_available and os.path.exists(self.alert_sound):
                    pygame.mixer.music.load(self.alert_sound)
                    pygame.mixer.music.play()
                else:
                    # Fallback beep
                    print('\a')  # System beep
            except Exception as e:
                print(f"Error playing alert: {e}")
    
    def start_detection(self):
        """Start mobile violation detection on selected source"""
        if self.model is None:
            messagebox.showerror("Model Not Loaded", 
                               "YOLO model failed to load. Please check installation.")
            return
        
        source_type = self.source_var.get()
        
        if source_type == "file":
            selected_video = self.video_var.get()
            
            if not selected_video:
                messagebox.showwarning("No Video Selected", 
                                     "Please select a video from the dropdown.")
                return
            
            video_path = os.path.join(self.videos_folder, selected_video)
            
            if not os.path.exists(video_path):
                messagebox.showerror("Error", "Video file not found!")
                return
            
            source = video_path
            source_name = selected_video
        else:
            # Webcam source
            source = self.webcam_var.get()
            source_name = f"Webcam {source}"
        
        # Reset violation count
        self.violation_count = 0
        self.violation_label.config(text=f"Violations Detected: {self.violation_count}")
        
        # Start detection in a separate thread to keep UI responsive
        detection_thread = threading.Thread(target=self.detect_violations, 
                                           args=(source, source_type, source_name))
        detection_thread.daemon = True
        detection_thread.start()
    
    def detect_violations(self, source, source_type="file", source_name=""):
        """Process video/webcam and detect mobile phone violations"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            return
        
        self.status_label.config(text=f"Detecting on {source_name}... Press 'q' to quit", 
                               fg="blue")
        
        paused = False
        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame for performance
        
        # Classes to detect (cell phone = 67 in COCO dataset)
        cell_phone_class = 67
        
        while True:
            if not paused:
                ret, frame = video_capture.read()
                
                if not ret:
                    if source_type == "webcam":
                        print("Webcam feed interrupted")
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % process_every_n_frames == 0:
                    # Run YOLO detection
                    results = self.model(frame, conf=self.confidence_var.get(), 
                                       verbose=False)
                    
                    violation_detected = False
                    
                    # Process detections
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Check if cell phone detected
                            if cls == cell_phone_class:
                                violation_detected = True
                                
                                # Get box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Draw red rectangle
                                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                            (0, 0, 255), 3)
                                
                                # Add label
                                label = f"VIOLATION! {conf:.2f}"
                                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), 
                                            (0, 0, 255), -1)
                                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                          (255, 255, 255), 2)
                    
                    # Handle violation alert
                    if violation_detected:
                        current_time = datetime.now()
                        
                        # Check cooldown period
                        if (self.last_violation_time is None or 
                            (current_time - self.last_violation_time).total_seconds() 
                            >= self.alert_cooldown):
                            
                            self.violation_count += 1
                            self.violation_label.config(
                                text=f"Violations Detected: {self.violation_count}")
                            self.last_violation_time = current_time
                            
                            # Play alert sound
                            self.play_alert()
                
                # Add status overlay
                status_text = f"Violations: {self.violation_count} | " \
                             f"Press 'q' to quit | 'p' to pause | 's' for screenshot"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), 
                            (0, 0, 0), -1)
                cv2.putText(frame, status_text, (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Mobile Violation Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(self.violations_folder, 
                                             f"violation_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
        self.status_label.config(
            text=f"Detection completed. Total violations: {self.violation_count}", 
            fg="green")


def main():
    root = tk.Tk()
    app = MobileViolationDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
