import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import pygame
from datetime import datetime
from scipy.io import wavfile

class PremiumDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🔐 Maruthi Security Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0e27')
        
        # Color scheme - Premium dark theme with gradients
        self.colors = {
            'bg_dark': '#0a0e27',
            'bg_card': '#1a1f3a',
            'accent_blue': '#3b82f6',
            'accent_purple': '#8b5cf6',
            'accent_green': '#10b981',
            'accent_red': '#ef4444',
            'text_primary': '#ffffff',
            'text_secondary': '#94a3b8',
            'border': '#2d3748'
        }
        
        # Initialize application states
        self.current_tab = None
        self.is_detecting = False
        self.detection_thread = None
        
        # Paths
        self.videos_folder = "videos"
        self.known_faces_folder = "known_faces"
        self.violations_folder = "violations"
        self.alert_sound = "violation_alert.wav"
        
        # Create folders
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.known_faces_folder, exist_ok=True)
        os.makedirs(self.violations_folder, exist_ok=True)
        
        # Face detection data
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # YOLO model for violation detection
        self.yolo_model = None
        self.load_yolo_model()
        
        # Initialize audio
        self.audio_available = False
        try:
            pygame.mixer.init()
            self.audio_available = True
        except pygame.error as e:
            print(f"Warning: Audio not available ({e})")
        
        # Violation tracking
        self.violation_count = 0
        self.last_violation_time = None
        self.alert_cooldown = 3
        
        self.create_header()
        self.create_tab_buttons()
        self.create_main_container()
        self.show_tab("face_detection")
        
        
    def load_known_faces(self):
        """Load known faces from the known_faces folder"""
        if not os.path.exists(self.known_faces_folder):
            return
            
        self.known_face_encodings = []
        self.known_face_names = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(self.known_faces_folder):
            file_path = os.path.join(self.known_faces_folder, filename)
            
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
                            print(f"Loaded face: {name}")
                        else:
                            print(f"No face found in {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        
        if self.known_face_encodings:
            print(f"✓ Loaded {len(self.known_face_encodings)} known faces")
        else:
            print("ℹ No known faces loaded")
    
    def load_yolo_model(self):
        """Load YOLO model for mobile detection"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def play_alert(self):
        """Play violation alert sound"""
        try:
            if self.audio_available and os.path.exists(self.alert_sound):
                pygame.mixer.music.load(self.alert_sound)
                pygame.mixer.music.play()
            else:
                print('\a')  # System beep
        except Exception as e:
            print(f"Error playing alert: {e}")
    
    def create_header(self):
        """Create premium header with gradient effect"""
        header_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], height=100)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title with glow effect
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_dark'])
        title_frame.pack(expand=True)
        
        title_label = tk.Label(
            title_frame,
            text="🔐 MARUTHI SECURITY SYSTEM",
            font=('Helvetica', 32, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="AI-Powered Face Recognition & Violation Detection",
            font=('Helvetica', 12),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack()
        
    def create_tab_buttons(self):
        """Create attractive tab buttons with hover effects"""
        tab_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], height=80)
        tab_frame.pack(fill='x', padx=20, pady=(10, 0))
        tab_frame.pack_propagate(False)
        
        # Center the buttons
        button_container = tk.Frame(tab_frame, bg=self.colors['bg_dark'])
        button_container.pack(expand=True)
        
        # Face Detection Tab Button
        self.face_tab_btn = tk.Button(
            button_container,
            text="👤 FACE DETECTION",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['accent_blue'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['accent_purple'],
            activeforeground=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=40,
            pady=15,
            cursor='hand2',
            command=lambda: self.show_tab("face_detection")
        )
        self.face_tab_btn.pack(side='left', padx=10)
        self.add_hover_effect(self.face_tab_btn, self.colors['accent_blue'], self.colors['accent_purple'])
        
        # Mobile Violation Tab Button
        self.violation_tab_btn = tk.Button(
            button_container,
            text="📱 VIOLATION DETECTION",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary'],
            activebackground=self.colors['accent_purple'],
            activeforeground=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=40,
            pady=15,
            cursor='hand2',
            command=lambda: self.show_tab("violation_detection")
        )
        self.violation_tab_btn.pack(side='left', padx=10)
        self.add_hover_effect(self.violation_tab_btn, self.colors['bg_card'], self.colors['accent_purple'])
        
    def add_hover_effect(self, button, normal_color, hover_color):
        """Add hover effect to buttons"""
        def on_enter(e):
            button['bg'] = hover_color
            button['fg'] = self.colors['text_primary']
            
        def on_leave(e):
            if button == self.face_tab_btn and self.current_tab == "face_detection":
                button['bg'] = self.colors['accent_blue']
            elif button == self.violation_tab_btn and self.current_tab == "violation_detection":
                button['bg'] = self.colors['accent_blue']
            else:
                button['bg'] = normal_color
                button['fg'] = self.colors['text_secondary']
                
        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)
        
    def create_main_container(self):
        """Create main container for tab content"""
        self.main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        self.main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
    def show_tab(self, tab_name):
        """Switch between tabs"""
        # Clear current container
        for widget in self.main_container.winfo_children():
            widget.destroy()
            
        # Update button states
        if tab_name == "face_detection":
            self.face_tab_btn.config(bg=self.colors['accent_blue'], fg=self.colors['text_primary'])
            self.violation_tab_btn.config(bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
            self.create_face_detection_tab()
        else:
            self.violation_tab_btn.config(bg=self.colors['accent_blue'], fg=self.colors['text_primary'])
            self.face_tab_btn.config(bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
            self.create_violation_detection_tab()
            
        self.current_tab = tab_name
        
    def create_face_detection_tab(self):
        """Create Face Detection interface"""
        # Main card
        card = tk.Frame(self.main_container, bg=self.colors['bg_card'], relief='flat', bd=0)
        card.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_card'])
        header.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            header,
            text="👤 Face Recognition System",
            font=('Helvetica', 20, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="Detect and recognize faces from video files or live webcam feed",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w', pady=(5, 0))
        
        # Controls section
        controls_frame = tk.Frame(card, bg=self.colors['bg_card'])
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        # Source selection
        source_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        source_frame.pack(fill='x', pady=10)
        
        tk.Label(
            source_frame,
            text="Select Source:",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 20))
        
        self.face_source_var = tk.StringVar(value="video")
        
        tk.Radiobutton(
            source_frame,
            text="📹 Video File",
            variable=self.face_source_var,
            value="video",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['bg_card'],
            activeforeground=self.colors['text_primary'],
            command=self.face_toggle_source
        ).pack(side='left', padx=10)
        
        tk.Radiobutton(
            source_frame,
            text="📷 Live Webcam",
            variable=self.face_source_var,
            value="webcam",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['bg_card'],
            activeforeground=self.colors['text_primary'],
            command=self.face_toggle_source
        ).pack(side='left', padx=10)
        
        # Video selection
        self.face_video_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        self.face_video_frame.pack(fill='x', pady=10)
        
        tk.Label(
            self.face_video_frame,
            text="Select Video:",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.face_video_var = tk.StringVar()
        video_files = self.get_video_files()
        
        self.face_video_dropdown = ttk.Combobox(
            self.face_video_frame,
            textvariable=self.face_video_var,
            values=video_files,
            state='readonly',
            font=('Helvetica', 11),
            width=40
        )
        self.face_video_dropdown.pack(side='left', padx=10)
        if video_files:
            self.face_video_dropdown.current(0)
            
        # Webcam selection
        self.face_webcam_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        
        tk.Label(
            self.face_webcam_frame,
            text="Camera Index:",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.face_webcam_var = tk.IntVar(value=0)
        tk.Spinbox(
            self.face_webcam_frame,
            from_=0,
            to=5,
            textvariable=self.face_webcam_var,
            font=('Helvetica', 11),
            width=10
        ).pack(side='left', padx=10)
        
        # Action button
        self.face_start_btn = tk.Button(
            controls_frame,
            text="▶ START DETECTION",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['accent_green'],
            fg=self.colors['text_primary'],
            activebackground='#059669',
            activeforeground=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2',
            command=self.start_face_detection
        )
        self.face_start_btn.pack(pady=20)
        self.add_hover_effect(self.face_start_btn, self.colors['accent_green'], '#059669')
        
        # Display area
        display_frame = tk.Frame(card, bg=self.colors['bg_dark'], relief='solid', bd=2)
        display_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.face_display_label = tk.Label(
            display_frame,
            text="📹 Video feed will appear here",
            font=('Helvetica', 14),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        self.face_display_label.pack(expand=True)
        
    def create_violation_detection_tab(self):
        """Create Mobile Violation Detection interface"""
        # Main card
        card = tk.Frame(self.main_container, bg=self.colors['bg_card'], relief='flat', bd=0)
        card.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_card'])
        header.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            header,
            text="📱 Mobile Phone Violation Detection",
            font=('Helvetica', 20, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="AI-powered detection system with voice alerts for mobile phone violations",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w', pady=(5, 0))
        
        # Controls section
        controls_frame = tk.Frame(card, bg=self.colors['bg_card'])
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        # Source selection
        source_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        source_frame.pack(fill='x', pady=10)
        
        tk.Label(
            source_frame,
            text="Select Source:",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 20))
        
        self.violation_source_var = tk.StringVar(value="video")
        
        tk.Radiobutton(
            source_frame,
            text="📹 Video File",
            variable=self.violation_source_var,
            value="video",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['bg_card'],
            activeforeground=self.colors['text_primary'],
            command=self.violation_toggle_source
        ).pack(side='left', padx=10)
        
        tk.Radiobutton(
            source_frame,
            text="📷 Live Webcam",
            variable=self.violation_source_var,
            value="webcam",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['bg_card'],
            activeforeground=self.colors['text_primary'],
            command=self.violation_toggle_source
        ).pack(side='left', padx=10)
        
        # Video selection
        self.violation_video_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        self.violation_video_frame.pack(fill='x', pady=10)
        
        tk.Label(
            self.violation_video_frame,
            text="Select Video:",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_video_var = tk.StringVar()
        video_files = self.get_video_files()
        
        self.violation_video_dropdown = ttk.Combobox(
            self.violation_video_frame,
            textvariable=self.violation_video_var,
            values=video_files,
            state='readonly',
            font=('Helvetica', 11),
            width=40
        )
        self.violation_video_dropdown.pack(side='left', padx=10)
        if video_files:
            self.violation_video_dropdown.current(0)
            
        # Webcam selection
        self.violation_webcam_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        
        tk.Label(
            self.violation_webcam_frame,
            text="Camera Index:",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_webcam_var = tk.IntVar(value=0)
        tk.Spinbox(
            self.violation_webcam_frame,
            from_=0,
            to=5,
            textvariable=self.violation_webcam_var,
            font=('Helvetica', 11),
            width=10
        ).pack(side='left', padx=10)
        
        # Confidence threshold
        threshold_frame = tk.Frame(controls_frame, bg=self.colors['bg_card'])
        threshold_frame.pack(fill='x', pady=10)
        
        tk.Label(
            threshold_frame,
            text="Detection Confidence:",
            font=('Helvetica', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_confidence_var = tk.DoubleVar(value=0.3)
        tk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            variable=self.violation_confidence_var,
            font=('Helvetica', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=300
        ).pack(side='left', padx=10)
        
        # Action button
        self.violation_start_btn = tk.Button(
            controls_frame,
            text="▶ START DETECTION",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['accent_green'],
            fg=self.colors['text_primary'],
            activebackground='#059669',
            activeforeground=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2',
            command=self.start_violation_detection
        )
        self.violation_start_btn.pack(pady=20)
        self.add_hover_effect(self.violation_start_btn, self.colors['accent_green'], '#059669')
        
        # Stats and display area
        bottom_frame = tk.Frame(card, bg=self.colors['bg_card'])
        bottom_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Stats panel
        stats_frame = tk.Frame(bottom_frame, bg=self.colors['bg_dark'], relief='solid', bd=2, width=250)
        stats_frame.pack(side='left', fill='y', padx=(0, 10))
        stats_frame.pack_propagate(False)
        
        tk.Label(
            stats_frame,
            text="📊 STATISTICS",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_primary']
        ).pack(pady=20)
        
        self.violation_count_label = tk.Label(
            stats_frame,
            text="Violations: 0",
            font=('Helvetica', 12),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_red']
        )
        self.violation_count_label.pack(pady=10)
        
        # Display area
        display_frame = tk.Frame(bottom_frame, bg=self.colors['bg_dark'], relief='solid', bd=2)
        display_frame.pack(side='left', fill='both', expand=True)
        
        self.violation_display_label = tk.Label(
            display_frame,
            text="📹 Video feed will appear here",
            font=('Helvetica', 14),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        self.violation_display_label.pack(expand=True)
        
    def face_toggle_source(self):
        """Toggle between video and webcam for face detection"""
        if self.face_source_var.get() == "video":
            self.face_video_frame.pack(fill='x', pady=10)
            self.face_webcam_frame.pack_forget()
        else:
            self.face_video_frame.pack_forget()
            self.face_webcam_frame.pack(fill='x', pady=10)
            
    def violation_toggle_source(self):
        """Toggle between video and webcam for violation detection"""
        if self.violation_source_var.get() == "video":
            self.violation_video_frame.pack(fill='x', pady=10)
            self.violation_webcam_frame.pack_forget()
        else:
            self.violation_video_frame.pack_forget()
            self.violation_webcam_frame.pack(fill='x', pady=10)
            
    def get_video_files(self):
        """Get list of video files from videos directory"""
        videos_dir = os.path.join(os.path.dirname(__file__), 'videos')
        if not os.path.exists(videos_dir):
            return []
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        return [f for f in os.listdir(videos_dir) if f.lower().endswith(video_extensions)]
        
    def start_face_detection(self):
        """Start face detection"""
        if self.is_detecting:
            messagebox.showwarning("Already Running", "Detection is already running!")
            return
            
        source_type = self.face_source_var.get()
        
        if source_type == "video":
            selected_video = self.face_video_var.get()
            
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
            source = self.face_webcam_var.get()
            source_name = f"Webcam {source}"
        
        if not self.known_face_encodings:
            messagebox.showinfo("No Known Faces", 
                              "No known faces loaded. Faces will be detected but not recognized.\n\n" +
                              "Add images to 'known_faces' folder.")
        
        self.is_detecting = True
        self.face_start_btn.config(state='disabled')
        self.detection_thread = threading.Thread(
            target=self.run_face_detection, 
            args=(source, source_type, source_name)
        )
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
    def run_face_detection(self, source, source_type, source_name):
        """Run face detection in separate thread"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            self.is_detecting = False
            self.face_start_btn.config(state='normal')
            return
        
        frame_skip = 2
        frame_count = 0
        paused = False
        face_locations = []
        face_names = []
        
        while self.is_detecting:
            if not paused:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                                     face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        
                        if self.known_face_encodings:
                            matches = face_recognition.compare_faces(
                                self.known_face_encodings, face_encoding, tolerance=0.6)
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encoding)
                            
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                        
                        face_names.append(name)
                
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                                color, cv2.FILLED)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                              font, 0.8, (255, 255, 255), 1)
                
                status_text = f"Faces: {len(face_locations)} | Press 'q' to quit, 'p' to pause"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Face Detection - Maruthi Dashboard', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
        
        video_capture.release()
        cv2.destroyAllWindows()
        self.is_detecting = False
        self.face_start_btn.config(state='normal')
        
    def start_violation_detection(self):
        """Start violation detection"""
        if self.is_detecting:
            messagebox.showwarning("Already Running", "Detection is already running!")
            return
            
        if self.yolo_model is None:
            messagebox.showerror("Model Not Loaded", 
                               "YOLO model failed to load. Please check installation.")
            return
        
        source_type = self.violation_source_var.get()
        
        if source_type == "video":
            selected_video = self.violation_video_var.get()
            
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
            source = self.violation_webcam_var.get()
            source_name = f"Webcam {source}"
        
        self.violation_count = 0
        self.violation_count_label.config(text=f"Violations: {self.violation_count}")
        
        self.is_detecting = True
        self.violation_start_btn.config(state='disabled')
        self.detection_thread = threading.Thread(
            target=self.run_violation_detection, 
            args=(source, source_type, source_name)
        )
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def run_violation_detection(self, source, source_type, source_name):
        """Run violation detection in separate thread"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            self.is_detecting = False
            self.violation_start_btn.config(state='normal')
            return
        
        paused = False
        frame_count = 0
        process_every_n_frames = 2
        cell_phone_class = 67
        
        while self.is_detecting:
            if not paused:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % process_every_n_frames == 0:
                    confidence = self.violation_confidence_var.get()
                    results = self.yolo_model(frame, conf=confidence, verbose=False)
                    
                    violation_detected = False
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if cls == cell_phone_class:
                                violation_detected = True
                                
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                            (0, 0, 255), 3)
                                
                                label = f"VIOLATION! {conf:.2f}"
                                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), 
                                            (0, 0, 255), -1)
                                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                          (255, 255, 255), 2)
                    
                    if violation_detected:
                        current_time = datetime.now()
                        
                        if (self.last_violation_time is None or 
                            (current_time - self.last_violation_time).total_seconds() 
                            >= self.alert_cooldown):
                            
                            self.violation_count += 1
                            self.violation_count_label.config(
                                text=f"Violations: {self.violation_count}")
                            self.last_violation_time = current_time
                            self.play_alert()
                
                status_text = f"Violations: {self.violation_count} | Press 'q' to quit | 'p' to pause | 's' for screenshot"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), 
                            (0, 0, 0), -1)
                cv2.putText(frame, status_text, (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Mobile Violation Detection - Maruthi Dashboard', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(self.violations_folder, 
                                             f"violation_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        self.is_detecting = False
        self.violation_start_btn.config(state='normal')


if __name__ == "__main__":
    root = tk.Tk()
    app = PremiumDashboard(root)
    root.mainloop()
