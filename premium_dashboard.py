import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import os
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import threading
import pygame
from datetime import datetime


class PremiumDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Maruthi Security Dashboard")
        
        # Get screen dimensions and set window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1600, screen_width - 100)
        window_height = min(950, screen_height - 100)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Premium color scheme (matching HTML dashboard)
        self.colors = {
            'bg_dark': '#0f1112',
            'sidebar_bg': '#121314',
            'card_bg': '#17181a',
            'border': '#2a2d31',
            'accent_blue': '#3b82f6',
            'accent_green': '#26a44b',
            'accent_red': '#e04b4b',
            'accent_orange': '#f59e0b',
            'text_primary': '#f1f3f5',
            'text_secondary': '#9aa0a6',
            'text_muted': '#6d7377',
            'hover_bg': 'rgba(255,255,255,0.05)'
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Application state
        self.current_view = 'overview'
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
        
        # Load data
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # YOLO model
        self.yolo_model = None
        self.load_yolo_model()
        
        # Audio
        self.audio_available = False
        try:
            pygame.mixer.init()
            self.audio_available = True
        except:
            print("Audio not available")
        
        # Violation tracking
        self.violation_count = 0
        self.last_violation_time = None
        self.alert_cooldown = 3
        
        # Build UI
        self.create_ui()
        
    def load_known_faces(self):
        """Load known faces"""
        if not os.path.exists(self.known_faces_folder):
            return
            
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
                            print(f"✓ Loaded face: {name}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        
        if self.known_face_encodings:
            print(f"✓ Loaded {len(self.known_face_encodings)} known faces")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
    
    def play_alert(self):
        """Play alert sound"""
        try:
            if self.audio_available and os.path.exists(self.alert_sound):
                pygame.mixer.music.load(self.alert_sound)
                pygame.mixer.music.play()
            else:
                print('\a')
        except Exception as e:
            print(f"Error playing alert: {e}")
    
    def create_ui(self):
        """Create main UI"""
        # Main container with grid layout
        container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        container.pack(fill='both', expand=True, padx=18, pady=18)
        
        # Configure grid
        container.grid_columnconfigure(0, weight=0, minsize=260)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)
        
        # Create sidebar and main area
        self.create_sidebar(container)
        self.create_main_area(container)
        
    def create_sidebar(self, parent):
        """Create sidebar navigation"""
        sidebar = tk.Frame(parent, bg=self.colors['sidebar_bg'], width=260)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 24))
        sidebar.grid_propagate(False)
        
        # Brand section
        brand_frame = tk.Frame(sidebar, bg=self.colors['sidebar_bg'])
        brand_frame.pack(fill='x', pady=(18, 10), padx=18)
        
        brand_label = tk.Label(
            brand_frame,
            text="🔐 MARUTHI",
            font=('Inter', 18, 'bold'),
            bg=self.colors['sidebar_bg'],
            fg=self.colors['text_primary']
        )
        brand_label.pack(anchor='w')
        
        subtitle = tk.Label(
            brand_frame,
            text="Security System",
            font=('Inter', 10),
            bg=self.colors['sidebar_bg'],
            fg=self.colors['text_secondary']
        )
        subtitle.pack(anchor='w')
        
        # Navigation section
        nav_frame = tk.Frame(sidebar, bg=self.colors['sidebar_bg'])
        nav_frame.pack(fill='both', expand=True, pady=6, padx=18)
        
        # Section title
        section_title = tk.Label(
            nav_frame,
            text="NAVIGATION",
            font=('Inter', 10),
            bg=self.colors['sidebar_bg'],
            fg=self.colors['text_muted']
        )
        section_title.pack(anchor='w', pady=(8, 8), padx=6)
        
        # Navigation buttons
        self.nav_buttons = {}
        
        nav_items = [
            ('overview', '🏠', 'Overview'),
            ('face-detection', '👤', 'Face Detection'),
            ('violation-detection', '📱', 'Violation Detection')
        ]
        
        for view_id, icon, text in nav_items:
            self.nav_buttons[view_id] = self.create_nav_button(
                nav_frame, view_id, icon, text
            )
        
        # Set initial active button
        self.set_active_nav('overview')
        
    def create_nav_button(self, parent, view_id, icon, text):
        """Create a navigation button"""
        btn_frame = tk.Frame(
            parent,
            bg=self.colors['sidebar_bg'],
            cursor='hand2'
        )
        btn_frame.pack(fill='x', pady=4)
        
        # Button container
        btn = tk.Frame(btn_frame, bg=self.colors['sidebar_bg'], height=44)
        btn.pack(fill='x')
        btn.pack_propagate(False)
        
        # Icon container
        icon_frame = tk.Frame(
            btn,
            bg=self.colors['card_bg'],
            width=34,
            height=34
        )
        icon_frame.pack(side='left', padx=(12, 12), pady=5)
        icon_frame.pack_propagate(False)
        
        icon_label = tk.Label(
            icon_frame,
            text=icon,
            font=('Segoe UI Emoji', 16),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        icon_label.pack(expand=True)
        
        # Text label
        text_label = tk.Label(
            btn,
            text=text,
            font=('Inter', 11),
            bg=self.colors['sidebar_bg'],
            fg=self.colors['text_secondary'],
            anchor='w'
        )
        text_label.pack(side='left', fill='x', expand=True)
        
        # Bind click event
        for widget in [btn, btn_frame, icon_frame, icon_label, text_label]:
            widget.bind('<Button-1>', lambda e, v=view_id: self.switch_view(v))
        
        # Store references
        btn._icon_frame = icon_frame
        btn._icon_label = icon_label
        btn._text_label = text_label
        
        return btn
    
    def set_active_nav(self, view_id):
        """Set active navigation button"""
        for vid, btn in self.nav_buttons.items():
            if vid == view_id:
                # Active style
                btn.configure(bg=self.colors['accent_blue'])
                btn._icon_frame.configure(bg=self.colors['accent_blue'])
                btn._icon_label.configure(bg=self.colors['accent_blue'], fg='white')
                btn._text_label.configure(bg=self.colors['accent_blue'], fg='white')
            else:
                # Inactive style
                btn.configure(bg=self.colors['sidebar_bg'])
                btn._icon_frame.configure(bg=self.colors['card_bg'])
                btn._icon_label.configure(bg=self.colors['card_bg'], fg=self.colors['text_secondary'])
                btn._text_label.configure(bg=self.colors['sidebar_bg'], fg=self.colors['text_secondary'])
    
    def create_main_area(self, parent):
        """Create main content area"""
        self.main_area = tk.Frame(parent, bg=self.colors['bg_dark'])
        self.main_area.grid(row=0, column=1, sticky='nsew')
        
        # Top bar
        self.create_topbar()
        
        # Content sections
        self.overview_section = self.create_overview_section()
        self.face_section = self.create_face_detection_section()
        self.violation_section = self.create_violation_section()
        
        # Show overview initially
        self.show_section('overview')
    
    def create_topbar(self):
        """Create top bar"""
        topbar = tk.Frame(self.main_area, bg=self.colors['bg_dark'], height=80)
        topbar.pack(fill='x', pady=(0, 20))
        topbar.pack_propagate(False)
        
        # Title
        self.page_title = tk.Label(
            topbar,
            text="Dashboard Overview",
            font=('Inter', 24, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_primary']
        )
        self.page_title.pack(side='left', anchor='w')
        
        # Subtitle
        self.page_subtitle = tk.Label(
            topbar,
            text="Real-time monitoring",
            font=('Inter', 12),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        self.page_subtitle.pack(side='left', anchor='w', padx=(15, 0))
    
    def create_overview_section(self):
        """Create overview section"""
        section = tk.Frame(self.main_area, bg=self.colors['bg_dark'])
        
        # Metrics grid
        metrics_frame = tk.Frame(section, bg=self.colors['bg_dark'])
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure grid for 2 columns
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        
        # Face Detection Card
        face_card = self.create_metric_card(
            metrics_frame,
            "👤 Face Detection",
            "Recognize and identify faces",
            lambda: self.switch_view('face-detection')
        )
        face_card.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Violation Detection Card
        violation_card = self.create_metric_card(
            metrics_frame,
            "📱 Violation Detection",
            "Detect mobile phone violations",
            lambda: self.switch_view('violation-detection')
        )
        violation_card.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        # Configure row weights
        metrics_frame.grid_rowconfigure(0, weight=1)
        
        return section
    
    def create_metric_card(self, parent, title, subtitle, click_command):
        """Create a metric card"""
        card = tk.Frame(
            parent,
            bg=self.colors['card_bg'],
            cursor='hand2',
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        
        # Add padding
        inner = tk.Frame(card, bg=self.colors['card_bg'])
        inner.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            inner,
            text=title,
            font=('Inter', 18, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            anchor='w'
        )
        title_label.pack(fill='x', pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(
            inner,
            text=subtitle,
            font=('Inter', 12),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary'],
            anchor='w'
        )
        subtitle_label.pack(fill='x', pady=(0, 20))
        
        # Click to view button
        btn_label = tk.Label(
            inner,
            text="Click to open →",
            font=('Inter', 12),
            bg=self.colors['card_bg'],
            fg=self.colors['accent_blue'],
            anchor='w'
        )
        btn_label.pack(fill='x')
        
        # Bind click
        for widget in [card, inner, title_label, subtitle_label, btn_label]:
            widget.bind('<Button-1>', lambda e: click_command())
        
        return card
    
    def create_face_detection_section(self):
        """Create face detection section"""
        section = tk.Frame(self.main_area, bg=self.colors['bg_dark'])
        
        # Back button
        back_btn = tk.Button(
            section,
            text="← Back to Overview",
            font=('Inter', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary'],
            activebackground=self.colors['border'],
            relief='flat',
            bd=0,
            padx=16,
            pady=10,
            cursor='hand2',
            command=lambda: self.switch_view('overview')
        )
        back_btn.pack(anchor='w', pady=(0, 20), padx=10)
        
        # Controls card
        controls_card = tk.Frame(
            section,
            bg=self.colors['card_bg'],
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        controls_card.pack(fill='x', padx=10, pady=(0, 20))
        
        controls_inner = tk.Frame(controls_card, bg=self.colors['card_bg'])
        controls_inner.pack(fill='both', padx=20, pady=20)
        
        # Title
        tk.Label(
            controls_inner,
            text="👤 Face Recognition System",
            font=('Inter', 16, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        tk.Label(
            controls_inner,
            text="Detect and recognize faces from video files or live webcam feed",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w', pady=(0, 20))
        
        # Source selection
        source_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        source_frame.pack(fill='x', pady=10)
        
        tk.Label(
            source_frame,
            text="Select Source:",
            font=('Inter', 11, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 20))
        
        self.face_source_var = tk.StringVar(value="video")
        
        tk.Radiobutton(
            source_frame,
            text="📹 Video File",
            variable=self.face_source_var,
            value="video",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['card_bg'],
            command=self.face_toggle_source
        ).pack(side='left', padx=10)
        
        tk.Radiobutton(
            source_frame,
            text="📷 Live Webcam",
            variable=self.face_source_var,
            value="webcam",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['card_bg'],
            command=self.face_toggle_source
        ).pack(side='left', padx=10)
        
        # Video selection
        self.face_video_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        self.face_video_frame.pack(fill='x', pady=10)
        
        tk.Label(
            self.face_video_frame,
            text="Select Video:",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.face_video_var = tk.StringVar()
        video_files = self.get_video_files()
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TCombobox',
                       fieldbackground=self.colors['bg_dark'],
                       background=self.colors['card_bg'],
                       foreground=self.colors['text_primary'])
        
        self.face_video_dropdown = ttk.Combobox(
            self.face_video_frame,
            textvariable=self.face_video_var,
            values=video_files,
            state='readonly',
            font=('Inter', 10),
            width=40,
            style='Dark.TCombobox'
        )
        self.face_video_dropdown.pack(side='left', padx=10)
        if video_files:
            self.face_video_dropdown.current(0)
        
        # Webcam selection
        self.face_webcam_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        
        tk.Label(
            self.face_webcam_frame,
            text="Camera Index:",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.face_webcam_var = tk.IntVar(value=0)
        tk.Spinbox(
            self.face_webcam_frame,
            from_=0,
            to=5,
            textvariable=self.face_webcam_var,
            font=('Inter', 10),
            width=10,
            bg=self.colors['bg_dark'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=10)
        
        # Start button
        self.face_start_btn = tk.Button(
            controls_inner,
            text="▶ START DETECTION",
            font=('Inter', 12, 'bold'),
            bg=self.colors['accent_green'],
            fg='white',
            activebackground='#1e8a3d',
            relief='flat',
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2',
            command=self.start_face_detection
        )
        self.face_start_btn.pack(pady=20)
        
        # Info
        info_text = (
            "✓ Place images in 'known_faces' folder for recognition\n"
            "✓ Press 'q' to quit, 'p' to pause during detection\n"
            f"✓ Loaded {len(self.known_face_encodings)} known face(s)"
        )
        
        tk.Label(
            controls_inner,
            text=info_text,
            font=('Inter', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary'],
            justify='left'
        ).pack(anchor='w', pady=(10, 0))
        
        return section
    
    def create_violation_section(self):
        """Create violation detection section"""
        section = tk.Frame(self.main_area, bg=self.colors['bg_dark'])
        
        # Back button
        back_btn = tk.Button(
            section,
            text="← Back to Overview",
            font=('Inter', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary'],
            activebackground=self.colors['border'],
            relief='flat',
            bd=0,
            padx=16,
            pady=10,
            cursor='hand2',
            command=lambda: self.switch_view('overview')
        )
        back_btn.pack(anchor='w', pady=(0, 20), padx=10)
        
        # Controls card
        controls_card = tk.Frame(
            section,
            bg=self.colors['card_bg'],
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        controls_card.pack(fill='x', padx=10, pady=(0, 20))
        
        controls_inner = tk.Frame(controls_card, bg=self.colors['card_bg'])
        controls_inner.pack(fill='both', padx=20, pady=20)
        
        # Title
        tk.Label(
            controls_inner,
            text="📱 Mobile Phone Violation Detection",
            font=('Inter', 16, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        tk.Label(
            controls_inner,
            text="AI-powered detection system with voice alerts for mobile phone violations",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w', pady=(0, 20))
        
        # Source selection
        source_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        source_frame.pack(fill='x', pady=10)
        
        tk.Label(
            source_frame,
            text="Select Source:",
            font=('Inter', 11, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 20))
        
        self.violation_source_var = tk.StringVar(value="video")
        
        tk.Radiobutton(
            source_frame,
            text="📹 Video File",
            variable=self.violation_source_var,
            value="video",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['card_bg'],
            command=self.violation_toggle_source
        ).pack(side='left', padx=10)
        
        tk.Radiobutton(
            source_frame,
            text="📷 Live Webcam",
            variable=self.violation_source_var,
            value="webcam",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_blue'],
            activebackground=self.colors['card_bg'],
            command=self.violation_toggle_source
        ).pack(side='left', padx=10)
        
        # Video selection
        self.violation_video_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        self.violation_video_frame.pack(fill='x', pady=10)
        
        tk.Label(
            self.violation_video_frame,
            text="Select Video:",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_video_var = tk.StringVar()
        video_files = self.get_video_files()
        
        self.violation_video_dropdown = ttk.Combobox(
            self.violation_video_frame,
            textvariable=self.violation_video_var,
            values=video_files,
            state='readonly',
            font=('Inter', 10),
            width=40,
            style='Dark.TCombobox'
        )
        self.violation_video_dropdown.pack(side='left', padx=10)
        if video_files:
            self.violation_video_dropdown.current(0)
        
        # Webcam selection
        self.violation_webcam_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        
        tk.Label(
            self.violation_webcam_frame,
            text="Camera Index:",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_webcam_var = tk.IntVar(value=0)
        tk.Spinbox(
            self.violation_webcam_frame,
            from_=0,
            to=5,
            textvariable=self.violation_webcam_var,
            font=('Inter', 10),
            width=10,
            bg=self.colors['bg_dark'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=10)
        
        # Confidence threshold
        conf_frame = tk.Frame(controls_inner, bg=self.colors['card_bg'])
        conf_frame.pack(fill='x', pady=10)
        
        tk.Label(
            conf_frame,
            text="Detection Confidence:",
            font=('Inter', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side='left', padx=(0, 10))
        
        self.violation_confidence_var = tk.DoubleVar(value=0.3)
        tk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            variable=self.violation_confidence_var,
            font=('Inter', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=300,
            troughcolor=self.colors['bg_dark']
        ).pack(side='left', padx=10)
        
        # Start button
        self.violation_start_btn = tk.Button(
            controls_inner,
            text="▶ START DETECTION",
            font=('Inter', 12, 'bold'),
            bg=self.colors['accent_green'],
            fg='white',
            activebackground='#1e8a3d',
            relief='flat',
            bd=0,
            padx=30,
            pady=12,
            cursor='hand2',
            command=self.start_violation_detection
        )
        self.violation_start_btn.pack(pady=20)
        
        # Stats card
        stats_card = tk.Frame(
            section,
            bg=self.colors['card_bg'],
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        stats_card.pack(fill='x', padx=10)
        
        stats_inner = tk.Frame(stats_card, bg=self.colors['card_bg'])
        stats_inner.pack(fill='both', padx=20, pady=20)
        
        tk.Label(
            stats_inner,
            text="📊 STATISTICS",
            font=('Inter', 14, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(pady=(0, 15))
        
        self.violation_count_label = tk.Label(
            stats_inner,
            text="Violations: 0",
            font=('Inter', 20, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent_red']
        )
        self.violation_count_label.pack()
        
        return section
    
    def face_toggle_source(self):
        """Toggle face detection source"""
        if self.face_source_var.get() == "video":
            self.face_video_frame.pack(fill='x', pady=10)
            self.face_webcam_frame.pack_forget()
        else:
            self.face_video_frame.pack_forget()
            self.face_webcam_frame.pack(fill='x', pady=10)
    
    def violation_toggle_source(self):
        """Toggle violation detection source"""
        if self.violation_source_var.get() == "video":
            self.violation_video_frame.pack(fill='x', pady=10)
            self.violation_webcam_frame.pack_forget()
        else:
            self.violation_video_frame.pack_forget()
            self.violation_webcam_frame.pack(fill='x', pady=10)
    
    def get_video_files(self):
        """Get list of video files"""
        if not os.path.exists(self.videos_folder):
            return []
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        return [f for f in os.listdir(self.videos_folder) 
                if f.lower().endswith(video_extensions)]
    
    def switch_view(self, view_id):
        """Switch between views"""
        self.current_view = view_id
        self.set_active_nav(view_id)
        
        # Update page title
        titles = {
            'overview': ('Dashboard Overview', 'Real-time monitoring'),
            'face-detection': ('Face Detection', 'Recognition system'),
            'violation-detection': ('Violation Detection', 'Mobile phone detection')
        }
        
        if view_id in titles:
            title, subtitle = titles[view_id]
            self.page_title.config(text=title)
            self.page_subtitle.config(text=subtitle)
        
        # Show appropriate section
        self.show_section(view_id)
    
    def show_section(self, section_name):
        """Show specific section"""
        # Hide all sections
        self.overview_section.pack_forget()
        self.face_section.pack_forget()
        self.violation_section.pack_forget()
        
        # Show requested section
        if section_name == 'overview':
            self.overview_section.pack(fill='both', expand=True)
        elif section_name == 'face-detection':
            self.face_section.pack(fill='both', expand=True)
        elif section_name == 'violation-detection':
            self.violation_section.pack(fill='both', expand=True)
    
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
                              "No known faces loaded. Faces will be detected but not recognized.")
        
        self.is_detecting = True
        self.face_start_btn.config(state='disabled', bg=self.colors['text_muted'])
        self.detection_thread = threading.Thread(
            target=self.run_face_detection, 
            args=(source, source_type, source_name),
            daemon=True
        )
        self.detection_thread.start()
    
    def run_face_detection(self, source, source_type, source_name):
        """Run face detection"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            self.is_detecting = False
            self.face_start_btn.config(state='normal', bg=self.colors['accent_green'])
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
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
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
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
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
        self.face_start_btn.config(state='normal', bg=self.colors['accent_green'])
    
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
        self.violation_start_btn.config(state='disabled', bg=self.colors['text_muted'])
        self.detection_thread = threading.Thread(
            target=self.run_violation_detection, 
            args=(source, source_type, source_name),
            daemon=True
        )
        self.detection_thread.start()
    
    def run_violation_detection(self, source, source_type, source_name):
        """Run violation detection"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            self.is_detecting = False
            self.violation_start_btn.config(state='normal', bg=self.colors['accent_green'])
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
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                label = f"VIOLATION! {conf:.2f}"
                                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 255), -1)
                                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if violation_detected:
                        current_time = datetime.now()
                        
                        if (self.last_violation_time is None or 
                            (current_time - self.last_violation_time).total_seconds() >= self.alert_cooldown):
                            
                            self.violation_count += 1
                            self.violation_count_label.config(text=f"Violations: {self.violation_count}")
                            self.last_violation_time = current_time
                            self.play_alert()
                
                status_text = f"Violations: {self.violation_count} | Press 'q' to quit | 'p' to pause | 's' for screenshot"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
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
        self.violation_start_btn.config(state='normal', bg=self.colors['accent_green'])


if __name__ == "__main__":
    root = tk.Tk()
    app = PremiumDashboard(root)
    root.mainloop()
