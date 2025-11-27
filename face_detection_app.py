import cv2
import face_recognition
import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
from pathlib import Path


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection System")
        self.root.geometry("800x600")
        
        # Paths
        self.videos_folder = "videos"
        self.known_faces_folder = "known_faces"
        
        # Storage for known faces
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create folders if they don't exist
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.known_faces_folder, exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Face Detection System", 
                              font=("Arial", 20, "bold"), pady=20)
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
        
        # Dropdown for video selection
        self.video_var = tk.StringVar()
        self.video_dropdown = ttk.Combobox(self.file_frame, 
                                          textvariable=self.video_var,
                                          state="readonly",
                                          width=35)
        self.video_dropdown.pack(side="left", padx=5)
        
        # Refresh button
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
        
        # Start detection button
        start_btn = tk.Button(self.root, text="Start Face Detection", 
                             command=self.start_detection,
                             bg="#2196F3", fg="white", 
                             font=("Arial", 12, "bold"),
                             padx=20, pady=10)
        start_btn.pack(pady=20)
        
        # Info frame
        info_frame = tk.Frame(self.root, pady=10)
        info_frame.pack(fill="both", expand=True, padx=20)
        
        tk.Label(info_frame, text="Instructions:", 
                font=("Arial", 12, "bold")).pack(anchor="w")
        
        instructions = [
            "1. Choose video source: Video File or Live Webcam",
            "2. For videos: Place files in 'videos' folder & click 'Refresh'",
            "3. For webcam: Select camera index (usually 0)",
            "4. Add known face images in the 'known_faces' folder",
            "   - Name format: 'PersonName.jpg' (e.g., 'John_Doe.jpg')",
            "5. Click 'Start Face Detection' to begin",
            "",
            "Controls during playback:",
            "- Press 'q' to quit",
            "- Press 'p' to pause/resume"
        ]
        
        for instruction in instructions:
            tk.Label(info_frame, text=instruction, 
                    font=("Arial", 10), anchor="w").pack(anchor="w", pady=2)
        
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
                        # Load image
                        image = face_recognition.load_image_file(file_path)
                        # Get face encoding
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            # Use filename without extension as name
                            name = os.path.splitext(filename)[0].replace('_', ' ')
                            self.known_face_names.append(name)
                            print(f"Loaded face: {name}")
                        else:
                            print(f"No face found in {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        
        if self.known_face_encodings:
            print(f"Loaded {len(self.known_face_encodings)} known faces")
        else:
            print("No known faces loaded. Add images to 'known_faces' folder.")
    
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
    
    def start_detection(self):
        """Start face detection on selected source"""
        source_type = self.source_var.get()
        
        if source_type == "file":
            selected_video = self.video_var.get()
            
            if not selected_video:
                messagebox.showwarning("No Video Selected", 
                                     "Please select a video from the dropdown.")
                return
            
            if not self.known_face_encodings:
                messagebox.showinfo("No Known Faces", 
                                  "No known faces loaded. Faces will be detected but not recognized.\n\n" +
                                  "Add images to 'known_faces' folder and restart the app.")
            
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
        
        self.detect_faces_in_video(source, source_type, source_name)
    
    def detect_faces_in_video(self, source, source_type="file", source_name=""):
        """Process video/webcam and detect faces"""
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open {source_type}!")
            return
        
        self.status_label.config(text=f"Processing {source_name}... Press 'q' to quit, 'p' to pause", 
                               fg="blue")
        self.root.update()
        
        # Process every nth frame for better performance
        frame_skip = 2
        frame_count = 0
        paused = False
        
        # For tracking faces between frames
        face_locations = []
        face_names = []
        
        while True:
            if not paused:
                ret, frame = video_capture.read()
                
                if not ret:
                    if source_type == "webcam":
                        print("Webcam feed interrupted")
                    break
                
                frame_count += 1
                
                # Only process every nth frame
                if frame_count % frame_skip == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                                     face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        
                        if self.known_face_encodings:
                            # Compare with known faces
                            matches = face_recognition.compare_faces(
                                self.known_face_encodings, face_encoding, tolerance=0.6)
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encoding)
                            
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                        
                        face_names.append(name)
                
                # Draw rectangles and names
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up coordinates
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    # Draw rectangle
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                                color, cv2.FILLED)
                    
                    # Draw name
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                              font, 0.8, (255, 255, 255), 1)
                
                # Add status text
                status_text = f"Faces: {len(face_locations)} | Press 'q' to quit, 'p' to pause"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Face Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
        self.status_label.config(text="Video processing completed", fg="green")


def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
