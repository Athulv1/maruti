# Face Detection System

A Python-based face detection and recognition system with a user-friendly GUI that allows you to select videos from a dropdown and detect/recognize faces in real-time.

## Features

### Use Case 1: Face Detection & Recognition
- **Video Selection**: Dropdown list to select from videos in your videos folder
- **Face Detection**: Automatically detects faces in the selected video
- **Face Recognition**: Recognizes known faces and displays their names
- **Real-time Processing**: Processes video frames in real-time with optimized performance
- **User-friendly GUI**: Simple tkinter-based interface
- **Pause/Resume**: Control playback during face detection

### Use Case 2: Mobile Phone Violation Detection
- **Video Selection**: Same dropdown interface for video selection
- **Mobile Detection**: Detects people using mobile phones/cell phones
- **Audio Alerts**: Plays sound alert when violation is detected
- **Violation Counter**: Real-time count of violations detected
- **Screenshot Capture**: Save screenshots of violations
- **Adjustable Sensitivity**: Configure confidence threshold
- **Violation Logging**: Automatically saves violation screenshots

## Project Structure

```
maruthi/
├── face_detection_app.py           # Use Case 1: Face Detection & Recognition
├── mobile_violation_detection.py   # Use Case 2: Mobile Phone Violation Detection
├── add_known_face.py               # Helper script to add known faces
├── create_alert_sound.py           # Helper to generate alert sound
├── requirements.txt                # Python dependencies
├── run.sh                          # Convenience script to run apps
├── videos/                         # Place your video files here
├── known_faces/                    # Place known face images here (Use Case 1)
├── violations/                     # Screenshots of violations saved here
└── violation_alert.wav             # Alert sound for violations
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Install System Dependencies

**For Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-tk cmake libopenblas-dev liblapack-dev
```

**For Fedora/CentOS:**
```bash
sudo dnf install python3-tkinter cmake openblas-devel lapack-devel
```

**For macOS:**
```bash
brew install cmake
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installing `dlib` may take some time as it needs to be compiled.

## Usage

### Use Case 1: Face Detection & Recognition

#### 1. Add Known Faces

To recognize faces, you need to add known face images to the `known_faces` folder:

**Option A: Manual Method**
- Add face images to the `known_faces/` folder
- Name format: `FirstName_LastName.jpg` (e.g., `John_Doe.jpg`, `Jane_Smith.jpg`)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Each image should contain only one face

**Option B: Using Helper Script**
```bash
python add_known_face.py
```
This script will guide you through adding a known face from an image file.

#### 2. Add Videos

- Place your video files in the `videos/` folder
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

#### 3. Run Face Detection Application

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Then run the application
python face_detection_app.py
```

#### 4. Using Face Detection App

1. The application window will open
2. Click "Refresh" to load videos from the videos folder
3. Select a video from the dropdown list
4. Click "Start Face Detection"
5. The video will play with face detection:
   - **Green boxes**: Recognized faces (with names)
   - **Red boxes**: Unknown faces
6. Controls during playback:
   - Press **'q'** to quit
   - Press **'p'** to pause/resume

---

### Use Case 2: Mobile Phone Violation Detection

#### 1. Prepare Alert Sound (Optional)

The app will auto-generate an alert sound, but you can create a custom one:

```bash
source venv/bin/activate
python create_alert_sound.py
```

Or replace `violation_alert.wav` with your own audio file.

#### 2. Add Videos

Use the same `videos/` folder as Use Case 1.

#### 3. Run Mobile Violation Detection

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac

# Run the application
python mobile_violation_detection.py
```

#### 4. Using Violation Detection App

1. The application window will open
2. Configure settings:
   - Adjust **Confidence Threshold** (0.1 to 0.9)
     - Lower = More sensitive (more detections)
     - Higher = More strict (fewer false positives)
   - Enable/Disable audio alerts
3. Click "Test Alert Sound" to verify audio works
4. Select a video from the dropdown
5. Click "Start Detection"
6. The video will play with detection:
   - **Red boxes**: Mobile phone detected (violation)
   - **Audio alert**: Plays when violation found
   - **Counter**: Shows total violations
7. Controls during playback:
   - Press **'q'** to quit
   - Press **'p'** to pause/resume
   - Press **'s'** to save screenshot
8. Screenshots are automatically saved in `violations/` folder

---

## Detection Details

### Mobile Phone Detection
- Uses **YOLOv8** (You Only Look Once) deep learning model
- Detects cell phones in real-time video
- Confidence threshold adjustable (default: 0.3 for high sensitivity)
- 3-second cooldown between alerts to avoid spam
- Processes every 2nd frame for better performance
- **Voice Alert**: Says "Violation Found" when mobile phone detected

## Tips for Best Results

1. **Known Face Images**:
   - Use clear, front-facing photos
   - Good lighting conditions
   - One face per image
   - Higher resolution is better

2. **Video Quality**:
   - Higher resolution videos work better
   - Good lighting in the video improves detection

3. **Performance**:
   - The app skips frames for better performance
   - Lower resolution videos process faster
   - You can adjust `frame_skip` in the code for speed/accuracy trade-off

## Troubleshooting

### Installation Issues

**dlib installation fails:**
```bash
# Install build tools first
pip install cmake
# Then try again
pip install dlib
```

**face_recognition installation fails:**
Make sure dlib is installed successfully first, then install face_recognition.

### Runtime Issues

**"No known faces loaded"**:
- Add face images to the `known_faces/` folder
- Restart the application

**"No videos found"**:
- Add video files to the `videos/` folder
- Click the "Refresh" button

**Video won't play**:
- Ensure the video format is supported
- Check if opencv-python is installed correctly

**No audio alert (Use Case 2)**:
- Check if `violation_alert.wav` exists
- Run `python create_alert_sound.py` to generate it
- Verify system audio is not muted
- Check "Enable Audio Alert" is checked
- **For headless/server systems**: Audio may not work without sound card
  - The app will still detect violations and save screenshots
  - Visual alerts (red boxes and counter) will still work
  - Consider using a system with audio output for alerts

**YOLO model not loading (Use Case 2)**:
- First run will download the model automatically
- Requires internet connection for initial download
- Model will be cached for future use

## Project Applications

## Future Enhancements (Use Case 3)

This project currently implements two use cases:
- ✅ Use Case 1: Face Detection & Recognition
- ✅ Use Case 2: Mobile Phone Violation Detection
- ⏳ Use Case 3: TBD

## License

This project is open-source and available for educational purposes.
