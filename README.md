# Face Detection System

A Python-based face detection and recognition system with a user-friendly GUI that allows you to select videos from a dropdown and detect/recognize faces in real-time.

## Features

- **Video Selection**: Dropdown list to select from videos in your videos folder
- **Face Detection**: Automatically detects faces in the selected video
- **Face Recognition**: Recognizes known faces and displays their names
- **Real-time Processing**: Processes video frames in real-time with optimized performance
- **User-friendly GUI**: Simple tkinter-based interface
- **Pause/Resume**: Control playback during face detection

## Project Structure

```
maruthi/
├── face_detection_app.py      # Main application
├── add_known_face.py          # Helper script to add known faces
├── requirements.txt           # Python dependencies
├── videos/                    # Place your video files here
└── known_faces/              # Place known face images here
    ├── John_Doe.jpg
    ├── Jane_Smith.jpg
    └── ...
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

### 1. Add Known Faces

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

### 2. Add Videos

- Place your video files in the `videos/` folder
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

### 3. Run the Application

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Then run the application
python face_detection_app.py
```

### 4. Using the Application

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

## Future Enhancements (Use Cases 2 & 3)

This is the first use case. Additional features will be added:
- Use Case 2: TBD
- Use Case 3: TBD

## License

This project is open-source and available for educational purposes.
