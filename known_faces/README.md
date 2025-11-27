# Known Faces Folder

Place face images in this folder for recognition.

## Naming Convention
- Use format: `FirstName_LastName.jpg`
- Examples: `John_Doe.jpg`, `Jane_Smith.jpg`
- Underscores will be replaced with spaces in the display

## Image Requirements
- One face per image
- Clear, front-facing photo
- Good lighting
- Higher resolution is better

## Supported Formats
- .jpg / .jpeg
- .png
- .bmp

## How to Add Known Faces

### Method 1: Manual
1. Copy face images to this folder
2. Rename them according to the naming convention above
3. Restart the face detection application

### Method 2: Using Helper Script
Run: `python add_known_face.py`

The script will:
- Verify that the image contains a face
- Ask for the person's name
- Copy and rename the file automatically
