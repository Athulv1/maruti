import os
import shutil
import face_recognition
from pathlib import Path


def add_known_face():
    """Helper script to add a known face to the system"""
    
    known_faces_folder = "known_faces"
    os.makedirs(known_faces_folder, exist_ok=True)
    
    print("\n" + "="*60)
    print("         Add Known Face to Recognition System")
    print("="*60 + "\n")
    
    # Get image path
    while True:
        image_path = input("Enter the path to the face image: ").strip()
        
        if not image_path:
            print("Path cannot be empty. Please try again.")
            continue
            
        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'")
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                return
            continue
        
        # Check if it's a valid image
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext not in valid_extensions:
            print(f"Error: Invalid file format. Supported formats: {', '.join(valid_extensions)}")
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                return
            continue
        
        break
    
    # Try to load and verify face in image
    print("\nVerifying face in image...")
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            print("Error: No face detected in the image!")
            print("Please make sure:")
            print("  - The image contains a clear, front-facing face")
            print("  - The face is well-lit")
            print("  - The image quality is good")
            return
        
        if len(face_encodings) > 1:
            print(f"Warning: Detected {len(face_encodings)} faces in the image.")
            print("For best results, use images with only one face.")
            proceed = input("Do you want to proceed anyway? (y/n): ").lower()
            if proceed != 'y':
                return
        
        print("✓ Face detected successfully!")
        
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return
    
    # Get person's name
    while True:
        name = input("\nEnter the person's name (e.g., John Doe): ").strip()
        
        if not name:
            print("Name cannot be empty. Please try again.")
            continue
        
        break
    
    # Create filename (replace spaces with underscores)
    filename = name.replace(' ', '_') + os.path.splitext(image_path)[1]
    destination_path = os.path.join(known_faces_folder, filename)
    
    # Check if file already exists
    if os.path.exists(destination_path):
        print(f"\nWarning: A face image for '{name}' already exists.")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
    
    # Copy the file
    try:
        shutil.copy2(image_path, destination_path)
        print(f"\n✓ Successfully added '{name}' to known faces!")
        print(f"  Saved as: {destination_path}")
        print("\nThe face will be recognized when you run the face detection app.")
        
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return
    
    # Ask if user wants to add more
    print("\n" + "-"*60)
    add_more = input("Would you like to add another face? (y/n): ").lower()
    if add_more == 'y':
        add_known_face()


def list_known_faces():
    """List all known faces in the system"""
    known_faces_folder = "known_faces"
    
    if not os.path.exists(known_faces_folder):
        print("No known faces folder found.")
        return
    
    files = [f for f in os.listdir(known_faces_folder) 
             if os.path.isfile(os.path.join(known_faces_folder, f))]
    
    if not files:
        print("No known faces found.")
        return
    
    print("\n" + "="*60)
    print("                    Known Faces")
    print("="*60)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    
    for filename in sorted(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            count += 1
            name = os.path.splitext(filename)[0].replace('_', ' ')
            print(f"{count}. {name} ({filename})")
    
    print("="*60)
    print(f"Total: {count} known face(s)\n")


def main():
    print("\n" + "="*60)
    print("           Face Recognition - Known Faces Manager")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Add a new known face")
        print("2. List all known faces")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            add_known_face()
        elif choice == '2':
            list_known_faces()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
