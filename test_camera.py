#!/usr/bin/env python3
"""
Quick camera test script to find working camera index
"""
import cv2
import sys


def test_camera(index):
    """Test if camera at given index works"""
    print(f"\nTesting camera index {index}...")
    
    # Try different backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_GSTREAMER, "GSTREAMER"),
    ]
    
    for backend, name in backends:
        print(f"  Trying backend: {name}")
        cap = cv2.VideoCapture(index, backend)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ SUCCESS! Camera {index} works with backend {name}")
                print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            cap.release()
    
    print(f"  ✗ Camera {index} failed with all backends")
    return False


def find_cameras():
    """Find all available cameras"""
    print("="*60)
    print("Camera Detection Test")
    print("="*60)
    
    working_cameras = []
    
    # Test indices 0-5
    for i in range(6):
        if test_camera(i):
            working_cameras.append(i)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    if working_cameras:
        print(f"\n✓ Found {len(working_cameras)} working camera(s): {working_cameras}")
        print(f"\nUse camera index: {working_cameras[0]} in the applications")
        return working_cameras[0]
    else:
        print("\n✗ No working cameras found!")
        print("\nTroubleshooting:")
        print("1. Check if camera is connected")
        print("2. Check camera permissions: ls -l /dev/video*")
        print("3. Add user to video group: sudo usermod -a -G video $USER")
        print("4. Try external USB webcam")
        return None


def test_live_feed(camera_index):
    """Test live feed from camera"""
    print(f"\nTesting live feed from camera {camera_index}...")
    print("Press 'q' to quit the test")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("✗ Could not open camera!")
        return
    
    print("✓ Camera opened successfully!")
    print("  Displaying live feed...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error reading frame")
            break
        
        # Add text overlay
        cv2.putText(frame, f"Camera {camera_index} - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera Test - Index {camera_index}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera test completed!")


if __name__ == "__main__":
    camera_index = find_cameras()
    
    if camera_index is not None:
        print("\n" + "="*60)
        response = input("\nWould you like to test live feed? (y/n): ").strip().lower()
        if response == 'y':
            test_live_feed(camera_index)
    else:
        print("\nNo cameras available to test.")
        sys.exit(1)
