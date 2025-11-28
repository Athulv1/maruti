"""
Interactive ROI Line Setup Tool
Click two points on the video frame to set the ROI line for IN/OUT counting
"""

import cv2
import json
from pathlib import Path

class ROILineSetup:
    def __init__(self, video_path):
        self.video_path = video_path
        self.points = []
        self.frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to set ROI line points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Draw the point
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                
                # If we have 2 points, draw the line
                if len(self.points) == 2:
                    cv2.line(self.frame, self.points[0], self.points[1], (0, 255, 255), 3)
                    cv2.putText(self.frame, "Press 'S' to save, 'R' to reset", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Setup ROI Line', self.frame)
    
    def setup(self):
        """Open video and allow user to set ROI line"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from video")
            return None
        
        # Resize for display if too large
        height, width = frame.shape[:2]
        max_width = 1280
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            print(f"Display resized to {new_width}x{new_height} (original: {width}x{height})")
            # Store scale for converting back
            self.scale = scale
            self.original_size = (width, height)
        else:
            self.scale = 1.0
            self.original_size = (width, height)
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        # Instructions
        cv2.putText(self.frame, "Click 2 points to draw ROI line", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.frame, "Point 1 side = IN | Point 2 side = OUT", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Setup ROI Line', self.frame)
        cv2.setMouseCallback('Setup ROI Line', self.mouse_callback)
        
        print("\n" + "="*60)
        print("ROI LINE SETUP")
        print("="*60)
        print("Instructions:")
        print("  1. Click TWO points to define the ROI line")
        print("  2. Direction: Objects crossing FROM Point-1 side TO Point-2 side = OUT")
        print("  3. Direction: Objects crossing FROM Point-2 side TO Point-1 side = IN")
        print("\nControls:")
        print("  'S' - Save ROI line configuration")
        print("  'R' - Reset and redraw line")
        print("  'Q' - Quit without saving")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if len(self.points) == 2:
                    cv2.destroyAllWindows()
                    return self.get_roi_config()
                else:
                    print("Please select 2 points first!")
            
            elif key == ord('r') or key == ord('R'):
                # Reset
                self.points = []
                self.frame = self.original_frame.copy()
                cv2.putText(self.frame, "Click 2 points to draw ROI line", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(self.frame, "Point 1 side = IN | Point 2 side = OUT", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow('Setup ROI Line', self.frame)
                print("Reset! Click 2 points again.")
            
            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return None
        
    def get_roi_config(self):
        """Convert selected points to ROI configuration"""
        if len(self.points) != 2:
            return None
        
        # Scale points back to original video size
        p1 = (int(self.points[0][0] / self.scale), int(self.points[0][1] / self.scale))
        p2 = (int(self.points[1][0] / self.scale), int(self.points[1][1] / self.scale))
        
        config = {
            'line_points': [p1, p2],
            'original_resolution': self.original_size,
            'description': 'Custom ROI line for IN/OUT counting'
        }
        
        print("\n" + "="*60)
        print("ROI LINE CONFIGURATION")
        print("="*60)
        print(f"Point 1: {p1}")
        print(f"Point 2: {p2}")
        print(f"Original video resolution: {self.original_size}")
        print("\nDirection Rules:")
        print(f"  • Crossing from Point-1 side → Point-2 side = OUT")
        print(f"  • Crossing from Point-2 side → Point-1 side = IN")
        print("="*60 + "\n")
        
        return config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive ROI Line Setup')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='roi_config.json', help='Output JSON config file')
    
    args = parser.parse_args()
    
    # Setup ROI
    setup = ROILineSetup(args.video)
    config = setup.setup()
    
    if config:
        # Save configuration
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ ROI configuration saved to: {args.output}")
        print("\nTo use this ROI line in inference, run:")
        print(f"  python3 inference.py --model weights/best.pt --source {args.video} \\")
        print(f"      --output output.mp4 --roi-config {args.output}")
    else:
        print("Setup cancelled.")


if __name__ == "__main__":
    main()
