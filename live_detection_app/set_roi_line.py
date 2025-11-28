"""
Simple ROI Line Configuration Tool
Set ROI line coordinates via command line for IN/OUT counting
"""

import json
import sys

def create_roi_config():
    print("\n" + "="*70)
    print("ROI LINE CONFIGURATION TOOL")
    print("="*70)
    print("\nChoose ROI line type:")
    print("  1. Horizontal line (Y position)")
    print("  2. Vertical line (X position)")
    print("  3. Custom diagonal line (two points)")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    config = {}
    
    if choice == "1":
        # Horizontal line
        print("\n📏 HORIZONTAL LINE SETUP")
        print("   Direction: Top → Bottom = OUT | Bottom → Top = IN")
        y_pos = int(input("\nEnter Y position (0-1080 for 1920x1080 video): "))
        
        config = {
            'type': 'horizontal',
            'y': y_pos,
            'description': f'Horizontal line at Y={y_pos}'
        }
        
    elif choice == "2":
        # Vertical line
        print("\n📏 VERTICAL LINE SETUP")
        print("   Direction: Left → Right = OUT | Right → Left = IN")
        x_pos = int(input("\nEnter X position (0-1920 for 1920x1080 video): "))
        
        config = {
            'type': 'vertical',
            'x': x_pos,
            'description': f'Vertical line at X={x_pos}'
        }
        
    elif choice == "3":
        # Custom line with two points
        print("\n📏 CUSTOM LINE SETUP")
        print("   Direction: Point1 side → Point2 side = OUT")
        print("   Direction: Point2 side → Point1 side = IN")
        
        print("\nPoint 1 (IN side):")
        x1 = int(input("  X1: "))
        y1 = int(input("  Y1: "))
        
        print("\nPoint 2 (OUT side):")
        x2 = int(input("  X2: "))
        y2 = int(input("  Y2: "))
        
        config = {
            'type': 'custom',
            'line_points': [[x1, y1], [x2, y2]],
            'description': f'Custom line from ({x1},{y1}) to ({x2},{y2})'
        }
    else:
        print("❌ Invalid choice!")
        return None
    
    return config


def main():
    print("\n🎯 Create ROI Line Configuration for IN/OUT Counting")
    
    # Get video resolution
    print("\n" + "="*70)
    print("VIDEO INFORMATION")
    print("="*70)
    print("Your video resolution: 1920x1080")
    print("If different, adjust coordinates accordingly")
    print("="*70)
    
    config = create_roi_config()
    
    if config:
        output_file = 'roi_config.json'
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ CONFIGURATION SAVED")
        print("="*70)
        print(f"File: {output_file}")
        print(f"Config: {config['description']}")
        print("\n🚀 To use this ROI line, run:")
        print(f"\n  python3 inference.py --model weights/best.pt \\")
        print(f"      --source 00119.mp4 --output output_roi.mp4 \\")
        print(f"      --roi-config {output_file}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
