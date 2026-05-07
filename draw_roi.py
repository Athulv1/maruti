"""
ROI Zone Drawing Tool — Polygons + Tilted Line

Steps:
  1. Click 4 points for UPPER zone (GREEN)
  2. Click 4 points for LOWER zone (RED)
  3. Click 2 points for COUNTING LINE (YELLOW)
  4. Drag any point to adjust!

Controls:
  Left Click/Drag = Place or move points
  R               = Reset all
  S               = Save and exit
  Q / ESC         = Quit without saving
"""

import cv2
import json
import sys
import numpy as np

VIDEO_PATH = 'videos/test.mp4'
ROI_CONFIG_FILE = 'roi_config.json'

upper_pts = []
lower_pts = []
line_pts = []  # 2 endpoints for tilted line
phase = 'upper'  # 'upper', 'lower', 'line', 'adjust'
current_frame = None
dragging = None
DRAG_RADIUS = 18


def find_nearest_point(x, y):
    for i, p in enumerate(upper_pts):
        if ((p[0]-x)**2 + (p[1]-y)**2)**0.5 < DRAG_RADIUS:
            return ('upper', i)
    for i, p in enumerate(lower_pts):
        if ((p[0]-x)**2 + (p[1]-y)**2)**0.5 < DRAG_RADIUS:
            return ('lower', i)
    for i, p in enumerate(line_pts):
        if ((p[0]-x)**2 + (p[1]-y)**2)**0.5 < DRAG_RADIUS:
            return ('line', i)
    return None


def mouse_callback(event, x, y, flags, param):
    global upper_pts, lower_pts, line_pts, phase, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # Try drag existing point first
        hit = find_nearest_point(x, y)
        if hit:
            dragging = hit
            return

        # Place new points
        if phase == 'upper' and len(upper_pts) < 4:
            upper_pts.append([x, y])
            if len(upper_pts) == 4:
                phase = 'lower'
                print(f"  ✅ Upper zone — now click 4 pts for LOWER (red)")
        elif phase == 'lower' and len(lower_pts) < 4:
            lower_pts.append([x, y])
            if len(lower_pts) == 4:
                phase = 'line'
                print(f"  ✅ Lower zone — now click 2 pts for LINE (yellow)")
        elif phase == 'line' and len(line_pts) < 2:
            line_pts.append([x, y])
            if len(line_pts) == 2:
                phase = 'adjust'
                print(f"  ✅ All set! Drag to adjust | S=Save")
        redraw()

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            name, idx = dragging
            if name == 'upper':
                upper_pts[idx] = [x, y]
            elif name == 'lower':
                lower_pts[idx] = [x, y]
            elif name == 'line':
                line_pts[idx] = [x, y]
            redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = None


def draw_zone(display, pts, color, label):
    poly = np.array(pts, np.int32).reshape((-1, 1, 2))
    overlay = display.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
    cv2.polylines(display, [poly], True, color, 2)
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    cv2.putText(display, label, (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, p in enumerate(pts):
        cv2.circle(display, tuple(p), DRAG_RADIUS, color, 2)
        cv2.circle(display, tuple(p), 4, (255, 255, 255), -1)


def redraw():
    display = current_frame.copy()
    h, w = display.shape[:2]

    # Instructions
    if phase == 'upper':
        msg = f"Click 4 pts for UPPER zone [{len(upper_pts)}/4]"
    elif phase == 'lower':
        msg = f"Click 4 pts for LOWER zone [{len(lower_pts)}/4]"
    elif phase == 'line':
        msg = f"Click 2 pts for COUNTING LINE [{len(line_pts)}/2]"
    else:
        msg = "Drag any point to adjust | S=Save | R=Reset"
    cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw partial points
    for pts_list, color in [(upper_pts, (0,255,0)), (lower_pts, (0,0,255))]:
        if 0 < len(pts_list) < 4:
            for i, p in enumerate(pts_list):
                cv2.circle(display, tuple(p), 6, color, -1)
                if i > 0:
                    cv2.line(display, tuple(pts_list[i-1]), tuple(p), color, 2)

    # Draw completed zones
    if len(upper_pts) == 4:
        draw_zone(display, upper_pts, (0, 255, 0), "UPPER")
    if len(lower_pts) == 4:
        draw_zone(display, lower_pts, (0, 0, 255), "LOWER")

    # Draw line
    if len(line_pts) == 1:
        cv2.circle(display, tuple(line_pts[0]), 8, (0, 255, 255), -1)
    elif len(line_pts) == 2:
        cv2.line(display, tuple(line_pts[0]), tuple(line_pts[1]), (0, 255, 255), 3)
        # Endpoints with handles
        cv2.circle(display, tuple(line_pts[0]), DRAG_RADIUS, (0, 255, 255), 2)
        cv2.circle(display, tuple(line_pts[0]), 5, (0, 255, 255), -1)
        cv2.circle(display, tuple(line_pts[1]), DRAG_RADIUS, (0, 255, 255), 2)
        cv2.circle(display, tuple(line_pts[1]), 5, (0, 255, 255), -1)
        # Labels
        mid_x = (line_pts[0][0] + line_pts[1][0]) // 2
        mid_y = (line_pts[0][1] + line_pts[1][1]) // 2
        cv2.putText(display, "COUNT LINE", (mid_x - 60, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Direction arrows (perpendicular to line)
        cv2.putText(display, "^ IN (upper)", (mid_x + 40, mid_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display, "v OUT (lower)", (mid_x + 40, mid_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Draw ROI Zones', display)


def main():
    global upper_pts, lower_pts, line_pts, phase, current_frame

    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    ret, current_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read frame")
        return

    print(f"\n{'='*60}")
    print(f"  ROI Zone Drawing — Polygons + Tilted Line")
    print(f"{'='*60}")
    print(f"  Video: {video_path} ({current_frame.shape[1]}x{current_frame.shape[0]})")
    print(f"")
    print(f"  1. Click 4 points for UPPER zone (green)")
    print(f"  2. Click 4 points for LOWER zone (red)")
    print(f"  3. Click 2 points for COUNTING LINE (yellow)")
    print(f"  4. Drag any circle to adjust!")
    print(f"")
    print(f"  Upper→Lower = OUT | Lower→Upper = IN")
    print(f"  S=Save  R=Reset  Q=Quit")
    print(f"{'='*60}\n")

    cv2.namedWindow('Draw ROI Zones', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Draw ROI Zones', 1280, 720)
    cv2.setMouseCallback('Draw ROI Zones', mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            upper_pts, lower_pts, line_pts = [], [], []
            phase = 'upper'
            redraw()
            print("  Reset")

        elif key == ord('s'):
            if len(upper_pts) == 4 and len(lower_pts) == 4 and len(line_pts) == 2:
                config = {
                    'type': 'zones',
                    'upper_box': upper_pts,
                    'lower_box': lower_pts,
                    'line_points': line_pts,
                    'description': 'Polygon zones + tilted counting line'
                }
                with open(ROI_CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"\n  ✅ Saved to {ROI_CONFIG_FILE}")
                print(f"     Upper: {upper_pts}")
                print(f"     Lower: {lower_pts}")
                print(f"     Line:  {line_pts[0]} → {line_pts[1]}")
                print(f"\n     Restart app.py to use!")
                break
            else:
                print(f"  ⚠️ Need all: upper({len(upper_pts)}/4) lower({len(lower_pts)}/4) line({len(line_pts)}/2)")

        elif key in (ord('q'), 27):
            print("  Cancelled")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
