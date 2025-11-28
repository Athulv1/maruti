"""
YOLOv11 Inference Script for MOBILE Detection and OUT Counting
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import time
from scipy.spatial import distance as dist


class CentroidTracker:
    """Simple centroid-based object tracker for counting line crossings"""
    
    def __init__(self, max_disappeared=50):
        """
        Initialize the centroid tracker
        
        Args:
            max_disappeared: Maximum frames an object can be missing before being deregistered
        """
        self.next_object_id = 0
        self.objects = {}  # ID -> centroid
        self.disappeared = {}  # ID -> frame count
        self.max_disappeared = max_disappeared
        
        # Track crossing state for each object
        self.crossed = {}  # ID -> {'crossed': bool, 'direction': 'in'/'out', 'start_side': 'top'/'bottom'}
    
    def register(self, centroid):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.crossed[self.next_object_id] = {'crossed': False, 'direction': None, 'start_side': None}
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.crossed:
            del self.crossed[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Dictionary of object_id -> centroid
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Calculate centroids from bounding boxes
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no objects tracked yet, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance between existing and new centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update object position
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects


class MobileOutDetector:
    """Detector for MOBILE and OUT objects with counting capabilities"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the trained model weights (best.pt)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = {0: 'MOBILE', 1: 'OUT'}
        
        print(f"✓ Model loaded successfully")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  IOU threshold: {iou_threshold}")
    
    def detect_image(self, image_path, save_path=None, show=False):
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            show: Whether to display the result
            
        Returns:
            Dictionary with detection results and counts
        """
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Count detections by class
        counts = {'MOBILE': 0, 'OUT': 0}
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            class_name = self.class_names[class_id]
            counts[class_name] += 1
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox.tolist()
            })
        
        # Prepare result
        result = {
            'image': str(image_path),
            'counts': counts,
            'total_detections': len(detections),
            'detections': detections
        }
        
        # Save annotated image
        if save_path or show:
            annotated = results.plot()
            
            # Add count text
            h, w = annotated.shape[:2]
            text = f"MOBILE: {counts['MOBILE']} | OUT: {counts['OUT']}"
            cv2.rectangle(annotated, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(annotated, text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if save_path:
                cv2.imwrite(str(save_path), annotated)
                print(f"✓ Saved annotated image to: {save_path}")
            
            if show:
                cv2.imshow('Detection', annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return result
    
    def detect_video(self, video_path, output_path=None, show=False, process_every_n_frames=2, resize_width=640, 
                     roi_line=None, roi_config_file=None, enable_tracking=True):
        """
        Detect objects in a video with optional ROI line-crossing counting
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show: Whether to display the result in real-time
            process_every_n_frames: Process every N frames (default: 2, process every 2nd frame for speed)
            resize_width: Resize frame width for faster processing (default: 640)
            roi_line: ROI line coordinates as dict {'y': y_position} for horizontal line or 
                     {'x': x_position} for vertical line, or {'line_points': [(x1,y1), (x2,y2)]} for custom line.
            roi_config_file: Path to JSON file with ROI configuration (from setup_roi.py)
            enable_tracking: Enable object tracking for IN/OUT counting (default: True)
            
        Returns:
            Dictionary with detection statistics including IN/OUT counts
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate resize dimensions
        resize_height = height
        scale_factor = 1.0
        if resize_width and resize_width < width:
            resize_height = int(height * (resize_width / width))
            scale_factor = resize_width / width
            print(f"  Resizing frames from {width}x{height} to {resize_width}x{resize_height} for faster processing")
        
        # Load ROI configuration from file if provided
        if roi_config_file:
            with open(roi_config_file, 'r') as f:
                roi_config = json.load(f)
                config_type = roi_config.get('type', 'custom')
                
                if config_type == 'horizontal':
                    y_pos = roi_config.get('y')
                    roi_line = {'y': int(y_pos * scale_factor)}
                    print(f"  Loaded horizontal ROI line at Y={y_pos} from: {roi_config_file}")
                elif config_type == 'vertical':
                    x_pos = roi_config.get('x')
                    roi_line = {'x': int(x_pos * scale_factor)}
                    print(f"  Loaded vertical ROI line at X={x_pos} from: {roi_config_file}")
                elif config_type == 'custom' or 'line_points' in roi_config:
                    line_points = roi_config.get('line_points')
                    if line_points:
                        # Scale points if video is resized
                        p1 = (int(line_points[0][0] * scale_factor), int(line_points[0][1] * scale_factor))
                        p2 = (int(line_points[1][0] * scale_factor), int(line_points[1][1] * scale_factor))
                        roi_line = {'line_points': [p1, p2]}
                        print(f"  Loaded custom ROI line from: {roi_config_file}")
        
        # Set up ROI line (default: horizontal line at center)
        if roi_line is None:
            roi_line = {'y': resize_height // 2 if resize_width and resize_width < width else height // 2}
        
        # Determine line type
        if 'line_points' in roi_line:
            # Custom line with two points
            is_custom_line = True
            is_horizontal = False
            line_pos = None
            line_p1, line_p2 = roi_line['line_points']
        else:
            # Simple horizontal or vertical line
            is_custom_line = False
            is_horizontal = 'y' in roi_line
            line_pos = roi_line.get('y') if is_horizontal else roi_line.get('x')
            line_p1, line_p2 = None, None
        
        print(f"\nProcessing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        if enable_tracking:
            if is_custom_line:
                print(f"  ROI Line: Custom line from {line_p1} to {line_p2}")
            else:
                print(f"  ROI Line: {'Horizontal' if is_horizontal else 'Vertical'} at {line_pos}")
            print(f"  Tracking enabled for IN/OUT counting")
        
        if show:
            # Test if display is available
            try:
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_img)
                cv2.waitKey(1)
                cv2.destroyWindow('test')
                display_available = True
            except:
                print(f"\n⚠️  Display not available (headless/WSL environment)")
                print(f"  Switching to background mode automatically")
                show = False
                display_available = False
            
            if display_available:
                print(f"\n📺 LIVE PREVIEW MODE")
                print(f"  Press 'Q' to stop processing early")
                print(f"  Press 'P' to pause/resume")
                print(f"  Processing will continue in background while saving video")
        
        # Initialize video writer
        writer = None
        output_size = (resize_width, resize_height) if resize_width and resize_width < width else (width, height)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
        
        # Initialize tracker for IN/OUT counting
        tracker = CentroidTracker(max_disappeared=30) if enable_tracking else None
        in_count = 0
        out_count = 0
        counted_ids = set()  # Track which objects have been counted already
        
        # Process video
        frame_count = 0
        processed_count = 0
        total_counts = defaultdict(int)
        frame_results = []
        paused = False  # Pause state for live preview
        
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % process_every_n_frames != 0:
                continue
            
            processed_count += 1
            
            # Resize frame for faster inference
            if resize_width and resize_width < width:
                frame_resized = cv2.resize(frame, (resize_width, resize_height))
            else:
                frame_resized = frame
            
            # Run inference on frame
            results = self.model(
                frame_resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            # Count detections and collect bounding boxes for tracking
            frame_counts = {'MOBILE': 0, 'OUT': 0}
            detections_for_tracking = []
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                frame_counts[class_name] += 1
                total_counts[class_name] += 1
                
                # Collect bounding boxes for tracking (only track OUT for IN/OUT counting)
                if enable_tracking and class_name == 'OUT':
                    bbox = box.xyxy[0].cpu().numpy()
                    detections_for_tracking.append(bbox)
            
            # Update tracker and detect line crossings
            if enable_tracking and tracker is not None:
                objects = tracker.update(detections_for_tracking)
                
                # Check line crossings for each tracked object
                for object_id, centroid in objects.items():
                    cx, cy = centroid
                    
                    # Initialize tracking data if new object
                    if object_id not in tracker.crossed:
                        tracker.crossed[object_id] = {'crossed': False, 'direction': None, 'start_side': None}
                    
                    # Determine which side of the line the object is on
                    if is_custom_line:
                        # For custom line, use cross product to determine side
                        # Vector from p1 to p2
                        v1 = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
                        # Vector from p1 to centroid
                        v2 = (cx - line_p1[0], cy - line_p1[1])
                        # Cross product
                        cross = v1[0] * v2[1] - v1[1] * v2[0]
                        current_side = 'left' if cross > 0 else 'right'
                    elif is_horizontal:
                        current_side = 'top' if cy < line_pos else 'bottom'
                    else:
                        current_side = 'left' if cx < line_pos else 'right'
                    
                    # Set initial side if not set
                    if tracker.crossed[object_id]['start_side'] is None:
                        tracker.crossed[object_id]['start_side'] = current_side
                    
                    # Check if crossed and hasn't been counted yet
                    if object_id not in counted_ids:
                        start_side = tracker.crossed[object_id]['start_side']
                        
                        # Detect crossing
                        if is_custom_line or not is_horizontal:
                            # Custom line or vertical: left to right = OUT, right to left = IN
                            if start_side == 'left' and current_side == 'right':
                                out_count += 1
                                counted_ids.add(object_id)
                                tracker.crossed[object_id]['crossed'] = True
                                tracker.crossed[object_id]['direction'] = 'out'
                            elif start_side == 'right' and current_side == 'left':
                                in_count += 1
                                counted_ids.add(object_id)
                                tracker.crossed[object_id]['crossed'] = True
                                tracker.crossed[object_id]['direction'] = 'in'
                        else:
                            # Horizontal line: top to bottom = OUT, bottom to top = IN
                            if start_side == 'top' and current_side == 'bottom':
                                out_count += 1
                                counted_ids.add(object_id)
                                tracker.crossed[object_id]['crossed'] = True
                                tracker.crossed[object_id]['direction'] = 'out'
                            elif start_side == 'bottom' and current_side == 'top':
                                in_count += 1
                                counted_ids.add(object_id)
                                tracker.crossed[object_id]['crossed'] = True
                                tracker.crossed[object_id]['direction'] = 'in'
            
            frame_results.append({
                'frame': frame_count,
                'counts': frame_counts.copy(),
                'in_count': in_count,
                'out_count': out_count
            })
            
            # Annotate frame
            annotated = results.plot()
            
            # Draw ROI line
            if is_custom_line:
                cv2.line(annotated, line_p1, line_p2, (0, 255, 255), 3)
                # Draw arrows to show direction
                mid_x = (line_p1[0] + line_p2[0]) // 2
                mid_y = (line_p1[1] + line_p2[1]) // 2
                cv2.circle(annotated, line_p1, 8, (0, 255, 0), -1)  # Start point (IN side)
                cv2.circle(annotated, line_p2, 8, (0, 0, 255), -1)  # End point (OUT side)
                cv2.putText(annotated, "IN", (line_p1[0] - 30, line_p1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated, "OUT", (line_p2[0] + 10, line_p2[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif is_horizontal:
                cv2.line(annotated, (0, line_pos), (annotated.shape[1], line_pos), (0, 255, 255), 3)
                cv2.putText(annotated, "ROI LINE", (annotated.shape[1] - 150, line_pos - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.line(annotated, (line_pos, 0), (line_pos, annotated.shape[0]), (0, 255, 255), 3)
                cv2.putText(annotated, "ROI", (line_pos + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw tracked objects with IDs
            if enable_tracking and tracker is not None:
                for object_id, centroid in objects.items():
                    cx, cy = centroid
                    # Draw centroid
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                    # Draw ID
                    text_id = f"ID:{object_id}"
                    cv2.putText(annotated, text_id, (cx - 20, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add count text
            if enable_tracking:
                text = f"Frame {frame_count}/{total_frames} | IN: {in_count} | OUT: {out_count}"
                cv2.rectangle(annotated, (10, 10), (700, 50), (0, 0, 0), -1)
                cv2.putText(annotated, text, (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                text = f"Frame {frame_count}/{total_frames} | MOBILE: {frame_counts['MOBILE']} | OUT: {frame_counts['OUT']}"
                cv2.rectangle(annotated, (10, 10), (700, 50), (0, 0, 0), -1)
                cv2.putText(annotated, text, (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            if writer:
                writer.write(annotated)
            
            # Show frame with pause/play controls
            if show:
                # Add pause status if paused
                if paused:
                    cv2.rectangle(annotated, (10, 60), (250, 90), (0, 0, 0), -1)
                    cv2.putText(annotated, "PAUSED - Press 'P'", (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow('Live Detection', annotated)
                
                # Wait time: 1ms if not paused, indefinite if paused
                wait_time = 0 if paused else 1
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
                    status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                    print(f"\r{status}", end='', flush=True)
            
            # Progress update
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed
                print(f"  Progress: {progress:.1f}% | Processed {processed_count} frames | FPS: {fps_proc:.1f}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        avg_fps = processed_count / elapsed_time
        
        # Calculate statistics
        avg_mobile = total_counts['MOBILE'] / processed_count if processed_count > 0 else 0
        avg_out = total_counts['OUT'] / processed_count if processed_count > 0 else 0
        
        result = {
            'video': str(video_path),
            'total_frames': frame_count,
            'processing_time': elapsed_time,
            'avg_fps': avg_fps,
            'total_counts': dict(total_counts),
            'average_per_frame': {
                'MOBILE': avg_mobile,
                'OUT': avg_out
            },
            'frame_results': frame_results
        }
        
        # Add IN/OUT counting results if tracking was enabled
        if enable_tracking:
            result['line_crossing'] = {
                'in_count': in_count,
                'out_count': out_count,
                'total_crossings': in_count + out_count,
                'roi_line': roi_line
            }
        
        print(f"\n✓ Video processing complete")
        
        if enable_tracking:
            print(f"  LINE CROSSING COUNTS:")
            print(f"    IN:  {in_count}")
            print(f"    OUT: {out_count}")
            print(f"    Total Crossings: {in_count + out_count}")
        
        print(f"  Total detections - MOBILE: {total_counts['MOBILE']}, OUT: {total_counts['OUT']}")
        print(f"  Average per frame - MOBILE: {avg_mobile:.2f}, OUT: {avg_out:.2f}")
        print(f"  Processing FPS: {avg_fps:.2f}")
        
        if output_path:
            print(f"  Output saved to: {output_path}")
        
        return result
    
    def detect_batch(self, image_dir, output_dir=None):
        """
        Detect objects in a batch of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save annotated images (optional)
            
        Returns:
            List of detection results
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nProcessing {len(image_files)} images from {image_dir}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_counts = {'MOBILE': 0, 'OUT': 0}
        
        for i, img_path in enumerate(image_files, 1):
            save_path = None
            if output_dir:
                save_path = output_dir / img_path.name
            
            result = self.detect_image(img_path, save_path=save_path)
            results.append(result)
            
            total_counts['MOBILE'] += result['counts']['MOBILE']
            total_counts['OUT'] += result['counts']['OUT']
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(image_files)} images")
        
        print(f"\n✓ Batch processing complete")
        print(f"  Total detections - MOBILE: {total_counts['MOBILE']}, OUT: {total_counts['OUT']}")
        print(f"  Average per image - MOBILE: {total_counts['MOBILE']/len(image_files):.2f}, OUT: {total_counts['OUT']/len(image_files):.2f}")
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11 MOBILE and OUT Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (best.pt)')
    parser.add_argument('--source', type=str, required=True, help='Path to image/video/directory')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--show', action='store_true', help='Show live detection preview (real-time display)')
    parser.add_argument('--no-show', action='store_true', help='Disable live preview (process in background only)')
    parser.add_argument('--save-json', type=str, help='Save results to JSON file')
    parser.add_argument('--roi-y', type=int, help='Horizontal ROI line Y position (for counting IN/OUT)')
    parser.add_argument('--roi-x', type=int, help='Vertical ROI line X position (for counting IN/OUT)')
    parser.add_argument('--roi-config', type=str, help='Path to ROI config JSON file (from setup_roi.py)')
    parser.add_argument('--no-tracking', action='store_true', help='Disable object tracking for IN/OUT counting')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MobileOutDetector(args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    
    source_path = Path(args.source)
    
    # Set up ROI line
    roi_line = None
    if args.roi_y is not None:
        roi_line = {'y': args.roi_y}
    elif args.roi_x is not None:
        roi_line = {'x': args.roi_x}
    
    enable_tracking = not args.no_tracking
    
    # Determine if live preview should be shown (default: True for videos, unless --no-show is used)
    show_live = not args.no_show  # Show by default, disable only if --no-show is specified
    if args.show:  # If --show is explicitly set, enable it
        show_live = True
    
    # Determine source type and process
    if source_path.is_file():
        # Check if video or image
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video - show live preview by default
            results = detector.detect_video(
                source_path, 
                output_path=args.output, 
                show=show_live,
                roi_line=roi_line,
                roi_config_file=args.roi_config,
                enable_tracking=enable_tracking
            )
        else:
            # Image
            results = detector.detect_image(source_path, save_path=args.output, show=args.show)
    elif source_path.is_dir():
        # Batch processing
        results = detector.detect_batch(source_path, output_dir=args.output)
    else:
        raise ValueError(f"Invalid source: {args.source}")
    
    # Save JSON results
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
