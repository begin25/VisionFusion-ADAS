import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

class VisionFusionADAS:
    def __init__(self, yolo_model='yolov8n.pt'):
        print("[INFO] Booting VisionFusion Engine (YOLOv8 Tracking + MiDaS)...")

        # 1. Semantic Layer: YOLOv8
        self.yolo = YOLO(yolo_model)
        self.target_classes = [2, 3, 5, 7] # Car, Motorcycle, Bus, Truck

        # 2. Spatial Layer: MiDaS Depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Compute Device: {self.device}")

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # 3. Anti-Jitter Smoothing Parameters (Exponential Moving Average)
        # 1.0 = No smoothing (flashing colors). 0.1 = Max smoothing (laggy).
        self.alpha_dist = 0.3  # Smooths the frame-to-frame depth bouncing
        self.alpha_vel = 0.15  # Heavily smooths the velocity spikes to stabilize TTC

        # Object History Format: {track_id: {'dist': float, 'vel': float, 'time': float}}
        self.object_history = {}

    def process_stream(self, input_path, output_path='fusion_output.mp4'):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"[ERROR] Cannot load video at {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print("[INFO] Processing stream... Press 'q' to terminate.")

        while cap.isOpened():
            current_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # --- 1. SPATIAL PERCEPTION (MiDaS) ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transforms(img_rgb).to(self.device)

            with torch.no_grad():
                pred = self.midas(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()

            disparity_map = pred.cpu().numpy()
            distance_map = 10000.0 / (disparity_map + 1.0) # Pseudo-distance proxy

            # --- 2. SEMANTIC TRACKING (YOLOv8) ---
            results = self.yolo.track(frame, classes=self.target_classes, persist=True, verbose=False)[0]

            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            classes = results.boxes.cls.cpu().numpy() if results.boxes else []
            track_ids = results.boxes.id.cpu().numpy() if results.boxes and results.boxes.id is not None else []

            # --- 3. TTC CALCULATION (WITH EMA SMOOTHING) & UI ---
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cls = int(classes[i])
                t_id = int(track_ids[i]) if len(track_ids) > i else -1

                # Extract raw median depth
                obj_depth_region = distance_map[y1:y2, x1:x2]
                if obj_depth_region.size == 0: continue
                raw_dist = np.median(obj_depth_region)

                smoothed_dist = raw_dist
                smoothed_vel = 0.0
                ttc = float('inf')

                # Calculate smoothed physics if we are tracking an object
                if t_id != -1:
                    if t_id in self.object_history:
                        prev_data = self.object_history[t_id]
                        prev_dist = prev_data['dist']
                        prev_vel = prev_data['vel']
                        prev_time = prev_data['time']

                        dt = current_time - prev_time

                        if dt > 0:
                            # Step A: Smooth the distance (absorbs YOLO bounding box jitter)
                            smoothed_dist = (raw_dist * self.alpha_dist) + (prev_dist * (1.0 - self.alpha_dist))

                            # Step B: Calculate raw velocity based on the smoothed distance
                            raw_vel = (prev_dist - smoothed_dist) / dt

                            # Step C: Smooth the velocity (absorbs extreme Time-To-Collision spikes)
                            smoothed_vel = (raw_vel * self.alpha_vel) + (prev_vel * (1.0 - self.alpha_vel))

                            # Step D: Calculate Time-To-Collision using stable variables
                            if smoothed_vel > 0.5: # Object must be consistently moving closer
                                ttc = smoothed_dist / smoothed_vel

                    # Update memory dict for the next frame
                    self.object_history[t_id] = {
                        'dist': smoothed_dist,
                        'vel': smoothed_vel,
                        'time': current_time
                    }

                # --- UI Rendering ---
                ttc_str = f"{ttc:.1f}s" if ttc < 100 else "Safe"

                # Warning Colors (Using the stable smoothed_dist and ttc)
                if ttc < 3.0 or smoothed_dist < 40:
                    color = (0, 0, 255) # Red (CRITICAL)
                elif ttc < 6.0 or smoothed_dist < 80:
                    color = (0, 165, 255) # Orange (WARNING)
                else:
                    color = (0, 255, 0) # Green (SAFE)

                # Draw Bounding Box & Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                class_name = self.yolo.names[cls]
                label = f"{class_name} | TTC: {ttc_str} | Depth: {smoothed_dist:.0f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # --- 4. RENDER VISIBLE MIDAS DEPTH MAP ---
            disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)

            pip_h, pip_w = int(height * 0.25), int(width * 0.25)
            pip_resized = cv2.resize(depth_colormap, (pip_w, pip_h))

            frame[20:20+pip_h, width-pip_w-20:width-20] = pip_resized
            cv2.rectangle(frame, (width-pip_w-20, 20), (width-20, 20+pip_h), (255, 255, 255), 2)
            cv2.putText(frame, "Live MiDaS Depth Radar", (width-pip_w-15, pip_h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Telemetry HUD
            latency = (time.time() - current_time) * 1000
            cv2.putText(frame, f"Latency: {latency:.1f}ms | Trackers Active: {len(track_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            cv2.imshow('VisionFusion ADAS', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Complete! Video saved to {output_path}")

if __name__ == "__main__":
    system = VisionFusionADAS()
    # Ensure your video file is correctly referenced
    system.process_stream('driving_video.mp4', 'vision_fusion_output.mp4')