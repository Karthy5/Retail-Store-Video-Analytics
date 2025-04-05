# ultimate.py
# (Version with bottle counting, local display, MongoDB closing, local heatmaps,
#  conditionally logged bottle positions, AND conditionally logged person positions - FULL CODE)

import cv2
import numpy as np
import time
import argparse
import os
from collections import OrderedDict, deque
from scipy.spatial import distance as dist
import mediapipe as mp
import torch
import torch.nn as nn
from datetime import datetime, timezone, timedelta # Import timedelta
from ultralytics import YOLO
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, InvalidOperation
import math # Needed for rounding

# --- Configuration ---
TARGET_FRAMES = 75
NUM_LANDMARKS = 33
NUM_COORDS = 3
ACTION_CLASSES = ["reaching", "standing", "walking"]
CAMERA_ID = "StoreCam_01"
TARGET_OBJECT_LABEL = "bottle"

# --- Heatmap Configuration ---
HEATMAP_ALPHA = 0.5
HEATMAP_RADIUS = 15
HEATMAP_BLUR_KSIZE = (21, 21)
HEATMAP_COLORMAP = cv2.COLORMAP_JET

# --- Logging Configuration ---
BOTTLE_POS_ROUNDING_BASE = 10 # Log change if bottle center moves by ~10 pixels
PERSON_POS_LOG_INTERVAL_SECONDS = 10 # Log person position every 10 seconds

# --- MongoDB Connection ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "retail_analytics"
COLLECTION_NAME = "customer_events"

# --- Global variable for MongoDB client ---
mongo_client = None
db_collection = None

# --- MongoDB Connection Function ---
def connect_to_mongodb():
    global mongo_client, db_collection
    try:
        print(f"Attempting to connect to MongoDB at {MONGO_URI}...")
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        db = mongo_client[DB_NAME]
        db_collection = db[COLLECTION_NAME]
        # Ensure indexes exist
        db_collection.create_index([("timestamp", -1)])
        db_collection.create_index([("customer_id", 1)])
        db_collection.create_index([("event_type", 1)])
        db_collection.create_index([("camera_id", 1)])
        print(f"Successfully connected to MongoDB: DB='{DB_NAME}', Collection='{COLLECTION_NAME}'")
        return True
    except ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
        mongo_client = None; db_collection = None; return False
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        mongo_client = None; db_collection = None; return False

# --- Event Logging Function ---
def log_event_to_db(event_data):
    global db_collection
    if db_collection is None: return

    if 'timestamp' not in event_data:
        event_data['timestamp'] = datetime.now(timezone.utc)
    elif isinstance(event_data['timestamp'], datetime) and event_data['timestamp'].tzinfo is None:
         event_data['timestamp'] = event_data['timestamp'].astimezone(timezone.utc)

    if 'camera_id' not in event_data:
        event_data['camera_id'] = CAMERA_ID

    try:
        db_collection.insert_one(event_data)
    except Exception as e:
        print(f"Error logging event to MongoDB: {e}")
        print(f"Failed Event Data: {event_data}")


# --- Centroid Tracker (MODIFIED for conditional person position logging) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = OrderedDict() # {objectID: (centroid, bbox)}
        self.disappeared = OrderedDict() # {objectID: disappear_count}
        self.maxDisappeared = maxDisappeared
        # ***** NEW: Store last log time for person position *****
        self.last_position_log_time = {} # {objectID: timestamp}
        self.log_interval = timedelta(seconds=PERSON_POS_LOG_INTERVAL_SECONDS)

    def register(self, centroid, bbox, current_bottle_count):
        objectID = self.nextObjectID
        self.objects[objectID] = (centroid, bbox)
        self.disappeared[objectID] = 0
        self.nextObjectID += 1
        print(f"Registered new person ID: {objectID}")

        # Log entry event
        log_event_to_db({
            "event_type": "entry", "customer_id": objectID,
            "timestamp": datetime.now(timezone.utc), "position": [int(c) for c in centroid],
            "bottle_count_at_entry": current_bottle_count })

        # ***** Log the *first* position update AND record the time *****
        now = datetime.now(timezone.utc)
        log_event_to_db({
            "event_type": "position_update", "customer_id": objectID,
            "timestamp": now, "position": [int(c) for c in centroid] })
        self.last_position_log_time[objectID] = now # Store log time

    def deregister(self, objectID, current_bottle_count):
        print(f"Deregistering person ID: {objectID} (disappeared {self.disappeared.get(objectID, '?')} frames)")

        # Log exit event
        log_event_to_db({
            "event_type": "exit", "customer_id": objectID,
            "timestamp": datetime.now(timezone.utc),
            "bottle_count_at_exit": current_bottle_count })

        # Clean up tracked data
        if objectID in self.objects: del self.objects[objectID]
        if objectID in self.disappeared: del self.disappeared[objectID]
        # ***** Clean up log time tracker *****
        if objectID in self.last_position_log_time:
            del self.last_position_log_time[objectID]

    def update(self, person_rects, current_bottle_count):
        if len(person_rects) == 0:
            ids_to_deregister = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                     ids_to_deregister.append(objectID)
            for oid in ids_to_deregister: self.deregister(oid, current_bottle_count)
            return self.objects

        inputCentroids = np.array([(int(x + w / 2.0), int(y + h / 2.0)) for x, y, w, h in person_rects], dtype="int")

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], person_rects[i], current_bottle_count)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array([data[0] for data in self.objects.values()])
            D = dist.cdist(objectCentroids, inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue

                objectID = objectIDs[row]
                new_centroid = inputCentroids[col]
                self.objects[objectID] = (new_centroid, person_rects[col])
                self.disappeared[objectID] = 0

                # ***** Conditional Position Update Logging *****
                now = datetime.now(timezone.utc)
                last_log = self.last_position_log_time.get(objectID)

                # Log if enough time has passed since the last log for this ID
                if last_log is None or (now - last_log >= self.log_interval):
                    # print(f"Logging position update for ID {objectID} (Interval: {self.log_interval})") # Optional debug print
                    log_event_to_db({
                        "event_type": "position_update",
                        "customer_id": objectID,
                        "timestamp": now,
                        "position": [int(c) for c in new_centroid]
                    })
                    # Update the last log time for this ID
                    self.last_position_log_time[objectID] = now
                # ***** End Conditional Logging *****

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            ids_to_deregister = []
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    ids_to_deregister.append(objectID)
            # Pass current bottle count when deregistering
            for oid in ids_to_deregister: self.deregister(oid, current_bottle_count)

            # Register new objects (will log their first position update)
            for col in unusedCols:
                self.register(inputCentroids[col], person_rects[col], current_bottle_count)

        return self.objects


# --- Action Recognition Model Definitions ---
class EnhancedTSM(nn.Module):
    def __init__(self, num_joints=NUM_LANDMARKS, embed_dim=64, num_coords=NUM_COORDS):
        super(EnhancedTSM, self).__init__()
        self.joint_encoder = nn.Sequential(nn.Linear(num_coords, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        self.temporal_fc = nn.Linear(num_joints * embed_dim, embed_dim)
    def forward(self, x):
        B, T, J, C = x.shape; x_r = x.reshape(-1, C); x_e = self.joint_encoder(x_r); x = x_e.view(B, T, J, -1); x_f = x.view(B, T, -1); tc = J*x.shape[-1]; sc = max(1,tc//8); zp = torch.zeros(B,1,sc,device=x.device,dtype=x.dtype); fwd=torch.cat((x_f[:,1:,:sc],zp),dim=1); bwd=torch.cat((zp,x_f[:,:-1,sc:2*sc]),dim=1); uc=x_f[:,:,2*sc:]; xt=torch.cat((fwd,bwd,uc),dim=2); x=self.temporal_fc(xt); return x.mean(dim=1)

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, embed_dim=64):
        super().__init__(); self.tsm_feature_extractor = EnhancedTSM(embed_dim=embed_dim); self.dropout = nn.Dropout(0.5); self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x): x = self.tsm_feature_extractor(x); x = self.dropout(x); x = self.classifier(x); return x

# --- Helper Functions ---
def load_yolov8(model_path='yolov8s.pt', device='cpu'):
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"YOLOv8 model '{model_path}' loaded successfully on {device}.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv8 model '{model_path}': {e}")
        return None

def detect_objects_yolov8(model, frame, conf_threshold=0.4, iou_threshold=0.5, target_classes=['person', 'bottle']):
    boxes, confidences, class_names = [], [], [] # Return confidences now
    if model is None: return boxes, confidences, class_names

    target_class_indices = None
    if target_classes and hasattr(model, 'names') and isinstance(model.names, dict):
        try:
            target_class_indices = [i for i, name in model.names.items() if name in target_classes]
            if not target_class_indices:
                 print(f"Warning: None of the target classes {target_classes} found in model classes: {list(model.names.values())}")
        except Exception as e:
             print(f"Warning: Could not get class indices from model.names: {e}")
    elif not hasattr(model, 'names') or not isinstance(model.names, dict):
         print("Warning: Cannot filter by class name - model.names attribute missing or not a dictionary.")

    try:
        # Perform prediction, filtering by class index if possible
        results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, classes=target_class_indices, verbose=False)

        if isinstance(results, list) and results:
            result = results[0] # Get the first result object
            if result.boxes:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy() # Bounding box coordinates
                    conf = box.conf[0].item()       # Confidence score
                    cls_idx = int(box.cls[0].item())  # Class index

                    # Get class name safely
                    class_name = model.names.get(cls_idx, f"Unknown_{cls_idx}")

                    # Optional secondary filter (if predict didn't filter perfectly)
                    if target_classes and class_name not in target_classes:
                        continue

                    xmin, ymin, xmax, ymax = xyxy
                    # Append box in x, y, w, h format
                    boxes.append([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)])
                    confidences.append(conf) # Append confidence
                    class_names.append(class_name)
    except Exception as e:
        print(f"Error during YOLOv8 prediction: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    return boxes, confidences, class_names

def load_face_detector(prototxt, weights):
    if not os.path.exists(prototxt): print(f"Error: Face prototxt file not found at {prototxt}"); return None
    if not os.path.exists(weights): print(f"Error: Face weights file not found at {weights}"); return None
    try:
        face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        print("Face detector loaded successfully.")
        return face_net
    except cv2.error as e:
        print(f"Error loading face detector: {e}")
        return None

def detect_faces(face_net, frame, conf_threshold=0.6):
    faces = []
    if face_net is None: return faces
    try:
        h_roi, w_roi = frame.shape[:2]
        if h_roi == 0 or w_roi == 0: return faces # Avoid error on empty ROI
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w_roi, h_roi, w_roi, h_roi])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_roi - 1, x2), min(h_roi - 1, y2)
                if x2 > x1 and y2 > y1: # Check for valid box dimensions
                    faces.append((x1, y1, x2 - x1, y2 - y1)) # Return x, y, w, h
    except Exception as e:
        print(f"Error during face detection: {e}")
    return faces

def compute_iou(boxA, boxB): # Requires box format [x, y, w, h]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union
    if boxAArea <= 0 or boxBArea <= 0: return 0.0
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0: return 0.0

    iou = interArea / denominator
    return iou

def preprocess_action_buffer(keypoint_buffer_deque, device): # Only needed if action model loaded
     buffer_list = list(keypoint_buffer_deque)
     processed_list = []
     expected_shape = (NUM_LANDMARKS, NUM_COORDS)
     for item in buffer_list:
         if isinstance(item, np.ndarray) and item.shape == expected_shape:
             processed_list.append(item)
         else:
             processed_list.append(np.zeros(expected_shape, dtype=np.float32))
     # Ensure correct length
     if len(processed_list) != TARGET_FRAMES:
         if len(processed_list) < TARGET_FRAMES:
             padding = [np.zeros(expected_shape, dtype=np.float32)] * (TARGET_FRAMES - len(processed_list))
             processed_list.extend(padding)
         else:
             processed_list = processed_list[:TARGET_FRAMES]

     np_buffer = np.array(processed_list, dtype=np.float32)
     return torch.from_numpy(np_buffer).unsqueeze(0).to(device) # Add batch dim

def round_to_nearest(n, base):
    """Rounds number n to the nearest multiple of base."""
    # Handle potential division by zero if base is 0, though it shouldn't be
    if base == 0: return n
    return base * round(n / base)

def update_heatmap(heatmap_grid, points, radius):
    """Adds 'heat' to the grid at specified points."""
    h, w = heatmap_grid.shape
    for (x, y) in points:
        # Ensure coordinates are within bounds
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            # Define the bounding box for the circle/square area
            xmin = max(0, x - radius)
            xmax = min(w, x + radius + 1)
            ymin = max(0, y - radius)
            ymax = min(h, y + radius + 1)
            # Increment the region
            heatmap_grid[ymin:ymax, xmin:xmax] += 1
            # Optional: Use cv2.circle to make it round? Might be slower.
            # cv2.circle(heatmap_grid, (x, y), radius, 1, thickness=-1) # Adds 1 in a circle
    return heatmap_grid

def draw_heatmap_on_frame(frame, heatmap_grid, alpha, blur_ksize, colormap):
    """Generates and overlays the heatmap onto the frame."""
    if np.max(heatmap_grid) == 0: # Avoid processing if grid is empty
        return frame

    # Blur and normalize
    heatmap_blurred = cv2.GaussianBlur(heatmap_grid, blur_ksize, 0)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)

    # Blend with original frame
    output_frame = cv2.addWeighted(heatmap_colored, alpha, frame, 1 - alpha, 0)
    return output_frame


# --- Main Processing Loop ---
def main(args):
    global mongo_client, db_collection

    # --- Device Setup ---
    if args.device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: device = args.device
    if device == 'cuda' and not torch.cuda.is_available(): print("CUDA specified but not available, falling back to CPU."); device = 'cpu'
    print(f"Using device: {device}")

    # --- Connect to MongoDB ---
    if not connect_to_mongodb(): print("Exiting due to MongoDB connection failure."); return

    # --- Load Models ---
    yolo_model = load_yolov8(model_path=args.yolo_model, device=device)
    if yolo_model is None: return
    face_net = load_face_detector(args.face_prototxt, args.face_weights)
    action_model = None; mp_pose_instance = None; action_buffers = {}
    if args.action_model_path and os.path.exists(args.action_model_path):
        try:
            num_action_classes = len(ACTION_CLASSES)
            action_model = ActionRecognitionModel(num_classes=num_action_classes)
            state_dict = torch.load(args.action_model_path, map_location=device)
            classifier_key = next((k for k in state_dict if 'classifier.bias' in k), None)
            if classifier_key:
                 loaded_classes = state_dict[classifier_key].shape[0]
                 if loaded_classes != num_action_classes:
                      print(f"FATAL WARNING: Action model classes mismatch ({loaded_classes} vs {num_action_classes}). Update ACTION_CLASSES list.")
                      action_model = None
                 else:
                      action_model.load_state_dict(state_dict); action_model.to(device); action_model.eval()
                      print(f"Loaded action model '{args.action_model_path}'.")
                      try:
                          mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles
                          mp_pose = mp.solutions.pose
                          mp_pose_instance = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                          print("MediaPipe Pose initialized for action recognition.")
                      except Exception as e: print(f"Error initializing MediaPipe Pose: {e}. Action recognition disabled."); action_model = None; mp_pose_instance = None
            else:
                print(f"Warning: Action model structure not verified. Loading anyway."); action_model.load_state_dict(state_dict); action_model.to(device); action_model.eval()
                print(f"Loaded action model '{args.action_model_path}'.")
                try:
                     mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles; mp_pose = mp.solutions.pose
                     mp_pose_instance = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5); print("MediaPipe Pose initialized.")
                except Exception as e: print(f"Error initializing MediaPipe Pose: {e}."); action_model=None; mp_pose_instance = None
        except Exception as e: print(f"Error loading action model: {e}"); traceback.print_exc(); action_model = None; mp_pose_instance = None
    else: print(f"Action model path ('{args.action_model_path}') not provided or not found. Action recognition disabled.")


    # --- Video Capture ---
    cap = cv2.VideoCapture(args.input_video if args.input_video else args.cam_index)
    if not cap.isOpened(): print(f"Error: Could not open video source: {args.input_video or args.cam_index}"); return
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video stream opened ({fw}x{fh}).")

    # --- Log Camera Configuration ---
    log_event_to_db({ "event_type": "camera_config", "timestamp": datetime.now(timezone.utc),
                      "camera_id": CAMERA_ID, "frame_width": fw, "frame_height": fh })
    print(f"Logged camera configuration to DB: {fw}x{fh}")

    # --- Initialization ---
    ct = CentroidTracker(maxDisappeared=args.max_disappeared) # Tracker handles timed person logging
    print(f"\nStarting local analysis loop (Counting: {TARGET_OBJECT_LABEL})...")
    frame_count = 0; start_time = time.time(); display_frame = not args.no_display; initial_bottle_count = -1
    person_heatmap_grid = np.zeros((fh, fw), dtype=np.float32)
    bottle_heatmap_grid = np.zeros((fh, fw), dtype=np.float32)
    previous_rounded_bottle_centers = set()

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret: print("End of stream or error reading frame."); break

        # ***** INSERT FLIP CODE HERE *****
        frame = cv2.flip(frame, 1) # 1 for horizontal flip
        # ***** END OF INSERTION *****

        frame_count += 1
        current_timestamp = datetime.now(timezone.utc)

        draw_frame = frame.copy() if display_frame else None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if action_model and mp_pose_instance else None

        # --- Object Detection ---
        boxes, confidences, class_names = detect_objects_yolov8(
            yolo_model, frame, conf_threshold=args.yolo_conf, iou_threshold=args.yolo_iou,
            target_classes=['person', TARGET_OBJECT_LABEL]
        )
        person_boxes = []
        current_detected_bottles_boxes = []; bottle_centers = []; current_frame_bottle_data_for_log = []
        for i, (box, conf, name) in enumerate(zip(boxes, confidences, class_names)):
            (x,y,w,h) = box
            if name == "person":
                person_boxes.append(box)
            elif name == TARGET_OBJECT_LABEL:
                current_detected_bottles_boxes.append(box)
                cx = x + w // 2; cy = y + h // 2
                bottle_centers.append((cx, cy))
                current_frame_bottle_data_for_log.append({
                    "position": [int(cx), int(cy)],
                    "confidence": float(conf),
                    "bbox": [int(x), int(y), int(w), int(h)]
                })
        current_bottle_count = len(current_detected_bottles_boxes)

        # --- Log Initial State (Once) ---
        if initial_bottle_count == -1 and frame_count > 0:
            initial_bottle_count = current_bottle_count
            log_event_to_db({ "event_type": "initial_state", "timestamp": current_timestamp,
                              "initial_bottle_count": initial_bottle_count })
            print(f"Logged initial state: {initial_bottle_count} bottles detected.")

        # --- Person Tracking (Tracker internally handles timed position logging) ---
        tracked_persons = ct.update(person_boxes, current_bottle_count)

        # --- Update Heatmap Grids (Local Display) ---
        if args.show_heatmaps:
            person_points = [centroid for centroid, bbox in tracked_persons.values()]
            person_heatmap_grid = update_heatmap(person_heatmap_grid, person_points, HEATMAP_RADIUS)
            bottle_heatmap_grid = update_heatmap(bottle_heatmap_grid, bottle_centers, HEATMAP_RADIUS)

        # --- Conditional Bottle Logging ---
        current_rounded_bottle_centers = {
            (round_to_nearest(cx, BOTTLE_POS_ROUNDING_BASE), round_to_nearest(cy, BOTTLE_POS_ROUNDING_BASE))
            for cx, cy in bottle_centers
        }
        if current_rounded_bottle_centers != previous_rounded_bottle_centers:
            # print(f"Bottle positions changed (Frame {frame_count}). Logging {len(current_frame_bottle_data_for_log)} bottles.") # Optional debug
            for bottle_data in current_frame_bottle_data_for_log:
                log_event_to_db({
                    "event_type": "bottle_detection", "timestamp": current_timestamp, "camera_id": CAMERA_ID,
                    "position": bottle_data["position"], "confidence": bottle_data["confidence"], "bbox": bottle_data["bbox"]
                })
            previous_rounded_bottle_centers = current_rounded_bottle_centers

        # --- Optional Pose Estimation ---
        full_frame_pose_results = None
        if frame_rgb is not None and mp_pose_instance:
            try:
                frame_rgb.flags.writeable = False
                full_frame_pose_results = mp_pose_instance.process(frame_rgb)
                frame_rgb.flags.writeable = True
            except Exception as e:
                print(f"Error processing MediaPipe Pose: {e}")
                full_frame_pose_results = None

        # --- Draw Heatmaps (if enabled and displaying) ---
        if display_frame and args.show_heatmaps:
            draw_frame = draw_heatmap_on_frame(draw_frame, person_heatmap_grid, HEATMAP_ALPHA, HEATMAP_BLUR_KSIZE, HEATMAP_COLORMAP)
            # Add bottle heatmap overlay if desired:
            # draw_frame = draw_heatmap_on_frame(draw_frame, bottle_heatmap_grid, HEATMAP_ALPHA, HEATMAP_BLUR_KSIZE, cv2.COLORMAP_HOT) # Different color?

        # --- Process Tracked Persons & Draw Overlays (if enabled) ---
        if display_frame:
            # Draw detected bottles FIRST
            for bottle_box in current_detected_bottles_boxes:
                 (x,y,w,h) = bottle_box
                 cv2.rectangle(draw_frame, (x,y), (x+w, y+h), (0,0,255), 2) # Red

            # Process and draw tracked persons
            for objectID, (centroid, bbox) in tracked_persons.items():
                (x,y,w,h)=bbox; x1,y1,x2,y2 = max(0,x),max(0,y),min(fw,x+w),min(fh,y+h)
                label_text_cv = f"ID {objectID}"
                cv2.rectangle(draw_frame,(x1,y1),(x2,y2),(0,255,0),2) # Green person box

                # Optional Face Detection Drawing
                if face_net:
                     if (x2 - x1) > 0 and (y2 - y1) > 0:
                        personROI = frame[y1:y2, x1:x2]
                        faces_in_roi = detect_faces(face_net, personROI, conf_threshold=0.6)
                        for (fx, fy, fw_f, fh_f) in faces_in_roi: # Use different var names
                             cv2.rectangle(draw_frame, (x1 + fx, y1 + fy), (x1 + fx + fw_f, y1 + fy + fh_f), (255, 0, 0), 1)

                # Optional Action Recognition & Logging / Drawing
                action_label_str = ""
                if action_model and mp_pose_instance:
                    keypoints = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
                    pose_landmarks_for_drawing = None
                    if full_frame_pose_results and full_frame_pose_results.pose_landmarks:
                        # Basic pose association check
                        nose = full_frame_pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        nx, ny = int(nose.x * fw), int(nose.y * fh)
                        if x1 <= nx <= x2 and y1 <= ny <= y2:
                             pose_landmarks_for_drawing = full_frame_pose_results.pose_landmarks
                             for idx, lm in enumerate(pose_landmarks_for_drawing.landmark):
                                 if idx < NUM_LANDMARKS: keypoints[idx] = [lm.x, lm.y, lm.z]

                             if objectID not in action_buffers:
                                action_buffers[objectID] = deque(maxlen=TARGET_FRAMES)
                                for _ in range(TARGET_FRAMES): action_buffers[objectID].append(np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32))
                             action_buffers[objectID].append(keypoints)

                             try:
                                 input_tensor = preprocess_action_buffer(action_buffers[objectID], device)
                                 with torch.no_grad():
                                     outputs = action_model(input_tensor)
                                     probs = torch.softmax(outputs, dim=1)
                                     conf, pred_idx = torch.max(probs, 1)
                                     pred_label_idx = pred_idx.item()
                                     if pred_label_idx < len(ACTION_CLASSES):
                                         pred_label = ACTION_CLASSES[pred_label_idx]
                                         pred_conf = conf.item()
                                         action_label_str = f"{pred_label} ({pred_conf:.1f})"
                                         # Log action event (logged every time detected, unlike position)
                                         log_event_to_db({
                                             "event_type": "action", "customer_id": objectID,
                                             "timestamp": current_timestamp, "action_label": pred_label,
                                             "confidence": float(pred_conf)
                                         })
                                     else: action_label_str = "Idx Error"
                             except Exception as e: action_label_str = "Predict Error"

                    if pose_landmarks_for_drawing:
                          mp_drawing.draw_landmarks(
                              draw_frame, pose_landmarks_for_drawing, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                if action_label_str: label_text_cv += f": {action_label_str}"
                cv2.putText(draw_frame, label_text_cv, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Display Frame (if enabled) ---
        if display_frame:
            end_time = time.time(); elapsed = end_time - start_time; fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(draw_frame, f"FPS: {fps:.2f}", (fw - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(draw_frame, f"Bottles: {current_bottle_count}", (fw - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if args.show_heatmaps: cv2.putText(draw_frame, "Heatmap ON", (fw - 120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(f"Local Analysis ({TARGET_OBJECT_LABEL} Count) - Press 'q'", draw_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): print("'q' pressed, stopping analysis loop."); break
        else:
             if frame_count % 100 == 0:
                  end_time = time.time(); elapsed = end_time - start_time; fps = frame_count / elapsed if elapsed > 0 else 0
                  print(f"Processed frame {frame_count}... FPS: {fps:.2f}, Bottles: {current_bottle_count} (Pos logged conditionally)")


    # --- Cleanup ---
    print("\nCleaning up...")
    cap.release();
    if display_frame: cv2.destroyAllWindows()
    if mp_pose_instance: mp_pose_instance.close()
    print("Local analysis finished.")


# --- ArgParse ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Local Analysis: Multi-User Tracking, Bottle Counting & Heatmaps")
    parser.add_argument('--input_video', type=str, default=None, help="Path to input video file. If None, uses webcam.")
    parser.add_argument('--cam_index', type=int, default=0, help="Webcam index (if --input_video is not set).")
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help="Path to YOLOv8 model file (e.g., yolov8n.pt).")
    parser.add_argument('--action_model_path', type=str, default=None, help="Optional: Path to action recognition model file.")
    parser.add_argument('--face_prototxt', type=str, default='models/deploy.prototxt', help="Optional: Path to Caffe face detector prototxt.")
    parser.add_argument('--face_weights', type=str, default='models/res10_300x300_ssd_iter_140000.caffemodel', help="Optional: Path to Caffe face detector weights.")
    parser.add_argument('--yolo_conf', type=float, default=0.4, help="YOLO detection confidence threshold.")
    parser.add_argument('--yolo_iou', type=float, default=0.5, help="YOLO Non-Maximum Suppression (NMS) IoU threshold.")
    parser.add_argument('--max_disappeared', type=int, default=30, help="Max frames an object can disappear before being deregistered.")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Processing device ('auto', 'cuda', or 'cpu').")
    parser.add_argument('--no_display', action='store_true', help="Run analysis without displaying the video window locally.")
    parser.add_argument('--show_heatmaps', action='store_true', help="Overlay real-time heatmaps on the local display (requires --no_display to be false).")
    args = parser.parse_args()

    if args.show_heatmaps and args.no_display:
        print("Warning: --show_heatmaps requires local display. Heatmaps will not be generated.")
        args.show_heatmaps = False

    models_dir = os.path.dirname(args.face_prototxt)
    if models_dir and not os.path.exists(models_dir):
        try: os.makedirs(models_dir); print(f"Created directory for optional models: {models_dir}")
        except OSError as e: print(f"Warning: Could not create models directory '{models_dir}': {e}")

    # --- Run Main Function with Error Handling and Guaranteed Cleanup ---
    try:
        main(args)
    except Exception as e:
        print("\n--- Unhandled Error During Local Analysis ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print("------------------------------------------")
    finally:
        if mongo_client:
             try:
                 print("Closing MongoDB connection (finally block)...")
                 mongo_client.close()
                 print("MongoDB connection closed successfully.")
             except InvalidOperation:
                 print("Note: MongoDB connection was already closed.")
                 pass
             except Exception as e:
                 print(f"Note: Error occurred during final MongoDB close: {e}")
                 pass
        else:
             print("No active MongoDB connection to close.")