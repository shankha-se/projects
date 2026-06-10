import os
import time
from pathlib import Path

import cv2
import yaml
import face_recognition
from ultralytics import YOLO

class SecurityMLEngine:
    def __init__(self, config_path="config/settings.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent.parent / config_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Layer 1: Instantiate YOLOv8 Object Tracker
        model_path = self.config["ml_settings"]["yolo_model_path"]
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = config_path.parent / model_file
        self.yolo = YOLO(str(model_file.resolve()))

        self.tolerance = self.config["ml_settings"]["face_distance_tolerance"]
        self.conf_threshold = self.config["ml_settings"]["yolo_confidence"]

        self.known_encodings = []
        self.known_names = []
        self.locked_name = None
        self.locked_state = None
        self.unknown_frames = 0
        self.unknown_threshold = 5
        self.scanning_start = None
        self.scan_hold_seconds = self.config["ml_settings"].get("scan_hold_seconds", 5)
        self.locked_alert_state = None
        self.locked_alert_label = None
        self.load_authorized_profiles()

    def load_authorized_profiles(self):
        """Compiles multiple facial reference vectors per identity from the profiles workspace folder."""
        profile_dir = "profiles"
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
            return

        supported_ext = (".jpg", ".jpeg", ".png")
        for file in os.listdir(profile_dir):
            if file.lower().endswith(supported_ext):
                # Extract the base identity name before any underscore 
                # e.g., 'me_1.jpg' -> 'ME', 'wife_front.jpg' -> 'WIFE'
                raw_name = os.path.splitext(file)[0]
                base_identity = raw_name.split('_')[0].upper()
                
                img_path = os.path.join(profile_dir, file)
                try:
                    loaded_img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(loaded_img)
                    
                    if encodings:
                        # Append the vector map
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(base_identity)
                        print(f"[ML ENGINE] Registered vector facet for {base_identity} from file: {file}")
                    else:
                        print(f"[ML ENGINE] ⚠️ Warning: No clear face detected in profile asset: {file}")
                except Exception as e:
                    print(f"[ML ENGINE] Exception indexing profile image matrix for {file}: {e}")

    def _find_authorized_face(self, encoding):
        """Return the best authorized match for a given face encoding plus its distance score."""
        if not self.known_encodings:
            return None, None

        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_idx, best_distance = min(enumerate(distances), key=lambda item: item[1])
        if best_distance <= self.tolerance:
            return self.known_names[best_idx], best_distance
        return None, best_distance

    def analyze_frame(self, frame):
        """Runs layered classification analysis over incoming video stream frames."""
        alarm_state = "NOMINAL"
        detected_label = "Clear"
        safety_status = "SAFE"

        # Layer 1: Detect object classes using YOLO
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)[0]
        person_seen = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.yolo.names[cls_id]
            # consolidate safe animal classes
            safe_animals = {"cat", "dog", "rabbit", "cow", "horse", "sheep", "goat", "bird", "deer"}
            if label in safe_animals:
                detected_label = f"{label.title()} (Animal Detected)"
                print(f"[ML] OBJECT DETECTED: {label} -> {detected_label}")
                self._reset_person_lock()
                return "NOMINAL", detected_label, "SAFE"

            if label == "bear":
                detected_label = "WILD BEAR DETECTED"
                print(f"[ML] OBJECT DETECTED: bear -> {detected_label}")
                self._reset_person_lock()
                self.locked_alert_state = "CRITICAL_BEAR"
                self.locked_alert_label = detected_label
                return "CRITICAL_BEAR", detected_label, "UNSAFE"

            if label != "person":
                print(f"[ML] OBJECT IGNORED: {label}")
                continue

            person_seen = True
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"[ML] face_locations full frame={len(face_locations)}, face_encodings={len(face_encodings)}")

            if not face_encodings:
                if hasattr(box, 'xyxy'):
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    print(f"[ML] Trying face detection in person ROI: {x1},{y1},{x2},{y2}")
                    crop = rgb_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        face_locations = face_recognition.face_locations(crop)
                        face_encodings = face_recognition.face_encodings(crop, face_locations)
                        print(f"[ML] face_locations crop={len(face_locations)}, face_encodings crop={len(face_encodings)}")

            if not face_encodings:
                self.unknown_frames += 1
                print(f"[ML] PERSON DETECTED: no face encoding found (unknown_frames={self.unknown_frames})")
                if self.locked_state == "NOMINAL" and self.locked_name and self.unknown_frames < self.unknown_threshold:
                    self.scanning_start = None
                    print(f"[ML] AUTHORIZED TRACKING: {self.locked_name} still present, safe")
                    return "NOMINAL", f"Authorized: {self.locked_name} (tracking)", "SAFE"

                if self.scanning_start is None:
                    self.scanning_start = time.time()

                if time.time() - self.scanning_start < self.scan_hold_seconds:
                    detected_label = "Person detected, scanning face..."
                    return "NOMINAL", detected_label, "SAFE"

                self._reset_person_lock()
                detected_label = "Adult Intruder (Face Hidden)"
                print(f"[ML] ESCALATION: no face after hold time -> {detected_label}")
                self.locked_alert_state = "CRITICAL_INTRUDER"
                self.locked_alert_label = detected_label
                return "CRITICAL_INTRUDER", detected_label, "UNSAFE"

            for idx, encoding in enumerate(face_encodings):
                matched_name, match_score = self._find_authorized_face(encoding)
                print(f"[ML] PERSON FACE[{idx}] detected, best match score={match_score:.4f}, threshold={self.tolerance}")
                if matched_name:
                    self.locked_name = matched_name
                    self.locked_state = "NOMINAL"
                    self.unknown_frames = 0
                    self.scanning_start = None
                    self._reset_alert_lock()
                    known_type = "Kid Known" if "KID" in matched_name.upper() or "CHILD" in matched_name.upper() else "Adult Known"
                    detected_label = f"{known_type}: {matched_name}"
                    print(f"[ML] AUTHORIZED FACE MATCHED: {detected_label} (score={match_score:.4f})")
                    return "NOMINAL", detected_label, "SAFE"

            self.unknown_frames += 1
            print(f"[ML] PERSON FACE UNKNOWN: unknown_frames={self.unknown_frames}")
            if self.locked_state == "NOMINAL" and self.locked_name and self.unknown_frames < self.unknown_threshold:
                print(f"[ML] AUTHORIZED CONTEXT: previously known {self.locked_name}, keep tracking")
                return "NOMINAL", f"Authorized: {self.locked_name} (tracking)", "SAFE"

            if self.scanning_start is None:
                self.scanning_start = time.time()

            if time.time() - self.scanning_start < self.scan_hold_seconds:
                detected_label = "Person detected, verifying identity..."
                return "NOMINAL", detected_label, "SAFE"

            self._reset_person_lock()
            detected_label = "Adult Intruder"
            print(f"[ML] ESCALATION: face unknown after hold time -> {detected_label}")
            self.locked_alert_state = "CRITICAL_INTRUDER"
            self.locked_alert_label = detected_label
            return "CRITICAL_INTRUDER", detected_label, "UNSAFE"

        if not person_seen and self.locked_state == "NOMINAL" and self.locked_name and self.unknown_frames < self.unknown_threshold:
            self.unknown_frames += 1
            return "NOMINAL", f"Authorized: {self.locked_name} (tracking)", "SAFE"

        self._reset_person_lock()
        return alarm_state, detected_label, safety_status

    def _reset_person_lock(self):
        self.locked_name = None
        self.locked_state = None
        self.unknown_frames = 0
        self.scanning_start = None

    def _reset_alert_lock(self):
        self.locked_alert_state = None
        self.locked_alert_label = None

    def clear_locks(self):
        self._reset_person_lock()
        self._reset_alert_lock()
