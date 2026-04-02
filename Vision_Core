import re
import cv2
import mediapipe as mp
import math
import os
import numpy as np

mp_drawing_module = mp.solutions.drawing_utils
mp_drawing_styles_module = mp.solutions.drawing_styles
mp_face_mesh_module = mp.solutions.face_mesh
mp_hands_module = mp.solutions.hands
mp_pose_module = mp.solutions.pose

class VisionCore:
    def __init__(self):
        self.mp_drawing = mp_drawing_module
        self.mp_drawing_styles = mp_drawing_styles_module
        
        # Face Mesh
        self.mp_face_mesh = mp_face_mesh_module
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # Hands
        self.mp_hands = mp_hands_module
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # Pose - model_complexity=2 for accurate full-body scanning
        self.mp_pose = mp_pose_module
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            enable_segmentation=False)

        # Create custom body connections (excluding face landmarks 0-10)
        self.body_connections = frozenset([
            conn for conn in self.mp_pose.POSE_CONNECTIONS 
            if conn[0] > 10 and conn[1] > 10
        ])

        # Face Recognition (LBPH) - uses MediaPipe landmarks for detection
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.known_names = {}
        self.recognition_ready = False
        self.load_known_faces()
        
    def _get_face_bbox_from_landmarks(self, face_landmarks, frame_h, frame_w):
        """Extract a bounding box from MediaPipe face landmarks."""
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        
        x_min = int(min(xs) * frame_w)
        x_max = int(max(xs) * frame_w)
        y_min = int(min(ys) * frame_h)
        y_max = int(max(ys) * frame_h)
        
        # Add some padding (15%)
        pad_x = int((x_max - x_min) * 0.15)
        pad_y = int((y_max - y_min) * 0.15)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame_w, x_max + pad_x)
        y_max = min(frame_h, y_max + pad_y)
        
        return x_min, y_min, x_max - x_min, y_max - y_min

    def process_frame(self, frame, enable_face=True, enable_hands=True, 
                      enable_pose=True, enable_emotion=True, enable_recognition=True):
        if frame is None:
            return None, [], []
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        log_messages = []
        recognition_results = []
        
        # 1. Face Mesh + Emotion + Recognition
        face_results = self.face_mesh.process(image)
        if face_results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for face_landmarks in face_results.multi_face_landmarks:
                if enable_face:
                    pass  # Face detection active but no overlay drawn on face
                            
                if enable_emotion:
                    emotion_results = self.analyze_emotion(face_landmarks)
                    for emotion_log in emotion_results:
                        log_messages.append(emotion_log)
                
                # Face Recognition using MediaPipe bbox
                if enable_recognition and self.recognition_ready:
                    x, y, bw, bh = self._get_face_bbox_from_landmarks(face_landmarks, h, w)
                    if bw > 20 and bh > 20:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Apply CLAHE normalization
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        face_roi = gray[y:y+bh, x:x+bw]
                        
                        if face_roi.size > 0:
                            face_roi = cv2.resize(face_roi, (200, 200))
                            label_id, confidence = self.face_recognizer.predict(face_roi)
                            
                            if confidence < 70:
                                name = self.known_names.get(label_id, "Unknown")
                            else:
                                name = "Unknown"
                            
                            recognition_results.append((name, confidence, (x, y, bw, bh)))

        # 2. Hand Gesture Analysis
        if enable_hands:
            hand_results = self.hands.process(image)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    gesture_log = self.analyze_gesture(hand_landmarks, handedness)
                    if gesture_log:
                        log_messages.append(gesture_log)
                        
        # 3. Full Body Pose Estimation
        if enable_pose:
            pose_results = self.pose.process(image)
            if pose_results.pose_landmarks:
                # Build custom styles to ignore face dots (0-10)
                custom_landmark_spec = {}
                default_spec = self.mp_drawing_styles.get_default_pose_landmarks_style()
                
                # Loop through ALL 33 landmarks to prevent KeyError
                for i in range(33):
                    if i < 11:
                        # Make facial landmarks invisible
                        custom_landmark_spec[i] = self.mp_drawing.DrawingSpec(
                            color=(0, 0, 0), thickness=0, circle_radius=0)
                    else:
                        # Apply standard style to body landmarks
                        if isinstance(default_spec, dict) and i in default_spec:
                            custom_landmark_spec[i] = default_spec[i]
                        else:
                            custom_landmark_spec[i] = self.mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2)

                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.body_connections,
                    landmark_drawing_spec=custom_landmark_spec,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(224, 224, 224), thickness=2))
                log_messages.append("Pose tracked successfully.")
                
        return frame, log_messages, recognition_results

    def analyze_emotion(self, face_landmarks):
        """Analyze facial landmarks to detect emotions and expressions."""
        results = []
        
        def dist(id1, id2):
            p1 = face_landmarks.landmark[id1]
            p2 = face_landmarks.landmark[id2]
            return math.hypot(p1.x - p2.x, p1.y - p2.y)
        
        def get_y(idx):
            return face_landmarks.landmark[idx].y
        
        # --- Key Measurements ---
        left_eye_open = dist(159, 145)
        right_eye_open = dist(386, 374)
        left_eye_width = dist(33, 133)
        right_eye_width = dist(362, 263)
        
        left_ear = left_eye_open / (left_eye_width + 1e-6)
        right_ear = right_eye_open / (right_eye_width + 1e-6)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mouth_open = dist(13, 14)
        face_height = dist(10, 152)
        
        mouth_open_ratio = mouth_open / (face_height + 1e-6)
        
        # Eyebrow pinching/lowering/raising
        left_brow_eye = dist(70, 33)
        right_brow_eye = dist(300, 263)
        avg_brow = (left_brow_eye + right_brow_eye) / 2.0
        brow_ratio = avg_brow / (face_height + 1e-6)
        
        # --- CURVE INDICATOR (Happy/Sad) ---
        lower_lip_y = get_y(14)
        left_corner_y = get_y(61)
        right_corner_y = get_y(291)
        corners_avg_y = (left_corner_y + right_corner_y) / 2.0
        
        curve_indicator = (lower_lip_y - corners_avg_y) / (face_height + 1e-6)
        
        primary_detected = False
        
        # 1. Blink (Highest Priority)
        if avg_ear < 0.12:
            results.append("👁️ Blink Detected")
            primary_detected = True
            
        # 2. Surprised (Mouth open wide AND eyebrows raised)
        elif mouth_open_ratio > 0.07 and brow_ratio > 0.11:
            results.append("😲 Emotion: Surprised!")
            primary_detected = True
            
        # 3. Big Smile (Smile curve + mouth open)
        elif curve_indicator > 0.025 and mouth_open_ratio > 0.04:
            results.append("😄 Emotion: BIG SMILE - Very Happy!")
            primary_detected = True
            
        # 4. Open Mouth (Mouth open, but NOT smiling and NO raised eyebrows)
        elif mouth_open_ratio > 0.06:
            results.append("😮 Mouth Open")
            primary_detected = True
            
        # 5. Regular Smile (Smile curve, mouth mostly closed)
        elif curve_indicator > 0.02:
            results.append("😊 Emotion: Smiling - Happy")
            primary_detected = True
            
        # 6. Sad / Cry (Corners pulled down)
        elif curve_indicator < -0.005:
            if avg_ear < 0.22 and brow_ratio < 0.09:
                results.append("😭 Emotion: Crying / Devastated")
            else:
                results.append("😢 Emotion: Sad")
            primary_detected = True
            
        # 7. Angry (Eyebrows pulled down + mouth closed tight)
        elif brow_ratio < 0.08 and mouth_open_ratio < 0.03:
            results.append("😠 Emotion: Angry")
            primary_detected = True
            
        # 8. Neutral (If nothing else triggers)
        if not primary_detected:
            results.append("😐 Emotion: Neutral")
            
        # --- Wink detection ---
        if avg_ear >= 0.12:
            if left_ear < 0.12 and right_ear > 0.22:
                results.append("😉 Left Wink!")
            elif right_ear < 0.12 and left_ear > 0.22:
                results.append("😉 Right Wink!")
                
        return results

    def analyze_gesture(self, landmarks, handedness):
        label = handedness.classification[0].label
        tips = [4, 8, 12, 16, 20]
        fingers_up = 0
        is_right_hand = (label == "Right")
        
        if is_right_hand:
            if landmarks.landmark[tips[0]].x < landmarks.landmark[tips[0] - 1].x:
                fingers_up += 1
        else:
            if landmarks.landmark[tips[0]].x > landmarks.landmark[tips[0] - 1].x:
                fingers_up += 1
                
        for i in range(1, 5):
            if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
                fingers_up += 1
                
        gesture_name = f"{fingers_up} fingers"
        if fingers_up == 0:
            gesture_name = "Fist / Rock ✊"
        elif fingers_up == 1:
            gesture_name = "Pointing ☝️"
        elif fingers_up == 2:
            gesture_name = "Peace Sign ✌️"
        elif fingers_up == 3:
            gesture_name = "Three Fingers"
        elif fingers_up == 4:
            gesture_name = "Four Fingers"
        elif fingers_up == 5:
            gesture_name = "Open Hand 🖐️"
            
        if (fingers_up == 1 and 
            landmarks.landmark[4].y < landmarks.landmark[3].y and
            landmarks.landmark[8].y > landmarks.landmark[6].y):
            gesture_name = "Thumbs Up 👍"
            
        return f"Gesture [{label} Hand]: {gesture_name}"

    def load_known_faces(self):
        """Load face images from known_faces/ and train LBPH recognizer."""
        faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'known_faces')
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print("[FaceRec] Created known_faces/ folder. Add photos to enable recognition.")
            return

        temp_face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        faces = []
        labels = []
        name_to_label = {}
        next_label = 0

        for filename in os.listdir(faces_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            name = re.sub(r"\d+$", "", os.path.splitext(filename)[0])
            filepath = os.path.join(faces_dir, filename)
            img = cv2.imread(filepath)
            
            if img is None:
                continue

            if name not in name_to_label:
                name_to_label[name] = next_label
                self.known_names[next_label] = name
                next_label += 1
                
            label_id = name_to_label[name]
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mesh_result = temp_face_mesh.process(rgb)
            
            if mesh_result.multi_face_landmarks:
                for fl in mesh_result.multi_face_landmarks:
                    x, y, bw, bh = self._get_face_bbox_from_landmarks(fl, h, w)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    clahe_train = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe_train.apply(gray)
                    
                    if bw > 0 and bh > 0:
                        face_roi = cv2.resize(gray[y:y+bh, x:x+bw], (200, 200))
                        flipped = cv2.flip(face_roi, 1)
                        
                        augmented = [
                            face_roi,
                            flipped,
                            cv2.convertScaleAbs(face_roi, alpha=1.5, beta=40),
                            cv2.convertScaleAbs(face_roi, alpha=1.3, beta=30),
                            cv2.convertScaleAbs(face_roi, alpha=0.7, beta=-30),
                            cv2.convertScaleAbs(face_roi, alpha=0.5, beta=-10),
                            cv2.GaussianBlur(face_roi, (5, 5), 0),
                            cv2.GaussianBlur(face_roi, (7, 7), 0),
                            cv2.convertScaleAbs(flipped, alpha=1.3, beta=30),
                            cv2.convertScaleAbs(flipped, alpha=0.7, beta=-30),
                            cv2.convertScaleAbs(face_roi, alpha=1.8, beta=60),
                            cv2.convertScaleAbs(face_roi, alpha=1.0, beta=50),
                            cv2.GaussianBlur(cv2.convertScaleAbs(face_roi, alpha=1.3, beta=30), (5,5), 0),
                            cv2.GaussianBlur(cv2.convertScaleAbs(face_roi, alpha=0.7, beta=-20), (3,3), 0),
                        ]
                        
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        augmented.append(clahe.apply(face_roi))
                        augmented.append(clahe.apply(flipped))
                        
                        for aug in augmented:
                            faces.append(aug)
                            labels.append(label_id)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_fb = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe_fb.apply(gray)
                face_roi = cv2.resize(gray, (200, 200))
                faces.append(face_roi)
                labels.append(label_id)
                faces.append(cv2.flip(face_roi, 1))
                labels.append(label_id)

        temp_face_mesh.close()

        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            self.recognition_ready = True
            print(f"[FaceRec] Trained on {len(faces)} sample(s) from {next_label} person(s).")
        else:
            print("[FaceRec] No faces found in known_faces/ folder.")

    def release(self):
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
