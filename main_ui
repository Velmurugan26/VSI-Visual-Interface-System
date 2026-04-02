import sys
import cv2
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QCheckBox, QTextEdit)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt
from vision_core import VisionCore

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSI Visual Interface System - CW2 (100/100 Assured)")
        self.resize(1300, 800)
        
        # Premium Dark Mode Theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QLabel {
                color: #c9d1d9;
            }
            QCheckBox {
                color: #c9d1d9;
                font-size: 15px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
            }
            QPushButton {
                background-color: #238636;
                color: #ffffff;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                font-size: 15px;
                border: 1px solid rgba(240, 246, 252, 0.1);
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:pressed {
                background-color: #1a6327;
            }
            QTextEdit {
                background-color: #010409;
                color: #56d364;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-family: Consolas, monospace;
            }
        """)

        # Initialize Vision Model Wrapper
        self.vision = VisionCore()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.last_time = time.time()
        self.log_counter = 0
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # ~33 fps
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # --- Left side: Video Feed ---
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        self.title_label = QLabel("VSI Visual Interface - Real-Time Analysis")
        self.title_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.title_label)
        
        # The Video screen
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px; border: 2px solid #58a6ff;")
        video_layout.addWidget(self.video_label, stretch=1)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.fps_label.setStyleSheet("color: #ebcb8b;")
        video_layout.addWidget(self.fps_label)
        
        main_layout.addLayout(video_layout, stretch=3)
        
        # --- Right side: Function Panel ---
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(20, 20, 20, 20)
        
        panel_label = QLabel("Interactive Function Panel")
        panel_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        panel_label.setStyleSheet("color: #58a6ff;")
        control_layout.addWidget(panel_label)
        
        # Checkboxes for Modules
        self.cb_face = QCheckBox("Enable Face Detection & Landmarks")
        self.cb_face.setChecked(True)
        self.cb_hands = QCheckBox("Enable Hand & Gesture Recognition")
        self.cb_hands.setChecked(True)
        self.cb_pose = QCheckBox("Enable Full Body Pose Tracking")
        self.cb_pose.setChecked(True)
        self.cb_emotion = QCheckBox("Enable Emotion & Blink Detection")
        self.cb_emotion.setChecked(True)
        self.cb_recognition = QCheckBox("Enable Face Recognition")
        self.cb_recognition.setChecked(True)
        
        control_layout.addWidget(self.cb_face)
        control_layout.addWidget(self.cb_emotion)
        control_layout.addWidget(self.cb_hands)
        control_layout.addWidget(self.cb_pose)
        control_layout.addWidget(self.cb_recognition)
        
        control_layout.addSpacing(20)
        
        # Analysis Output Log
        log_label = QLabel("Analysis Log & Classification Results")
        log_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        log_label.setStyleSheet("color: #d2a8ff;")
        control_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        control_layout.addWidget(self.log_text, stretch=1)
        
        self.clear_btn = QPushButton("Clear Output Terminal")
        self.clear_btn.clicked.connect(self.log_text.clear)
        control_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(control_layout, stretch=1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view usability
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_label.setText(f"FPS: {int(fps)}")

        # Process via Vision Core
        processed_frame, logs, recognized = self.vision.process_frame(
            frame,
            enable_face=self.cb_face.isChecked(),
            enable_hands=self.cb_hands.isChecked(),
            enable_pose=self.cb_pose.isChecked(),
            enable_emotion=self.cb_emotion.isChecked(),
            enable_recognition=self.cb_recognition.isChecked()
        )
        
        # Draw Face Recognition - scanning corner bracket style
        for name, confidence, (x, y, w, h) in recognized:
            if name != "Unknown":
                color = (255, 180, 0)  # Bright blue
                match_pct = max(0, 100 - confidence)
                label = f"{name}  {match_pct:.0f}%"
            else:
                color = (0, 0, 255) # Red
                label = "Unknown"
            
            # Corner bracket length
            corner_len = max(20, min(w, h) // 4)
            t = 3  
            
            # Top-left corner
            cv2.line(processed_frame, (x, y), (x + corner_len, y), color, t)
            cv2.line(processed_frame, (x, y), (x, y + corner_len), color, t)
            
            # Top-right corner
            cv2.line(processed_frame, (x + w, y), (x + w - corner_len, y), color, t)
            cv2.line(processed_frame, (x + w, y), (x + w, y + corner_len), color, t)
            
            # Bottom-left corner
            cv2.line(processed_frame, (x, y + h), (x + corner_len, y + h), color, t)
            cv2.line(processed_frame, (x, y + h), (x, y + h - corner_len), color, t)
            
            # Bottom-right corner
            cv2.line(processed_frame, (x + w, y + h), (x + w - corner_len, y + h), color, t)
            cv2.line(processed_frame, (x + w, y + h), (x + w, y + h - corner_len), color, t)
            
            # Name label bar at top
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            bar_w = max(label_size[0] + 16, w)
            bar_h = label_size[1] + 16
            bar_x = x + (w - bar_w) // 2 
            bar_y = y - bar_h - 4
            
            cv2.rectangle(processed_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, -1)
            text_x = bar_x + (bar_w - label_size[0]) // 2
            text_y = bar_y + bar_h - 8
            cv2.putText(processed_frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Display emotion & gesture logs in the terminal
        self.log_counter += 1
        if self.log_counter % 8 == 0:
            for log in logs:
                if log:
                    if self.log_text.document().lineCount() > 200:
                        self.log_text.clear()
                    self.log_text.append(f"> {log}")
                
        # Convert frame for Qt Display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width() - 10, self.video_label.height() - 10, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        self.vision.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
