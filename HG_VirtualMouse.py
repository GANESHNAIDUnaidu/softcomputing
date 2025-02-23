import csv
from datetime import datetime
import time
import cv2
import mediapipe as mp
import psutil
import pyautogui
from pynput.mouse import Button, Controller
import numpy as np
import util
from collections import deque

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Initialize mouse controller and screen parameters
screen_width, screen_height = pyautogui.size()
mouse = Controller()
pyautogui.FAILSAFE = False

# Constants for gesture detection
FINGER_BENT_THRESHOLD = 45  # Reduced threshold for tighter fist detection
THUMB_L_SHAPE_MIN = 35
THUMB_L_SHAPE_MAX = 100

class SessionMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.fps_values = deque(maxlen=300)  # Store last 300 FPS values
        self.cpu_values = deque(maxlen=300)
        self.memory_values = deque(maxlen=300)
        self.gesture_counts = {
            "Move": 0,
            "Left Click": 0,
            "Right Click": 0,
            "Screenshot": 0,
            "Hold": 0
        }
        
    def update(self, fps, gesture):
        self.fps_values.append(fps)
        self.cpu_values.append(psutil.cpu_percent())
        self.memory_values.append(psutil.Process().memory_percent())
        if gesture in self.gesture_counts:
            self.gesture_counts[gesture] += 1
    
    def get_session_duration(self):
        return time.time() - self.start_time
    
    def save_metrics(self):
        duration = self.get_session_duration()
        avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
        avg_cpu = sum(self.cpu_values) / len(self.cpu_values) if self.cpu_values else 0
        avg_memory = sum(self.memory_values) / len(self.memory_values) if self.memory_values else 0
        
        metrics = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration (seconds)": round(duration, 2),
            "Average FPS": round(avg_fps, 2),
            "Average CPU Usage (%)": round(avg_cpu, 2),
            "Average Memory Usage (%)": round(avg_memory, 2),
            "Peak CPU Usage (%)": round(max(self.cpu_values, default=0), 2),
            "Peak Memory Usage (%)": round(max(self.memory_values, default=0), 2),
            "Minimum FPS": round(min(self.fps_values, default=0), 2),
            "Mouse Movements": self.gesture_counts["Move"],
            "Left Clicks": self.gesture_counts["Left Click"],
            "Right Clicks": self.gesture_counts["Right Click"],
            "Screenshots": self.gesture_counts["Screenshot"],
            "Total Gestures": sum(self.gesture_counts.values())
        }
        
        # Create or append to CSV file
        try:
            with open("SessionMetrics.csv", mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metrics.keys())
                # Write header if file is empty
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
        except Exception as e:
            print(f"Error saving metrics: {e}")
        
        return metrics

class GestureDetector:
    def __init__(self):
        self.current_gesture = "No Gesture"
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 2
        self.gesture_history = []
        self.smoothing_window = 3
        self.fist_start_time = None
        self.fist_duration_threshold = 0.5  # Hold fist for 0.5 seconds for screenshot
        self.screenshot_progress = 0  # Track progress towards screenshot

    def get_finger_angles(self, landmarks_list):
        if len(landmarks_list) < 21:
            return None
            
        thumb_index = util.get_angle(
            landmarks_list[4],
            landmarks_list[2],
            landmarks_list[8]
        )
        
        # Calculate bend angles for all fingers
        finger_bends = []
        for finger_base in [5, 9, 13, 17]:  # Index, Middle, Ring, Pinky
            finger_bends.append(util.get_angle(
                landmarks_list[finger_base],
                landmarks_list[finger_base + 1],
                landmarks_list[finger_base + 2]
            ))
        
        return {
            'thumb_index': thumb_index,
            'finger_bends': finger_bends
        }
    
    def is_tight_fist(self, finger_bends):
        """Check if all fingers are tightly bent"""
        return all(angle < 60 for angle in finger_bends)
    
    def smooth_gesture(self, gesture):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.smoothing_window:
            self.gesture_history.pop(0)
        return max(set(self.gesture_history), key=self.gesture_history.count) if self.gesture_history else gesture
    
    def detect_gesture(self, landmarks_list):
        angles = self.get_finger_angles(landmarks_list)
        if not angles:
            return None
            
        is_l_shape = THUMB_L_SHAPE_MIN < angles['thumb_index'] < THUMB_L_SHAPE_MAX
        is_fist = self.is_tight_fist(angles['finger_bends'])
        
        gesture = "No Gesture"
        current_time = time.time()
        
        # Handle fist detection for screenshot with progress tracking
        if is_fist:
            if self.fist_start_time is None:
                self.fist_start_time = current_time
                self.screenshot_progress = 0
            else:
                progress_time = current_time - self.fist_start_time
                self.screenshot_progress = min(100, int((progress_time / self.fist_duration_threshold) * 100))
                
                if (progress_time >= self.fist_duration_threshold and 
                    current_time - self.last_screenshot_time >= self.screenshot_cooldown):
                    gesture = "Screenshot"
                    self.last_screenshot_time = current_time
                    self.fist_start_time = None
                    self.screenshot_progress = 0
                else:
                    gesture = f"Screenshot Progress: {self.screenshot_progress}%"
        else:
            self.fist_start_time = None
            self.screenshot_progress = 0
            
            # Check individual finger states
            index_finger_bent = angles['finger_bends'][0] < FINGER_BENT_THRESHOLD
            middle_finger_bent = angles['finger_bends'][1] < FINGER_BENT_THRESHOLD
            
            if is_l_shape:
                if index_finger_bent and not middle_finger_bent:
                    gesture = "Left Click"
                elif middle_finger_bent and not index_finger_bent:
                    gesture = "Right Click"
                elif not index_finger_bent and not middle_finger_bent:
                    gesture = "Hold"
            elif angles['finger_bends'][0] > 150:  # Index finger straight
                gesture = "Move"
        
        return self.smooth_gesture(gesture)

def draw_text_with_background(image, text, position, font, font_scale, text_color, bg_color, thickness=2, margin=5):
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top_left = (position[0] - margin, position[1] - text_height - margin)
    bottom_right = (position[0] + text_width + margin, position[1] + margin)
    
    # Draw rectangle background
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    
    # Draw the text
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)    

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    draw = mp.solutions.drawing_utils
    gesture_detector = GestureDetector()
    session_metrics = SessionMetrics()
    
    frame_time = 0
    fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            fps = 1 / (current_time - frame_time) if frame_time else 0
            frame_time = current_time
            
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            
            gesture_text = "No Hand Detected"
            
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                gesture = gesture_detector.detect_gesture(landmarks_list)
                if gesture:
                    gesture_text = gesture
                    
                    # Only update metrics for actual gestures, not progress messages
                    if not gesture.startswith("Screenshot Progress"):
                        session_metrics.update(fps, gesture)
                    
                    if gesture == "Move":
                        x = int(hand_landmarks.landmark[8].x * screen_width)
                        y = int(hand_landmarks.landmark[8].y * screen_height)
                        pyautogui.moveTo(x, y, duration=0.1)
                    elif gesture == "Left Click":
                        mouse.click(Button.left)
                        time.sleep(0.2)
                    elif gesture == "Right Click":
                        mouse.click(Button.right)
                        time.sleep(0.2)
                    elif gesture == "Screenshot":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"screenshot_{timestamp}.png"
                        pyautogui.screenshot(filename)
                        gesture_text = f"Screenshot taken: {filename}"
                
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                
                # Highlight key points
                for idx in [4, 8, 12]:
                    x = int(hand_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            
            ## Display metrics on frame
            draw_text_with_background(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), (0, 255, 0), 2)
            draw_text_with_background(frame, f"Gesture: {gesture_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), (0, 255, 0), 2)
            draw_text_with_background(frame, f"Session Time: {int(session_metrics.get_session_duration())}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), (0, 255, 0), 2)
            
            # Draw screenshot progress bar if in progress
            if gesture_text.startswith("Screenshot Progress"):
                progress = gesture_detector.screenshot_progress
                bar_width = int((frame.shape[1] - 100) * (progress / 100))
                cv2.rectangle(frame, (50, 150), (50 + bar_width, 170), (0, 255, 0), -1)
                cv2.rectangle(frame, (50, 150), (frame.shape[1] - 50, 170), (255, 255, 255), 2)
            
            cv2.imshow('Hand Gesture Mouse Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Save session metrics before closing
        final_metrics = session_metrics.save_metrics()
        print("\nSession Summary:")
        for key, value in final_metrics.items():
            print(f"{key}: {value}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
  #to run the code python3 HG_VirtualMouse.py
