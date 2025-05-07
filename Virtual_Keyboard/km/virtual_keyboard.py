import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import json
import time
import os


# Configuration constants
CALIBRATION_FILE = "key_positions.json"          # File to store calibrated key positions
PRESS_DISTANCE_THRESHOLD = 20                   # Threshold for pinch detection (pixels)
STABILITY_THRESHOLD = 15                        # Maximum allowed movement during calibration (pixels)
MIN_STABLE_FRAMES = 10                          # Frames needed for stable position confirmation
KEY_RADIUS = 25                                 # Default radius for key activation zones (pixels)
EDGE_BUFFER = 0.1                               # Buffer to prevent finger from going off-screen (ratio)
RESET_DISTANCE = 150                            # Distance to reset calibration progress (pixels)
TYPING_COOLDOWN = 0.2                           # Delay between key presses (seconds)
HAND_SCALE_FACTOR = 0.15                        # Percentage of hand size to use as pinch threshold
CALIBRATION_HELP_TEXT = """
Calibration Guide:
1. Position pointing finger over target key
2. Make pinch gesture with other hand
3. Hold both hands steady until progress fills
4. Release pinch to confirm
"""

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # For video stream processing
    max_num_hands=2,               # Maximum number of hands to detect
    min_detection_confidence=0.6,  # Minimum confidence threshold for detection
    min_tracking_confidence=0.4,   # Minimum confidence threshold for tracking
    model_complexity=1             # Model complexity (0=light, 1=full, 2=heavy)
)


class VirtualKeyboard:
    """Main class for virtual keyboard functionality including calibration and typing."""
    
    def __init__(self):
        """Initialize keyboard with default settings and load existing calibration if available."""
        self.key_positions = {}                 # Dictionary to store key positions
        self.calibration_mode = True            # Flag for calibration state
        # Standard QWERTY keyboard layout for calibration
        self.calibration_order = [
            'Esc', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
            'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
            'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Space'
        ]
        self.current_key = None                  # Current key being calibrated
        self.last_calibration_pos = None        # Last recorded position during calibration
        self.stable_frames = 0                  # Counter for stable frames
        self.calibration_positions = []         # List of positions for stability check
        self.last_press_time = 0                # Timestamp of last key press
        self.pointing_hand_type = None          # Which hand is used for pointing ('Left' or 'Right')
        
        self.load_calibration()                 # Load existing calibration data
        print(CALIBRATION_HELP_TEXT)            # Show calibration instructions
        pyautogui.FAILSAFE = False              # Disable pyautogui failsafe feature


    def load_calibration(self):
        """Load calibration data from file if it exists."""
        try:
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.key_positions = data['positions']
                    self.pointing_hand_type = data.get('pointing_hand', None)
                
                # Check for missing keys or unset hand type
                missing = [k for k in self.calibration_order if k not in self.key_positions]
                if missing or not self.pointing_hand_type:
                    print(f"Resuming calibration. Missing keys: {', '.join(missing)}")
                    self.calibration_order = missing if missing else self.calibration_order
                    self.current_key = self.calibration_order[0]
                else:
                    print("Calibration loaded successfully!")
                    self.calibration_mode = False
            else:
                print("Starting new calibration")
                self.current_key = self.calibration_order[0]
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            self.key_positions = {}
            self.calibration_mode = True


    def save_calibration(self):
        """Save current calibration data to file."""
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump({
                    'positions': self.key_positions,
                    'pointing_hand': self.pointing_hand_type
                }, f, indent=2)
            print("Calibration saved successfully!")
        except Exception as e:
            print(f"Error saving calibration: {str(e)}")


    def get_finger_position(self, landmarks, frame_shape):
        """
        Extract and return the position of the index finger tip.
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the video frame (height, width)
            
        Returns:
            Tuple of (x, y) coordinates or None if detection fails
        """
        try:
            h, w = frame_shape
            tip = landmarks.landmark[8]  # Index finger tip landmark
            x = int(np.clip(tip.x, EDGE_BUFFER, 1-EDGE_BUFFER) * w)
            y = int(np.clip(tip.y, EDGE_BUFFER, 1-EDGE_BUFFER) * h)
            return (x, y)
        except:
            return None


    def is_stable_position(self, new_pos):
        """
        Check if the current position is stable based on recent history.
        
        Args:
            new_pos: Current (x, y) position
            
        Returns:
            Boolean indicating if position is stable
        """
        try:
            self.calibration_positions.append(new_pos)
            if len(self.calibration_positions) < MIN_STABLE_FRAMES:
                return False
            
            # Calculate average position over the stability window
            avg_x = np.mean([p[0] for p in self.calibration_positions[-MIN_STABLE_FRAMES:]])
            avg_y = np.mean([p[1] for p in self.calibration_positions[-MIN_STABLE_FRAMES:]])
            return all([
                abs(new_pos[0] - avg_x) < STABILITY_THRESHOLD,
                abs(new_pos[1] - avg_y) < STABILITY_THRESHOLD
            ])
        except:
            return False


    def draw_calibration_ui(self, frame, position, progress):
        """
        Draw calibration interface on the video frame.
        
        Args:
            frame: Video frame to draw on
            position: Current finger position
            progress: Calibration progress (0.0 to 1.0)
        """
        try:
            h, w = frame.shape[:2]
            # Display current key being calibrated
            cv2.putText(frame, f"Calibrating: {self.current_key}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if position:
                cv2.circle(frame, position, 15, (0, 0, 255), -1)
            
            # Draw progress bar with easing for smoother animation
            eased_progress = progress ** 0.5
            bar_width = int(w//4 + (w//2)*eased_progress)
            cv2.rectangle(frame, (w//4, h-50), (w//4*3, h-30), (50, 50, 50), -1)
            cv2.rectangle(frame, (w//4, h-50), (bar_width, h-30), (0, 200, 0), -1)
            
            # Display calibration instructions
            y_offset = 60
            for line in CALIBRATION_HELP_TEXT.split('\n')[2:]:
                cv2.putText(frame, line, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
        except:
            pass


    def handle_calibration(self, frame, position, pinching):
        """
        Handle the calibration process.
        
        Args:
            frame: Video frame
            position: Current finger position
            pinching: Whether pinch gesture is detected
            
        Returns:
            Current calibration progress (0.0 to 1.0)
        """
        try:
            if not position:
                return 0.0
            
            if self.last_calibration_pos is None:
                self.last_calibration_pos = position
                return 0.0
            
            # Calculate distance from last position
            distance = np.hypot(position[0]-self.last_calibration_pos[0],
                               position[1]-self.last_calibration_pos[1])
            
            progress = min(1.0, distance / RESET_DISTANCE)
            
            if pinching and self.is_stable_position(position):
                self.stable_frames += 1
                progress = self.stable_frames / MIN_STABLE_FRAMES
                
                if self.stable_frames >= MIN_STABLE_FRAMES:
                    # Save calibrated key position
                    self.key_positions[self.current_key] = {
                        'x': position[0],
                        'y': position[1],
                        'radius': KEY_RADIUS
                    }
                    self.calibration_order.pop(0)
                    self.stable_frames = 0
                    self.calibration_positions = []
                    self.last_calibration_pos = None
                    
                    if not self.calibration_order:
                        # Calibration complete
                        self.save_calibration()
                        self.calibration_mode = False
                        print("Calibration complete! Switch to typing mode with 't'")
                    else:
                        self.current_key = self.calibration_order[0]
            
            return progress
        except:
            return 0.0

    def find_active_key(self, position):
        """
        Find which key (if any) is currently being pointed at.
        
        Args:
            position: Current finger position
            
        Returns:
            Key name or None if no key is active
        """
        try:
            if not position:
                return None
            
            active_key = None
            min_distance = float('inf')
            
            # Check distance to all keys
            for key, data in self.key_positions.items():
                dx = position[0] - data['x']
                dy = position[1] - data['y']
                distance = (dx**2 + dy**2)**0.5
                
                if distance < data['radius'] and distance < min_distance:
                    min_distance = distance
                    active_key = key
                    
            return active_key
        except:
            return None


    def handle_key_press(self, key):
        """
        Handle the actual key press action.
        
        Args:
            key: The key to be pressed
        """
        try:
            current_time = time.time()
            if (current_time - self.last_press_time) < TYPING_COOLDOWN:
                return

            # Special key mappings
            special_keys = {
                'Esc': ('escape', '\x1b'),
                'Space': ('space', ' '),
                'Enter': ('enter', '\n'),
                'Backspace': ('backspace', '\x08'),
                'Tab': ('tab', '\t'),
                '↑': ('up', '[UP]'),
                '←': ('left', '[LEFT]'),
                '↓': ('down', '[DOWN]'),
                '→': ('right', '[RIGHT]')
            }

            # Handle special keys differently
            if key in special_keys:
                pyautogui_key, char = special_keys[key]
                pyautogui.press(pyautogui_key)
            else:
                char = key.lower()
                pyautogui.press(char)

            # Optional: Send character to kernel module (for LED feedback)
            try:
                with open('/dev/led_blinker', 'wb') as dev:
                    dev.write(char.encode())
                    print(f"Sent to kernel: {char!r}")
            except Exception as e:
                print(f"Device write error: {str(e)}")

            self.last_press_time = current_time
            print(f"Pressed: {key}")

        except Exception as e:
            print(f"Press error: {str(e)}")


def main():
    """Main function to run the virtual keyboard application."""
    cap = cv2.VideoCapture(0)  # Initialize video capture
    keyboard = VirtualKeyboard()
    press_triggered = False    # Flag to prevent repeated presses

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process frame with MediaPipe
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        position = None        # Current finger position
        pinching = False      # Pinch gesture state
        active_key = None      # Currently active key

        pointing_hand = None  # Hand used for pointing
        trigger_hand = None   # Hand used for triggering

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label
                
                # During first calibration, detect which hand is pointing
                if keyboard.calibration_mode and not keyboard.pointing_hand_type:
                    keyboard.pointing_hand_type = hand_type
                    pointing_hand = hand_landmarks
                else:
                    if hand_type == keyboard.pointing_hand_type:
                        pointing_hand = hand_landmarks
                    else:
                        trigger_hand = hand_landmarks

            # Get pointing finger position
            if pointing_hand:
                position = keyboard.get_finger_position(pointing_hand, frame.shape[:2])

            # Check for pinch gesture on trigger hand
            if trigger_hand:
                wrist = trigger_hand.landmark[0]
                middle_mcp = trigger_hand.landmark[9]
                # Calculate hand size for dynamic threshold
                hand_size = np.hypot(
                    (wrist.x - middle_mcp.x) * w,
                    (wrist.y - middle_mcp.y) * h
                )
                
                # Adjust threshold based on hand size
                dynamic_threshold = max(PRESS_DISTANCE_THRESHOLD, hand_size * HAND_SCALE_FACTOR)
                
                # Calculate distance between thumb and index finger
                thumb = trigger_hand.landmark[4]
                index = trigger_hand.landmark[8]
                distance = np.hypot(
                    (thumb.x - index.x) * w,
                    (thumb.y - index.y) * h
                )
                
                # Display pinch distance info
                cv2.putText(frame, f"Pinch: {int(distance)}/{int(dynamic_threshold)}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                pinching = distance < dynamic_threshold

        # Handle calibration or typing mode
        if keyboard.calibration_mode:
            progress = keyboard.handle_calibration(frame, position, pinching)
            keyboard.draw_calibration_ui(frame, position, progress)
        else:
            # Find which key is currently active
            active_key = keyboard.find_active_key(position)
            
            # Draw all keys on the frame
            for key, data in keyboard.key_positions.items():
                color = (0, 200, 200) if key == active_key else (0, 100, 100)
                cv2.circle(frame, (data['x'], data['y']), data['radius'], color, 2)
                cv2.putText(frame, key, (data['x']+10, data['y']-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Handle key press if conditions are met
            if active_key and pinching and not press_triggered:
                keyboard.handle_key_press(active_key)
                press_triggered = True
                cv2.circle(frame, position, 20, (0, 0, 255), 3)
            elif not pinching:
                press_triggered = False

        # Display the frame
        cv2.imshow('Virtual Keyboard', frame)
        
        key = cv2.waitKey(1)
        # Uncomment these for debugging controls:
        # if key == ord('q'):
        #    break
        # elif key == ord('r'):
        #    os.remove(CALIBRATION_FILE)
        #    keyboard = VirtualKeyboard()

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()