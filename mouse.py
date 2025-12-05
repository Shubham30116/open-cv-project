import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize mediapipe hand solution
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Start mediapipe
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the camera
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = hand_landmarks.landmark

                # Index finger tip
                index_x = int(landmarks[8].x * frame_width)
                index_y = int(landmarks[8].y * frame_height)

                # Thumb tip
                thumb_x = int(landmarks[4].x * frame_width)
                thumb_y = int(landmarks[4].y * frame_height)

                # Move Cursor
                screen_x = screen_width / frame_width * index_x
                screen_y = screen_height / frame_height * index_y
                pyautogui.moveTo(screen_x, screen_y)

                # Draw cursor point
                cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 255), -1)

                # Calculate distance between thumb & index finger
                distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

                # Gesture actions
                if distance < 30:
                    cv2.putText(frame, "Click!", (index_x, index_y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    pyautogui.click()
                elif 30 < distance < 70:
                    cv2.putText(frame, "Volume Control", (index_x - 50, index_y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    # You can map distance to volume % here (optional)
                else:
                    cv2.putText(frame, "Move Cursor", (index_x - 50, index_y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
