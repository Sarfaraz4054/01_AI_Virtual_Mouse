import cv2
import mediapipe as mp
import pyautogui

# Video capture and hand tracking initialization
cam = cv2.VideoCapture(0)
hand_mesh = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

# Screen dimensions for mouse control
screen_w, screen_h = pyautogui.size()

# Previous finger collision states for double-click detection
prev_thumb_collide = False
prev_middle_collide = False

# Main loop for hand tracking
while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (840, 680))  # Adjust resolution as needed
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand landmark detection
    output = hand_mesh.process(rgb_frame)
    hand_landmarks = output.multi_hand_landmarks

    frame_h, frame_w, _ = frame.shape

    # Process hand landmarks if detected
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            index_finger = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger = hand_landmark.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle_finger = hand_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Extract coordinates and draw circles
            index_x, index_y = int(index_finger.x * frame_w), int(index_finger.y * frame_h)
            thumb_x, thumb_y = int(thumb_finger.x * frame_w), int(thumb_finger.y * frame_h)
            middle_x, middle_y = int(middle_finger.x * frame_w), int(middle_finger.y * frame_h)

            cv2.circle(frame, (index_x, index_y), 3, (0, 255, 0))
            cv2.circle(frame, (thumb_x, thumb_y), 3, (0, 255, 255))
            cv2.circle(frame, (middle_x, middle_y), 3, (255, 0, 255))

            # Move the mouse using the index finger
            screen_x = screen_w * index_finger.x
            screen_y = screen_h * index_finger.y
            pyautogui.moveTo(screen_x, screen_y)

            # Check for thumb and middle finger collision for double-click
            thumb_middle_collide = abs(thumb_x - middle_x) < 20
            if thumb_middle_collide and prev_thumb_collide and prev_middle_collide:
                pyautogui.doubleClick()

            # Update previous collision states
            prev_thumb_collide = thumb_middle_collide
            prev_middle_collide = thumb_middle_collide

    # Display the frame with hand tracking
    cv2.imshow('Finger Tracking', frame)
    cv2.waitKey(1)
