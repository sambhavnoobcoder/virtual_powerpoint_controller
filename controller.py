import cv2
import mediapipe as mp
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)  # Use the default camera

# Set up the MediaPipe Hands model
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        # Read a frame from the camera
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a more natural feel
        image = cv2.flip(image, 1)

        # Convert the image to RGB and process it with MediaPipe Hands
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for swipe gesture
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if (index_tip.x - thumb_tip.x) > 0.05:
                    print("Swipe Right")
                    pyautogui.press("left")
                    time.sleep(1)
                elif (index_tip.x - thumb_tip.x) < -0.1:
                    print("Swipe Left")
                    pyautogui.press("right")
                    time.sleep(2)

        # Display the image
        cv2.imshow('MediaPipe Hands', image)

        # Check for exit key
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
