import cv2
import mediapipe as mp
import pyautogui
import time

# ==========================================
# MediaPipe
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================
# Camera
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

# Don't force 1280x720.
# Let the webcam use its natural resolution.

print("Camera started.")

# ==========================================
# Controller
# ==========================================

current_key = None


def release_keys():
    global current_key

    if current_key == "a":
        pyautogui.keyUp("a")

    elif current_key == "d":
        pyautogui.keyUp("d")

    current_key = None


def change_key(new_key):
    global current_key

    if new_key == current_key:
        return

    release_keys()

    if new_key == "a":
        pyautogui.keyDown("a")
        current_key = "a"

    elif new_key == "d":
        pyautogui.keyDown("d")
        current_key = "d"


# ==========================================
# Window
# ==========================================

WINDOW_NAME = "Vision Hill Climb Controller"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# 4:3 window — matches most laptop webcams
cv2.resizeWindow(
    WINDOW_NAME,
    960,
    720
)

# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame failed.")
        break

    # Mirror camera naturally
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # ======================================
    # MediaPipe
    # ======================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]

        x = wrist.x

        # LEFT
        if x < 0.40:

            gesture = "LEFT"
            change_key("a")

        # RIGHT
        elif x > 0.60:

            gesture = "RIGHT"
            change_key("d")

        # CENTER
        else:

            gesture = "CENTER"
            release_keys()

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Wrist marker
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            10,
            (0, 255, 0),
            -1
        )

    else:

        release_keys()

    # ======================================
    # Text
    # ======================================

    cv2.putText(
        frame,
        f"GESTURE: {gesture}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    if current_key:
        key_text = f"KEY: {current_key.upper()} DOWN"
    else:
        key_text = "KEY: RELEASED"

    cv2.putText(
        frame,
        key_text,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break


# ==========================================
# Safe shutdown
# ==========================================

release_keys()

camera.release()

cv2.destroyAllWindows()

hands.close()

print("Controller stopped safely.")