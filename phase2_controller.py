import cv2
import mediapipe as mp
import pyautogui
from collections import deque

# ==========================================
# SETTINGS
# ==========================================

LEFT_ZONE = 0.35
RIGHT_ZONE = 0.65

SMOOTHING_FRAMES = 7

# ==========================================
# MEDIAPIPE
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
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not detected.")
    exit()

# ==========================================
# SMOOTHING
# ==========================================

position_history = deque(
    maxlen=SMOOTHING_FRAMES
)

current_key = None


# ==========================================
# KEYBOARD CONTROL
# ==========================================

def release_all():

    global current_key

    if current_key == "a":
        pyautogui.keyUp("a")

    elif current_key == "d":
        pyautogui.keyUp("d")

    current_key = None


def set_key(key):

    global current_key

    if key == current_key:
        return

    release_all()

    if key == "a":
        pyautogui.keyDown("a")
        current_key = "a"

    elif key == "d":
        pyautogui.keyDown("d")
        current_key = "d"


# ==========================================
# WINDOW
# ==========================================

WINDOW = "Phase 2 - Vision Controller"

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    960,
    720
)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame failed.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe
    results = hands.process(rgb)

    gesture = "NO HAND"
    smoothed_x = 0.5

    # ======================================
    # HAND DETECTION
    # ======================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]

        x = wrist.x

        # Add position to history
        position_history.append(x)

        # Smooth position
        smoothed_x = (
            sum(position_history)
            / len(position_history)
        )

        # ==================================
        # CONTROL LOGIC
        # ==================================

        if smoothed_x < LEFT_ZONE:

            gesture = "BRAKE"

            set_key("a")

        elif smoothed_x > RIGHT_ZONE:

            gesture = "ACCELERATE"

            set_key("d")

        else:

            gesture = "NEUTRAL"

            release_all()

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Draw wrist marker
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

        # No hand = emergency release
        position_history.clear()

        release_all()

    # ======================================
    # DISPLAY TEXT
    # ======================================

    cv2.putText(
        frame,
        f"MODE: {gesture}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"HAND X: {smoothed_x:.2f}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if current_key:

        key_status = (
            f"KEY DOWN: "
            f"{current_key.upper()}"
        )

    else:

        key_status = "KEY: RELEASED"

    cv2.putText(
        frame,
        key_status,
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Instructions
    cv2.putText(
        frame,
        "LEFT = BRAKE",
        (25, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "CENTER = NEUTRAL",
        (width // 2 - 110, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "RIGHT = ACCELERATE",
        (width - 250, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        WINDOW,
        frame
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break


# ==========================================
# SAFE EXIT
# ==========================================

release_all()

camera.release()

cv2.destroyAllWindows()

hands.close()

print("Phase 2 controller stopped safely.")