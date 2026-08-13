import cv2
import mediapipe as mp
import pyautogui
from collections import deque

# ==========================================
# SETTINGS
# ==========================================

CENTER = 0.50

DEAD_ZONE = 0.08

MAX_DISTANCE = 0.40

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
    print("ERROR: Camera could not be opened.")
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


def press_key(key):

    global current_key

    if key == current_key:
        return

    release_all()

    pyautogui.keyDown(key)

    current_key = key


# ==========================================
# WINDOW
# ==========================================

WINDOW = "Phase 2.2 - Vision Controller"

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

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe
    results = hands.process(rgb)

    gesture = "NO HAND"
    intensity = 0.0
    smooth_x = CENTER


    # ======================================
    # HAND DETECTED
    # ======================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]

        x = wrist.x

        # Add position
        position_history.append(x)

        # Smooth position
        smooth_x = (
            sum(position_history)
            / len(position_history)
        )


        # ==================================
        # DISTANCE FROM CENTER
        # ==================================

        distance = smooth_x - CENTER


        # ==================================
        # CENTER / DEAD ZONE
        # ==================================

        if abs(distance) <= DEAD_ZONE:

            gesture = "NEUTRAL"

            intensity = 0.0

            release_all()


        # ==================================
        # RIGHT → ACCELERATE
        # ==================================

        elif distance > DEAD_ZONE:

            gesture = "ACCELERATE"

            intensity = (
                distance - DEAD_ZONE
            ) / (
                MAX_DISTANCE - DEAD_ZONE
            )

            intensity = max(
                0.0,
                min(1.0, intensity)
            )

            press_key("d")


        # ==================================
        # LEFT → BRAKE
        # ==================================

        else:

            gesture = "BRAKE"

            intensity = (
                abs(distance) - DEAD_ZONE
            ) / (
                MAX_DISTANCE - DEAD_ZONE
            )

            intensity = max(
                0.0,
                min(1.0, intensity)
            )

            press_key("a")


        # ==================================
        # DRAW HAND
        # ==================================

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # Wrist marker
        wrist_x = int(
            wrist.x * width
        )

        wrist_y = int(
            wrist.y * height
        )

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            10,
            (0, 255, 0),
            -1
        )


    # ======================================
    # NO HAND
    # ======================================

    else:

        position_history.clear()

        release_all()

        gesture = "NO HAND"

        intensity = 0.0


    # ======================================
    # TEXT DISPLAY
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
        f"INTENSITY: {intensity * 100:.0f}%",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    if current_key:

        status = (
            f"KEY: {current_key.upper()} DOWN"
        )

    else:

        status = "KEY: RELEASED"


    cv2.putText(
        frame,
        status,
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ======================================
    # CAMERA
    # ======================================

    cv2.imshow(
        WINDOW,
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ESC
    if key == 27:
        break


# ==========================================
# SAFE SHUTDOWN
# ==========================================

release_all()

camera.release()

cv2.destroyAllWindows()

hands.close()

print("Phase 2.2 controller stopped safely.")