import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque

# ============================================================
# PHASE 2.4 - COMPUTER VISION GAME CONTROLLER
# ============================================================
#
# Hand position:
#     LEFT   -> BRAKE  (A)
#     CENTER -> NEUTRAL
#     RIGHT  -> ACCELERATE (D)
#
# Hand angle:
#     LEFT TILT  -> Vehicle tilt signal
#     LEVEL      -> Neutral
#     RIGHT TILT -> Vehicle tilt signal
#
# IMPORTANT:
# No blue/cyan control-zone lines are used.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CENTER_X = 0.50

# Position dead zone
POSITION_DEAD_ZONE = 0.08

# Maximum distance used for intensity calculation
MAX_POSITION_DISTANCE = 0.40

# Angle dead zone
ANGLE_DEAD_ZONE = 10

# Smoothing
POSITION_SMOOTHING = 7
ANGLE_SMOOTHING = 7


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# CAMERA SETUP
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# SMOOTHING HISTORY
# ============================================================

position_history = deque(
    maxlen=POSITION_SMOOTHING
)

angle_history = deque(
    maxlen=ANGLE_SMOOTHING
)


# ============================================================
# KEYBOARD STATE
# ============================================================

current_key = None


# ============================================================
# SAFE KEYBOARD FUNCTIONS
# ============================================================

def release_all():

    global current_key

    try:

        if current_key == "a":
            pyautogui.keyUp("a")

        elif current_key == "d":
            pyautogui.keyUp("d")

    except Exception:
        pass

    current_key = None


def press_key(key):

    global current_key

    # Already holding the requested key
    if key == current_key:
        return

    # Release previous key
    release_all()

    try:

        pyautogui.keyDown(key)

        current_key = key

    except Exception:

        current_key = None


# ============================================================
# WINDOW
# ============================================================

WINDOW = "Vision Hill Climb - Phase 2.4"

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    960,
    720
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAMERA FRAME
    # --------------------------------------------------------

    success, frame = camera.read()

    if not success:

        print("Camera frame failed.")
        break


    # Mirror the camera
    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    position_mode = "NO HAND"

    tilt_mode = "NO HAND"

    intensity = 0.0

    smooth_x = CENTER_X

    angle = 0.0


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]


        # ----------------------------------------------------
        # LANDMARKS
        # ----------------------------------------------------

        wrist = hand.landmark[0]

        middle_mcp = hand.landmark[9]


        # ====================================================
        # 1. HAND POSITION
        # ====================================================

        x = wrist.x


        # Add position to smoothing history
        position_history.append(x)


        # Calculate smoothed position
        smooth_x = (
            sum(position_history)
            / len(position_history)
        )


        # Distance from center
        distance = (
            smooth_x - CENTER_X
        )


        # ====================================================
        # POSITION CONTROL
        # ====================================================

        # -------------------------------
        # CENTER
        # -------------------------------

        if abs(distance) <= POSITION_DEAD_ZONE:

            position_mode = "NEUTRAL"

            intensity = 0.0

            release_all()


        # -------------------------------
        # RIGHT
        # -------------------------------

        elif distance > POSITION_DEAD_ZONE:

            position_mode = "ACCELERATE"


            intensity = (
                distance - POSITION_DEAD_ZONE
            ) / (
                MAX_POSITION_DISTANCE
                - POSITION_DEAD_ZONE
            )


            # Keep between 0 and 1
            intensity = max(
                0.0,
                min(
                    1.0,
                    intensity
                )
            )


            # Current keyboard mapping
            press_key("d")


        # -------------------------------
        # LEFT
        # -------------------------------

        else:

            position_mode = "BRAKE"


            intensity = (
                abs(distance)
                - POSITION_DEAD_ZONE
            ) / (
                MAX_POSITION_DISTANCE
                - POSITION_DEAD_ZONE
            )


            # Keep between 0 and 1
            intensity = max(
                0.0,
                min(
                    1.0,
                    intensity
                )
            )


            # Current keyboard mapping
            press_key("a")


        # ====================================================
        # 2. HAND ANGLE
        # ====================================================

        # Wrist coordinates
        x1 = wrist.x * width
        y1 = wrist.y * height


        # Middle finger MCP coordinates
        x2 = middle_mcp.x * width
        y2 = middle_mcp.y * height


        # Vector
        dx = x2 - x1

        dy = y2 - y1


        # Calculate angle
        raw_angle = math.degrees(
            math.atan2(
                dx,
                -dy
            )
        )


        # Normalize angle
        if raw_angle > 180:

            raw_angle -= 360


        if raw_angle < -180:

            raw_angle += 360


        # Add angle to smoothing history
        angle_history.append(
            raw_angle
        )


        # Calculate smoothed angle
        angle = (
            sum(angle_history)
            / len(angle_history)
        )


        # ====================================================
        # TILT CLASSIFICATION
        # ====================================================

        if angle > ANGLE_DEAD_ZONE:

            tilt_mode = "TILT RIGHT"


        elif angle < -ANGLE_DEAD_ZONE:

            tilt_mode = "TILT LEFT"


        else:

            tilt_mode = "LEVEL"


        # ====================================================
        # DRAW HAND LANDMARKS
        # ====================================================

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # ====================================================
        # DRAW WRIST MARKER
        # ====================================================

        wrist_x = int(
            wrist.x * width
        )

        wrist_y = int(
            wrist.y * height
        )


        cv2.circle(
            frame,
            (
                wrist_x,
                wrist_y
            ),
            8,
            (0, 255, 0),
            -1
        )


        # ====================================================
        # HAND ANGLE REFERENCE
        # ====================================================
        #
        # This green line shows what the angle detector
        # is measuring.
        #
        # It is NOT a control-zone line.
        # ====================================================

        cv2.line(
            frame,
            (
                int(x1),
                int(y1)
            ),
            (
                int(x2),
                int(y2)
            ),
            (0, 255, 0),
            2
        )


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        # Clear history
        position_history.clear()

        angle_history.clear()


        # Emergency release
        release_all()


        position_mode = "NO HAND"

        tilt_mode = "NO HAND"

        intensity = 0.0

        smooth_x = CENTER_X

        angle = 0.0


    # ========================================================
    # CLEAN UI PANEL
    # ========================================================

    # Panel dimensions
    panel_x = 15

    panel_y = 15

    panel_width = 300

    panel_height = 170


    # Create overlay
    overlay = frame.copy()


    # Dark panel
    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (20, 20, 20),
        -1
    )


    # Blend panel
    frame = cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0
    )


    # ========================================================
    # TEXT
    # ========================================================

    # Main control
    cv2.putText(
        frame,
        position_mode,
        (
            panel_x + 15,
            panel_y + 32
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # Intensity
    cv2.putText(
        frame,
        f"Intensity : {intensity * 100:.0f}%",
        (
            panel_x + 15,
            panel_y + 62
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # Tilt
    cv2.putText(
        frame,
        f"Tilt      : {tilt_mode}",
        (
            panel_x + 15,
            panel_y + 90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # Angle
    cv2.putText(
        frame,
        f"Angle     : {angle:.1f} deg",
        (
            panel_x + 15,
            panel_y + 118
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # Keyboard
    if current_key:

        key_status = (
            f"Key       : "
            f"{current_key.upper()}"
        )

    else:

        key_status = (
            "Key       : RELEASED"
        )


    cv2.putText(
        frame,
        key_status,
        (
            panel_x + 15,
            panel_y + 146
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW,
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ESC = emergency exit
    if key == 27:

        break


# ============================================================
# SAFE SHUTDOWN
# ============================================================

release_all()

camera.release()

cv2.destroyAllWindows()

hands.close()


print(
    "Vision controller stopped safely."
)