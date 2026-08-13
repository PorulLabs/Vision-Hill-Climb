import cv2
import mediapipe as mp
import math
from collections import deque

# ==========================================
# SETTINGS
# ==========================================

SMOOTHING_FRAMES = 7

ANGLE_DEAD_ZONE = 10

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
# ANGLE SMOOTHING
# ==========================================

angle_history = deque(
    maxlen=SMOOTHING_FRAMES
)

# ==========================================
# WINDOW
# ==========================================

WINDOW = "Phase 2.3 - Hand Angle"

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
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    angle = 0.0
    tilt = "NEUTRAL"

    # ======================================
    # HAND DETECTED
    # ======================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Wrist = landmark 0
        wrist = hand.landmark[0]

        # Middle finger MCP = landmark 9
        middle_mcp = hand.landmark[9]

        # Convert to pixels
        x1 = int(wrist.x * width)
        y1 = int(wrist.y * height)

        x2 = int(middle_mcp.x * width)
        y2 = int(middle_mcp.y * height)

        # Difference
        dx = x2 - x1
        dy = y2 - y1

        # Calculate angle
        raw_angle = math.degrees(
            math.atan2(dx, -dy)
        )

        # Normalize
        if raw_angle > 180:
            raw_angle -= 360

        if raw_angle < -180:
            raw_angle += 360

        # Smooth angle
        angle_history.append(raw_angle)

        angle = (
            sum(angle_history)
            / len(angle_history)
        )

        # ==================================
        # TILT CLASSIFICATION
        # ==================================

        if angle > ANGLE_DEAD_ZONE:

            tilt = "TILT RIGHT"

        elif angle < -ANGLE_DEAD_ZONE:

            tilt = "TILT LEFT"

        else:

            tilt = "NEUTRAL"

        # ==================================
        # DRAW HAND
        # ==================================

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Draw wrist → middle finger line
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Wrist point
        cv2.circle(
            frame,
            (x1, y1),
            10,
            (0, 255, 0),
            -1
        )

    else:

        angle_history.clear()

        angle = 0.0

        tilt = "NO HAND"

    # ======================================
    # TEXT
    # ======================================

    cv2.putText(
        frame,
        f"TILT: {tilt}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"ANGLE: {angle:.1f} deg",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Tilt your hand naturally",
        (25, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        WINDOW,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ==========================================
# SAFE EXIT
# ==========================================

camera.release()

cv2.destroyAllWindows()

hands.close()

print("Hand angle test stopped.")