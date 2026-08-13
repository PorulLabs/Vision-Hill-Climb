import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time
from collections import deque


# ============================================================
# VISION HILL CLIMB - REAL GAME CONTROLLER
# ============================================================
#
# HAND POSITION
#
# LEFT   -> LEFT ARROW  -> BRAKE
# CENTER -> RELEASE
# RIGHT  -> RIGHT ARROW -> ACCELERATE
#
# IMPORTANT:
# This version uses LEFT/RIGHT arrows.
# It does NOT use A/D.
#
# No blue/cyan control lines.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

CENTER_X = 0.50

DEAD_ZONE = 0.08

MAX_DISTANCE = 0.40

SMOOTHING_FRAMES = 7

GAME_TITLE = "Hill Climb Racing"


# ============================================================
# MEDIAPIPE
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
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    hands.close()

    raise SystemExit


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    960
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# SMOOTHING
# ============================================================

position_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ============================================================
# KEY STATE
# ============================================================

current_key = None


# ============================================================
# FIND GAME WINDOW
# ============================================================

def find_game():

    windows = gw.getWindowsWithTitle(
        GAME_TITLE
    )

    if not windows:
        return None

    return windows[0]


# ============================================================
# FOCUS GAME
# ============================================================

def focus_game():

    game = find_game()

    if game is None:

        return False

    try:

        if game.isMinimized:

            game.restore()

        game.activate()

        return True

    except Exception:

        return False


# ============================================================
# RELEASE BOTH GAME KEYS
# ============================================================

def release_all():

    global current_key

    try:

        pyautogui.keyUp("left")

        pyautogui.keyUp("right")

    except Exception:

        pass

    current_key = None


# ============================================================
# SEND GAME KEY
# ============================================================

def send_game_key(key):

    global current_key

    # Nothing to change
    if current_key == key:

        return


    # Release old key
    release_all()


    # Make sure game is active
    if not focus_game():

        print("WARNING: Hill Climb Racing is not focused.")

        return


    try:

        pyautogui.keyDown(key)

        current_key = key

    except Exception:

        current_key = None


# ============================================================
# CAMERA WINDOW
# ============================================================

WINDOW = "Vision Hill Climb - Game Controller"

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
# STARTUP
# ============================================================

print()
print("==============================================")
print(" VISION HILL CLIMB - REAL GAME CONTROLLER")
print("==============================================")
print()
print("LEFT HAND   -> LEFT ARROW  -> BRAKE")
print("CENTER      -> RELEASE")
print("RIGHT HAND  -> RIGHT ARROW -> ACCELERATE")
print("NO HAND     -> RELEASE")
print()
print("==============================================")
print()


# ============================================================
# CHECK GAME
# ============================================================

game = find_game()


if game is None:

    print("ERROR: Hill Climb Racing was not found.")

    print()
    print("Open Hill Climb Racing first.")
    print("Then start a race.")
    print()

    camera.release()

    cv2.destroyAllWindows()

    hands.close()

    raise SystemExit


print(
    "Game found:",
    game.title
)

print()


# ============================================================
# COUNTDOWN
# ============================================================

for number in [5, 4, 3, 2, 1]:

    print(
        f"Starting in {number}..."
    )

    success, frame = camera.read()

    if success:

        frame = cv2.flip(
            frame,
            1
        )

        cv2.putText(
            frame,
            f"STARTING {number}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            WINDOW,
            frame
        )

        cv2.waitKey(1)

    time.sleep(1)


# ============================================================
# IMPORTANT:
# FOCUS THE GAME AFTER COUNTDOWN
# ============================================================

if focus_game():

    print()
    print("Hill Climb Racing is now focused.")
    print("GAME CONTROL ACTIVE.")
    print()

else:

    print()
    print("WARNING: Could not focus the game.")
    print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "Camera frame failed."
        )

        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe
    results = hands.process(
        rgb
    )


    # Default state
    mode = "NO HAND"

    intensity = 0.0


    # ========================================================
    # HAND FOUND
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]


        # ----------------------------------------------------
        # HAND X POSITION
        # ----------------------------------------------------

        x = wrist.x


        # Add smoothing
        position_history.append(x)


        smooth_x = (
            sum(position_history)
            /
            len(position_history)
        )


        # Difference from center
        distance = (
            smooth_x - CENTER_X
        )


        # ====================================================
        # LEFT = BRAKE
        # ====================================================

        if distance < -DEAD_ZONE:

            mode = "BRAKE"


            intensity = (
                abs(distance)
                - DEAD_ZONE
            ) / (
                MAX_DISTANCE
                - DEAD_ZONE
            )


            intensity = max(
                0.0,
                min(
                    1.0,
                    intensity
                )
            )


            send_game_key(
                "left"
            )


        # ====================================================
        # RIGHT = ACCELERATE
        # ====================================================

        elif distance > DEAD_ZONE:

            mode = "ACCELERATE"


            intensity = (
                distance
                - DEAD_ZONE
            ) / (
                MAX_DISTANCE
                - DEAD_ZONE
            )


            intensity = max(
                0.0,
                min(
                    1.0,
                    intensity
                )
            )


            send_game_key(
                "right"
            )


        # ====================================================
        # CENTER = RELEASE
        # ====================================================

        else:

            mode = "NEUTRAL"

            intensity = 0.0

            release_all()


        # ----------------------------------------------------
        # HAND LANDMARKS
        # ----------------------------------------------------

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
            (
                wrist_x,
                wrist_y
            ),
            7,
            (0, 255, 0),
            -1
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        position_history.clear()

        release_all()

        mode = "NO HAND"

        intensity = 0.0


    # ========================================================
    # CLEAN UI
    # ========================================================

    panel_x = 15
    panel_y = 15

    panel_width = 300
    panel_height = 125


    overlay = frame.copy()


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


    frame = cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0
    )


    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        mode,
        (
            30,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # INTENSITY
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Intensity : {intensity * 100:.0f}%",
        (
            30,
            75
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # KEY
    # --------------------------------------------------------

    if current_key:

        key_text = (
            f"Key       : "
            f"{current_key.upper()}"
        )

    else:

        key_text = (
            "Key       : RELEASED"
        )


    cv2.putText(
        frame,
        key_text,
        (
            30,
            105
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
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


    # ESC
    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


# ============================================================
# SAFE EXIT
# ============================================================

print()
print("Stopping controller...")


release_all()


camera.release()


cv2.destroyAllWindows()


hands.close()


print("Controller stopped safely.")