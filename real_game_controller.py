import cv2
import time
import math
from pathlib import Path

import mediapipe as mp
import pyautogui


# ============================================================
# VISION-X HILL CLIMB
# REALISTIC RED + WHITE HAND TRACKING
# ============================================================

WINDOW_NAME = "Vision-X Hill Climb - Gesture Controller"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Smaller image for MediaPipe = faster processing
PROCESS_WIDTH = 480
PROCESS_HEIGHT = 360


# ============================================================
# HAND SETTINGS
# ============================================================

LEFT_ZONE = 0.40
RIGHT_ZONE = 0.60

HAND_DETECTION_CONFIDENCE = 0.50
HAND_PRESENCE_CONFIDENCE = 0.50
HAND_TRACKING_CONFIDENCE = 0.50


# ============================================================
# BLINK SETTINGS
# ============================================================

BLINK_CLOSE_THRESHOLD = 0.20
BLINK_OPEN_THRESHOLD = 0.24

BLINK_MIN_TIME = 0.04
BLINK_MAX_TIME = 0.70

BLINK_COOLDOWN = 0.80


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# ============================================================
# CHECK MODELS
# ============================================================

print()
print("=" * 60)
print("                 VISION-X HILL CLIMB")
print("           REALISTIC HAND CONTROLLER")
print("=" * 60)
print()


if not HAND_MODEL.exists():

    print("ERROR: Hand Landmarker model missing:")
    print()
    print(HAND_MODEL)
    print()

    raise SystemExit


if not FACE_MODEL.exists():

    print("ERROR: Face Landmarker model missing:")
    print()
    print(FACE_MODEL)
    print()

    raise SystemExit


# ============================================================
# HAND LANDMARKER
# ============================================================

hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=
        HAND_DETECTION_CONFIDENCE,

    min_hand_presence_confidence=
        HAND_PRESENCE_CONFIDENCE,

    min_tracking_confidence=
        HAND_TRACKING_CONFIDENCE
)


# ============================================================
# FACE LANDMARKER
#
# IMPORTANT:
# Face is detected internally.
# NOTHING is drawn on the face.
# ============================================================

face_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL)
    ),

    running_mode=RunningMode.IMAGE,

    num_faces=1,

    min_face_detection_confidence=0.50,

    min_face_presence_confidence=0.50,

    min_tracking_confidence=0.50,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Hand Landmarker...")

hand_landmarker = HandLandmarker.create_from_options(
    hand_options
)

print("Hand Landmarker : READY")


print("Loading Face Landmarker...")

face_landmarker = FaceLandmarker.create_from_options(
    face_options
)

print("Face Landmarker : READY")


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not camera.isOpened():

    camera.release()

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    hand_landmarker.close()
    face_landmarker.close()

    raise SystemExit


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)

try:

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

except Exception:

    pass


# ============================================================
# HAND SKELETON
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# EYE LANDMARKS
# Used internally for blink.
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# STATE
# ============================================================

hand_detected = False

hand_position = "NO HAND"

hand_gesture = "NO HAND"

current_action = "RELEASE"

last_action = "RELEASE"


# ============================================================
# BLINK STATE
# ============================================================

eyes_closed = False

blink_start = 0.0

last_blink = 0.0

blink_message_until = 0.0

blink_count = 0


# ============================================================
# FPS
# ============================================================

fps = 0.0

fps_counter = 0

fps_timer = time.time()


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(

        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2

    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(
    landmarks,
    eye
):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(
        p2,
        p6
    )

    vertical_2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )

    if horizontal < 0.00001:

        return 0.0

    return (

        vertical_1 +
        vertical_2

    ) / (

        2.0 *
        horizontal

    )


# ============================================================
# BLINK DETECTION
# ============================================================

def process_blink(
    landmarks
):

    global eyes_closed
    global blink_start
    global last_blink
    global blink_message_until
    global blink_count

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE
    )

    ear = (
        left_ear +
        right_ear
    ) / 2.0

    now = time.time()


    # Eyes closing
    if ear < BLINK_CLOSE_THRESHOLD:

        if not eyes_closed:

            eyes_closed = True

            blink_start = now

        return


    # Eyes opening
    if ear > BLINK_OPEN_THRESHOLD:

        if eyes_closed:

            eyes_closed = False

            duration = (
                now -
                blink_start
            )


            if (

                BLINK_MIN_TIME
                <=
                duration
                <=
                BLINK_MAX_TIME

            ):

                if (

                    now -
                    last_blink
                    >=
                    BLINK_COOLDOWN

                ):

                    last_blink = now

                    blink_count += 1

                    blink_message_until = (
                        now + 0.7
                    )

                    print(
                        "BLINK -> ENTER"
                    )

                    press_enter()


# ============================================================
# ENTER
# ============================================================

def press_enter():

    try:

        pyautogui.press(
            "enter"
        )

    except Exception as error:

        print(
            "ENTER ERROR:",
            error
        )


# ============================================================
# FINGER COUNT
# ============================================================

def count_extended_fingers(
    landmarks
):

    count = 0


    # Index
    if landmarks[8].y < landmarks[6].y:

        count += 1


    # Middle
    if landmarks[12].y < landmarks[10].y:

        count += 1


    # Ring
    if landmarks[16].y < landmarks[14].y:

        count += 1


    # Little
    if landmarks[20].y < landmarks[18].y:

        count += 1


    return count


# ============================================================
# OPEN / CLOSED
# ============================================================

def detect_hand_gesture(
    landmarks
):

    fingers = count_extended_fingers(
        landmarks
    )


    if fingers <= 1:

        return "CLOSED"


    return "OPEN"


# ============================================================
# HAND POSITION
# ============================================================

def detect_hand_position(
    landmarks
):

    palm_x = (

        landmarks[0].x +
        landmarks[5].x +
        landmarks[9].x +
        landmarks[13].x +
        landmarks[17].x

    ) / 5.0


    if palm_x < LEFT_ZONE:

        return "LEFT"


    if palm_x > RIGHT_ZONE:

        return "RIGHT"


    return "CENTER"


# ============================================================
# ACTION
# ============================================================

def get_action(
    position,
    gesture
):

    if (

        position == "LEFT"
        and
        gesture == "CLOSED"

    ):

        return "BRAKE"


    if (

        position == "RIGHT"
        and
        gesture == "CLOSED"

    ):

        return "ACCELERATE"


    return "RELEASE"


# ============================================================
# RELEASE KEYS
# ============================================================

def release_keys():

    global current_action

    try:

        pyautogui.keyUp(
            "left"
        )

    except Exception:

        pass


    try:

        pyautogui.keyUp(
            "right"
        )

    except Exception:

        pass


    current_action = "RELEASE"


# ============================================================
# APPLY ACTION
# ============================================================

def apply_action(
    action
):

    global current_action
    global last_action


    if action == last_action:

        return


    last_action = action


    if action == "BRAKE":

        try:

            pyautogui.keyUp(
                "right"
            )

            pyautogui.keyDown(
                "left"
            )

        except Exception:

            pass


        current_action = "BRAKE"


    elif action == "ACCELERATE":

        try:

            pyautogui.keyUp(
                "left"
            )

            pyautogui.keyDown(
                "right"
            )

        except Exception:

            pass


        current_action = "ACCELERATE"


    else:

        release_keys()


# ============================================================
# TEXT
# ============================================================

def draw_text(
    frame,
    message,
    x,
    y,
    size=0.6,
    color=(255, 255, 255),
    thickness=2
):

    # Black outline/shadow

    cv2.putText(

        frame,

        message,

        (
            x + 2,
            y + 2
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        size,

        (0, 0, 0),

        thickness + 2,

        cv2.LINE_AA

    )


    # Main text

    cv2.putText(

        frame,

        message,

        (
            x,
            y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        size,

        color,

        thickness,

        cv2.LINE_AA

    )


# ============================================================
# REALISTIC HAND DRAWING
#
# RED + WHITE ONLY
#
# NO GREEN
# NO BLUE
# NO BOX
# ============================================================

def draw_hand(
    frame,
    landmarks
):

    height, width = frame.shape[:2]


    points = []


    # --------------------------------------------------------
    # Convert MediaPipe coordinates
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )

        points.append(
            (x, y)
        )


    # --------------------------------------------------------
    # WHITE SKELETON
    #
    # Thin and smooth for a more natural appearance.
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        if (

            start < len(points)
            and
            end < len(points)

        ):

            cv2.line(

                frame,

                points[start],

                points[end],

                (255, 255, 255),

                2,

                cv2.LINE_AA

            )


    # --------------------------------------------------------
    # RED LANDMARK JOINTS
    #
    # Small red circles.
    # --------------------------------------------------------

    for point in points:

        cv2.circle(

            frame,

            point,

            4,

            (0, 0, 255),

            -1,

            cv2.LINE_AA

        )


        # Very small white center highlight
        # makes the points look cleaner.

        cv2.circle(

            frame,

            point,

            1,

            (255, 255, 255),

            -1,

            cv2.LINE_AA

        )


# ============================================================
# TOP UI
#
# NO DIVIDER LINES
# ============================================================

def draw_top_ui(
    frame
):

    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # BRAKE
    # --------------------------------------------------------

    draw_text(

        frame,

        "BRAKE",

        25,

        55,

        0.65,

        (0, 0, 255),

        2

    )


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    center = "CENTER"

    center_size = cv2.getTextSize(

        center,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        2

    )[0]


    center_x = (

        width -
        center_size[0]
    ) // 2


    draw_text(

        frame,

        center,

        center_x,

        55,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # ACCELERATE
    # --------------------------------------------------------

    accelerate = "ACCELERATE"

    accelerate_size = cv2.getTextSize(

        accelerate,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        2

    )[0]


    accelerate_x = (

        width -
        accelerate_size[0] -
        25
    )


    draw_text(

        frame,

        accelerate,

        accelerate_x,

        55,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    draw_text(

        frame,

        f"FPS: {fps:.0f}",

        20,

        28,

        0.38,

        (255, 255, 255),

        1

    )


# ============================================================
# STATUS
#
# SMALL PANEL
# ============================================================

def draw_status(
    frame
):

    height, width = frame.shape[:2]


    panel_x = 18

    panel_y = height - 125

    panel_w = 335

    panel_h = 100


    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            panel_x,
            panel_y
        ),

        (
            panel_x + panel_w,
            panel_y + panel_h
        ),

        (0, 0, 0),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.50,

        frame,

        0.50,

        0,

        frame

    )


    # --------------------------------------------------------
    # HAND
    # --------------------------------------------------------

    hand_text = (

        "HAND: DETECTED"
        if hand_detected
        else
        "HAND: NOT DETECTED"

    )


    draw_text(

        frame,

        hand_text,

        panel_x + 12,

        panel_y + 25,

        0.42,

        (255, 255, 255),

        1

    )


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    draw_text(

        frame,

        f"POSITION: {hand_position}",

        panel_x + 12,

        panel_y + 50,

        0.42,

        (255, 255, 255),

        1

    )


    # --------------------------------------------------------
    # GESTURE
    # --------------------------------------------------------

    draw_text(

        frame,

        f"GESTURE: {hand_gesture}",

        panel_x + 12,

        panel_y + 75,

        0.42,

        (255, 255, 255),

        1

    )


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    draw_text(

        frame,

        f"ACTION: {current_action}",

        panel_x + 12,

        panel_y + 98,

        0.42,

        (255, 255, 255),

        1

    )


# ============================================================
# BLINK MESSAGE
# ============================================================

def draw_blink_message(
    frame
):

    if time.time() > blink_message_until:

        return


    height, width = frame.shape[:2]


    message = "BLINK -> ENTER"


    text_size = cv2.getTextSize(

        message,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        2

    )[0]


    x = (

        width -
        text_size[0]
    ) // 2


    draw_text(

        frame,

        message,

        x,

        95,

        0.65,

        (255, 255, 255),

        2

    )


# ============================================================
# HELP
# ============================================================

def draw_help(
    frame
):

    height, width = frame.shape[:2]


    message = (
        "BLINK = ENTER | "
        "LEFT CLOSED = BRAKE | "
        "RIGHT CLOSED = ACCELERATE | "
        "OPEN = RELEASE"
    )


    text_size = cv2.getTextSize(

        message,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.31,

        1

    )[0]


    x = (

        width -
        text_size[0]
    ) // 2


    draw_text(

        frame,

        message,

        x,

        height - 10,

        0.31,

        (255, 255, 255),

        1

    )


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 60)
print("CAMERA              : READY")
print("HAND LANDMARKER     : READY")
print("FACE TRACKING       : READY")
print("KEYBOARD CONTROLLER : READY")
print("=" * 60)
print()
print("RED  = HAND LANDMARKS")
print("WHITE = HAND SKELETON")
print()
print("NO GREEN")
print("NO BLUE DIVIDERS")
print("NO FACE LANDMARKS")
print()
print("Press Q or ESC to exit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = camera.read()


        if not success:

            continue


        # Mirror image
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # SMALL IMAGE FOR MEDIAPIPE
        # ----------------------------------------------------

        small = cv2.resize(

            frame,

            (
                PROCESS_WIDTH,
                PROCESS_HEIGHT
            ),

            interpolation=cv2.INTER_AREA

        )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(

            small,

            cv2.COLOR_BGR2RGB

        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb

        )


        # ====================================================
        # HAND
        # ====================================================

        hand_result = (
            hand_landmarker.detect(
                mp_image
            )
        )


        hand_detected = False

        hand_position = "NO HAND"

        hand_gesture = "NO HAND"

        action = "RELEASE"


        if hand_result.hand_landmarks:

            hand_detected = True


            landmarks = (
                hand_result.hand_landmarks[0]
            )


            # Position
            hand_position = (
                detect_hand_position(
                    landmarks
                )
            )


            # Open/closed
            hand_gesture = (
                detect_hand_gesture(
                    landmarks
                )
            )


            # Action
            action = get_action(

                hand_position,

                hand_gesture

            )


            # Draw hand
            #
            # ONLY RED + WHITE
            draw_hand(

                frame,

                landmarks

            )


        else:

            action = "RELEASE"


        # ====================================================
        # FACE
        #
        # INTERNAL BLINK DETECTION ONLY
        #
        # NOTHING IS DRAWN.
        # ====================================================

        face_result = (
            face_landmarker.detect(
                mp_image
            )
        )


        if face_result.face_landmarks:

            face_landmarks = (
                face_result.face_landmarks[0]
            )


            process_blink(
                face_landmarks
            )


        else:

            eyes_closed = False


        # ====================================================
        # KEYBOARD
        # ====================================================

        apply_action(
            action
        )


        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        now = time.time()

        elapsed = (
            now -
            fps_timer
        )


        if elapsed >= 1.0:

            fps = (
                fps_counter /
                elapsed
            )

            fps_counter = 0

            fps_timer = now


        # ====================================================
        # UI
        # ====================================================

        draw_top_ui(
            frame
        )

        draw_status(
            frame
        )

        draw_blink_message(
            frame
        )

        draw_help(
            frame
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            WINDOW_NAME,

            frame

        )


        # ====================================================
        # KEYBOARD EXIT
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key in (
            ord("q"),
            ord("Q"),
            27
        ):

            break


# ============================================================
# SHUTDOWN
# ============================================================

except KeyboardInterrupt:

    print()
    print("Controller interrupted.")


finally:

    print()
    print("Stopping controller...")

    release_keys()

    camera.release()

    cv2.destroyAllWindows()


    try:

        hand_landmarker.close()

    except Exception:

        pass


    try:

        face_landmarker.close()

    except Exception:

        pass


    print("Controller stopped.")