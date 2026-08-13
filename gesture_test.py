import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Wrist landmark
        wrist = hand.landmark[0]

        x = wrist.x

        # Gesture zones
        if x < 0.40:
            gesture = "LEFT"

        elif x > 0.60:
            gesture = "RIGHT"

        else:
            gesture = "NEUTRAL"

        # Draw hand
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Convert normalized coordinate to screen position
        px = int(x * width)

        cv2.circle(
            frame,
            (px, 100),
            15,
            (0, 255, 0),
            -1
        )

    # Draw zones
    cv2.line(
        frame,
        (int(width * 0.40), 0),
        (int(width * 0.40), height),
        (255, 255, 0),
        2
    )

    cv2.line(
        frame,
        (int(width * 0.60), 0),
        (int(width * 0.60), height),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"GESTURE: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Vision Hill Climb - Gesture Controller",
        frame
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
hands.close()