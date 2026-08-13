import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press ESC to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Vision Hill Climb - Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()