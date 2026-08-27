import cv2

cap_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)

cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)


print("Recording... press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save to file
    out.write(frame)

    # Show preview
    cv2.imshow("Recording", frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

