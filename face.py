import threading
import cv2
import face_recognition

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load the reference image and compute its encoding
reference_img = face_recognition.load_image_file("reference.jpg")
reference_encoding = face_recognition.face_encodings(reference_img)[0]

# Thread-safe lock for `face_match`
lock = threading.Lock()

# Function to check if a face matches the reference image
def check_face(frame):
    global face_match
    try:
        # Convert the frame to RGB (face_recognition uses RGB format)
        rgb_frame = frame[:, :, ::-1]
        # Detect face encodings in the current frame
        face_encodings = face_recognition.face_encodings(rgb_frame)

        # Compare detected encodings with the reference encoding
        for encoding in face_encodings:
            match = face_recognition.compare_faces([reference_encoding], encoding, tolerance=0.6)
            with lock:
                face_match = any(match)
    except Exception as e:
        with lock:
            face_match = False
        print(f"Error in check_face: {e}")

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:  # Run face recognition every 30 frames
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception as e:
                print(f"Error starting thread: {e}")

        counter += 1

        # Display match or no match on the video feed
        with lock:
            if face_match:
                cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
