import cv2
import pickle
import numpy as np
import os

def capture_faces(webcam_index):
    # Ensure correct path for Haar cascade file
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Using cascade file at: {cascade_path}")

    # Load Haar cascade
    facedetect = cv2.CascadeClassifier(cascade_path)
    if facedetect.empty():
        print(f"Error loading cascade file from {cascade_path}")
        return

    video = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)  # Use CAP_DSHOW for faster capture on Windows

    faces_data = []
    count = 0
    max_images = 200  # Maximum number of images to capture

    name = input("Enter Your Name: ")

    while count < max_images:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            crop_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            faces_data.append(crop_img)
            count += 1

            cv2.putText(frame, f"Captured: {count}/{max_images}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(-1, 50*50*3)  # Flatten to a 1D array per image

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Save names
    names_file = os.path.join(data_dir, 'names.pkl')
    if not os.path.exists(names_file):
        names = [name] * len(faces_data)
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * len(faces_data)
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)

    # Save faces data
    faces_file = os.path.join(data_dir, 'faces_data.pkl')
    if not os.path.exists(faces_file):
        with open(faces_file, 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open(faces_file, 'wb') as f:
            pickle.dump(faces, f)

if __name__ == "__main__":
    # Specify the index of your Logitech C270 HD Webcam
    webcam_index = 0  # Adjust this index based on your system configuration
    
    capture_faces(webcam_index)
