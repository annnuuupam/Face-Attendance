import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

def load_cascade(cascade_path):
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

def load_saved_data(names_path, faces_path):
    with open(names_path, 'rb') as f_names, open(faces_path, 'rb') as f_faces:
        labels = pickle.load(f_names)
        faces = pickle.load(f_faces)
    return labels, faces

def initialize_knn(faces, labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    return knn

def load_background_image(background_path):
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background image not found at {background_path}")
    img_background = cv2.imread(background_path)
    if img_background is None:
        raise IOError(f"Failed to load background image from {background_path}")
    return img_background

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    webcam_index = 0  # Change this to the index of your Logitech C270 HD Webcam
    video = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    
    if not video.isOpened():
        raise IOError(f"Failed to open webcam with index {webcam_index}")
    
    cascade_path = r'face attendance\haarcascade_frontalface_default .xml'
    facedetect = load_cascade(cascade_path)
    
    names_path = r'data/names.pkl'
    faces_path = r'data/faces_data.pkl'
    LABELS, FACES = load_saved_data(names_path, faces_path)
    
    print('Shape of Faces matrix --> ', FACES.shape)
    
    knn = initialize_knn(FACES, LABELS)
    
    background_path = r'face attendance\background.png'
    img_background = load_background_image(background_path)
    
    COL_NAMES = ['NAME', 'TIME']
    attendance_directory = "Attendance"
    ensure_directory_exists(attendance_directory)
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            
            attendance = [str(output[0]), str(timestamp)]
            
            # Drawing rectangles and text on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        img_background[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", img_background)
        
        key = cv2.waitKey(1)
        if key == ord('o'):
            speak("Attendance Taken..")
            time.sleep(5)
            
            attendance_file = f"{attendance_directory}/Attendance_{date}.csv"
            exist = os.path.isfile(attendance_file)
            
            with open(attendance_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
        
        elif key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
