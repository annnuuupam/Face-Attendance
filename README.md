Facial Recognition Attendance System | Python

This project implements an automatic attendance system that leverages facial recognition technology to streamline the attendance process. Built using Python, the system utilizes OpenCV for real-time image processing and face detection, while the K-Nearest Neighbors (KNN) algorithm from scikit-learn is used to classify and recognize faces.

Key Features:

    *Face Detection & Recognition: Utilizes OpenCV and Haar cascades to detect and recognize faces from a webcam.

    *Machine Learning Classification: Employs the KNN algorithm to identify individuals based on previously captured face data.

    *Real-Time Attendance Logging: Automatically records attendance in CSV files with timestamped entries for each recognized individual.

    *Text-to-Speech Interaction: Provides real-time verbal feedback using the win32com library to inform users when attendance is recorded.

    *User-Friendly Interface: A seamless and intuitive interface that makes the system easy to operate and responsive in real-time.



Project Components:

    *OpenCV for video capture and face detection.

    *scikit-learn's KNN for facial recognition and classification.

    *pickle for storing face data and labels.

    *CSV for tracking attendance records.

    *win32com for voice feedback.
    

    

---

## Components Required

### 1. **Python 3.x**
   Python is the core programming language used for this project. Make sure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/).

### 2. **OpenCV (cv2)**
   OpenCV is a powerful computer vision library used for image processing and real-time face detection. It handles webcam input, face detection, and drawing on frames.
   - Install it via pip:
     ```bash
     pip install opencv-python
     ```

### 3. **NumPy**
   NumPy is used for efficient numerical operations and matrix handling. It is crucial for reshaping face data into a form that can be used by the classifier.
   - Install it via pip:
     ```bash
     pip install numpy
     ```

### 4. **scikit-learn**
   scikit-learn is a machine learning library. In this project, it is used to implement the **K-Nearest Neighbors (KNN)** algorithm to classify and recognize faces based on the dataset.
   - Install it via pip:
     ```bash
     pip install scikit-learn
     ```

### 5. **Pickle**
   Pickle is used for serializing (saving) and deserializing (loading) the face data and label data (names). It allows the system to store trained models and face data to be used later without retraining.
   - Install it via pip (though it's part of the standard library in Python):
     ```bash
     pip install pickle-mixin
     ```

### 6. **Haar Cascades for Face Detection**
   Haar Cascades are pre-trained classifiers used by OpenCV for detecting objects in an image. In this case, we use the **haarcascade_frontalface_default.xml** for face detection.
   - This file is available in the OpenCV repository and can be downloaded from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

### 7. **CSV**
   CSV (Comma-Separated Values) is used for saving attendance records. The project uses Python's built-in CSV module to record the recognized face with a timestamp in a `.csv` file.

### 8. **win32com.client (pywin32)**
   This library provides text-to-speech functionality, allowing the system to verbally notify users when their attendance is recorded. It uses the **SAPI (Speech API)** in Windows.
   - Install it via pip:
     ```bash
     pip install pywin32
     ```

### 9. **Webcam (Logitech C270 HD Webcam or any equivalent)**
   A webcam is necessary for capturing real-time video feed. The system uses the webcam to capture and process live frames to detect and recognize faces.

### 10. **Background Image**
   A background image is used to overlay the live webcam feed onto a custom UI. This enhances the user experience by showing a more polished interface.

### 11. **Haar Cascade Classifier XML File**
   The XML file (`haarcascade_frontalface_default.xml`) is required to detect faces in the live feed. It's part of OpenCV's pre-trained models and is crucial for the facial recognition system to function properly.

---

### Installation of Dependencies

You can install the required Python libraries in one go by creating a `requirements.txt` file with the following contents:

```
opencv-python
numpy
scikit-learn
pickle-mixin
pywin32
```

Then, you can install them using:
```bash
pip install -r requirements.txt
```

---

This detailed list of components will help users understand what they need to set up and run the **Facial Recognition Attendance System** smoothly.
