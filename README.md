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