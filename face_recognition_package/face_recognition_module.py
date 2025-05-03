import cv2
import numpy as np
import json
import os
import time

class FaceRecognitionSystem:
    def __init__(self, model_dir='models'):
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(os.path.join(model_dir, 'face_recognizer.yml'))
        
        # Load label map
        with open(os.path.join(model_dir, 'label_map.json'), 'r') as f:
            self.label_map = json.load(f)
        
        # List of known people
        self.known_people = ['Harsha', 'Khushii']

    def preprocess_face(self, face_img):
        """Preprocess face image for better recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray

    def recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        results = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label_id, confidence = self.recognizer.predict(roi_gray)
            label_info = self.label_map.get(str(label_id)) or self.label_map.get(int(label_id))
            
            if label_info and label_info['name'] in self.known_people:
                role = label_info['role']
                name = label_info['name']
            else:
                role = 'Unknown'
                name = 'Unknown'
                
            alert = (role == 'Criminal' or role == 'Unknown')
            results.append({
                'box': (x, y, w, h),
                'role': role,
                'name': name,
                'confidence': confidence,
                'alert': alert
            })
        return results

    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        faces = self.recognize_faces(frame)
        return faces, []  # Empty list for weapons as we're not detecting them 