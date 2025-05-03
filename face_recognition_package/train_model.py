import cv2
import numpy as np
import os
import json
from pathlib import Path

def train_face_recognizer(dataset_dir='face-dataset', model_dir='models'):
    """
    Train a face recognition model using images from the dataset directory.
    
    Args:
        dataset_dir (str): Directory containing face images organized by role and person
        model_dir (str): Directory to save the trained model and label map
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize face detector and recognizer
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Initialize lists for training data
    faces = []
    labels = []
    label_map = {}
    current_id = 0
    
    # Process each role directory (Criminals, Employees)
    for role in ['Criminals', 'Employees']:
        role_dir = os.path.join(dataset_dir, role.lower())
        if not os.path.exists(role_dir):
            print(f"Warning: {role_dir} directory not found")
            continue
            
        # Process each person's directory
        for person_name in os.listdir(role_dir):
            person_dir = os.path.join(role_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            # Add person to label map
            label_map[current_id] = {
                'name': person_name,
                'role': role
            }
            
            # Process each image
            for image_name in os.listdir(person_dir):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                detected_faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                
                # Add each detected face to training data
                for (x, y, w, h) in detected_faces:
                    face_roi = gray[y:y+h, x:x+w]
                    faces.append(face_roi)
                    labels.append(current_id)
                    
            current_id += 1
    
    if not faces:
        print("Error: No faces found in the dataset")
        return
        
    # Train the recognizer
    print(f"Training on {len(faces)} faces...")
    recognizer.train(faces, np.array(labels))
    
    # Save the model and label map
    model_path = os.path.join(model_dir, 'face_recognizer.yml')
    label_map_path = os.path.join(model_dir, 'label_map.json')
    
    recognizer.save(model_path)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)
        
    print(f"Model saved to {model_path}")
    print(f"Label map saved to {label_map_path}")
    print(f"Trained on {len(label_map)} people:")
    for id, info in label_map.items():
        print(f"- {info['name']} ({info['role']})")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Set default paths relative to the script directory
    dataset_dir = os.path.join(script_dir, 'face-dataset')
    model_dir = os.path.join(script_dir, 'models')
    
    # Train the model
    train_face_recognizer(dataset_dir, model_dir) 