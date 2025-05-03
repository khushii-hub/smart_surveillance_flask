# Face Recognition Package

A simple Python package for face recognition using OpenCV's LBPH face recognizer. This package provides an easy-to-use interface for detecting and recognizing faces in images or video streams.

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python flask
   ```

## Training Your Own Model

Before using the face recognition system, you need to train it with your own dataset:

1. Create a dataset directory structure:
   ```
   face-dataset/
   ├── criminals/
   │   ├── person1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── person2/
   │       ├── image1.jpg
   │       └── ...
   └── employees/
       ├── person1/
       │   ├── image1.jpg
       │   └── ...
       └── person2/
           ├── image1.jpg
           └── ...
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

   This will:
   - Process all images in the dataset
   - Detect faces in each image
   - Train the recognition model
   - Save the model and label map to the `models` directory

## Usage

### Python Package

```python
from face_recognition_package import FaceRecognitionSystem

# Initialize the system
face_system = FaceRecognitionSystem()

# Process a frame
faces, _ = face_system.process_frame(frame)

# Each face contains:
# - box: (x, y, w, h) coordinates
# - role: 'Criminal', 'Employee', or 'Unknown'
# - name: Name of the person or 'Unknown'
# - confidence: Recognition confidence score
# - alert: True if role is 'Criminal' or 'Unknown'
```

### REST API

Start the API server:
```bash
python api_server.py
```

The server will run on `http://localhost:5000`

#### Endpoints

1. Check API Status
   ```
   GET /api/status
   Response: {"status": "running"}
   ```

2. Detect Faces
   ```
   POST /api/detect
   Request body:
   {
     "image": "base64_encoded_image",
     "draw_results": true/false  // Optional, defaults to false
   }
   
   Response:
   {
     "faces": [
       {
         "box": [x, y, w, h],
         "role": "Criminal/Employee/Unknown",
         "name": "Name/Unknown",
         "confidence": score,
         "alert": true/false
       },
       ...
     ],
     "image": "base64_encoded_result_image"  // Only if draw_results=true
   }
   ```

#### Example API Usage (JavaScript)

```javascript
// Convert image to base64
const image = document.getElementById('imageInput').files[0];
const reader = new FileReader();
reader.onload = function() {
    const base64Image = reader.result;
    
    // Send to API
    fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: base64Image,
            draw_results: true
        })
    })
    .then(response => response.json())
    .then(data => {
        // Display results
        if (data.image) {
            document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.image;
        }
        
        // Process face data
        data.faces.forEach(face => {
            console.log(`Detected ${face.name} (${face.role})`);
        });
    });
};
reader.readAsDataURL(image);
```

## Model Files

The system requires two model files:
1. `models/face_recognizer.yml` - The trained face recognition model
2. `models/label_map.json` - The mapping of labels to names and roles

Make sure these files are present in the `models` directory.

## Known People

The system currently recognizes:
- Harsha (Employee)
- Khushii (Criminal)

All other faces will be marked as "Unknown".

## Directory Structure

```
face_recognition_package/
├── __init__.py
├── face_recognition_module.py
├── example.py
├── train_model.py
├── api_server.py
├── README.md
├── models/           # Created after training
│   ├── face_recognizer.yml
│   └── label_map.json
└── face-dataset/    # Create this directory and add your images
    ├── criminals/
    └── employees/
```

## Notes

- The system uses OpenCV's LBPH face recognizer
- Face detection is done using Haar cascades
- The system includes preprocessing steps for better recognition
- Recognition results include bounding boxes, roles, and names
- You need to train the model with your own dataset before using it
- The API can be used by any web application that can make HTTP requests 