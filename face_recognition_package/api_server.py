from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from face_recognition_module import FaceRecognitionSystem

app = Flask(__name__)
face_system = FaceRecognitionSystem()

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    try:
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Convert base64 to image
        image = base64_to_image(data['image'])
        
        # Process image
        faces, _ = face_system.process_frame(image)
        
        # Draw results on image if requested
        if data.get('draw_results', False):
            for face in faces:
                x, y, w, h = face['box']
                role = face['role']
                name = face['name']
                
                # Draw bounding box
                color = (0, 0, 255) if role == 'Criminal' else (0, 255, 0)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{role}: {name}"
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert result image to base64
            result_image = image_to_base64(image)
        else:
            result_image = None
        
        # Prepare response
        response = {
            'faces': faces,
            'image': result_image
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API is running"""
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 