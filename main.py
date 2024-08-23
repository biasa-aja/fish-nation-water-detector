from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="water_detection_model_optimized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    try:
        # Preprocess the image
        image = image.convert('RGB').resize((150, 150))
        input_data = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    print(file, 'FILE')

    if file and file.filename != '':
        try:
            # Open the image
            image = Image.open(io.BytesIO(file.read()))

            # Predict using the model
            prediction = predict(image)

            # Return result
            if prediction > 0.5:
                return jsonify({"water_detected": True}), 200
            else:
                return jsonify({"water_detected": False}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid image"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
