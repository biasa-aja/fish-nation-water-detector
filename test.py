import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="water_detection_model_optimized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load an image to test
image_path = input("Enter the path of the image: ")
try:
    image = Image.open(image_path).convert('RGB').resize((150, 150))
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

input_data = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

print(input_details[0]['index'], 'INPUT INDEX', input_data, 'input_data')

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Interpret the result
if output_data[0] > 0.5:
    print("Water detected!")
else:
    print("No water detected.")
