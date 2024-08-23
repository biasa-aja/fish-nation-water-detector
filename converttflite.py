# Load the trained model
model = tf.keras.models.load_model('water_detection_model.h5')

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a file
with open('water_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
