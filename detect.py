import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Image Data Generator for loading and augmenting images
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load both water and non-water categories from the same parent directory
train_generator = train_datagen.flow_from_directory(
    'dataset/',  # Common parent directory for both classes
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/',  # Common parent directory for both classes
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the trained model in Keras format
model.save('water_detection_model.h5')

# Convert to TensorFlow Lite model (without optimization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the non-optimized model to a file
with open('water_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert the model with optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_optimized = converter.convert()

# Save the optimized model to a file
with open('water_detection_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model_optimized)

print("Model training and conversion to TFLite (with and without optimization) completed successfully.")
