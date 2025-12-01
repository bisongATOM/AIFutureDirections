# Step 1: Install and import dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Step 2: Load and preprocess data
!wget https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip
!unzip -q dataset-resized.zip

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_ds = datagen.flow_from_directory(
    'dataset-resized',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_ds = datagen.flow_from_directory(
    'dataset-resized',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Build lightweight model (MobileNetV2-inspired)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(6, activation='softmax')  # 6 classes in TrashNet
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Step 5: Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
with open('recycle_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Step 6: Evaluate TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test on a batch
test_image, test_label = next(val_ds)
input_data = np.array(test_image[0:1], dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

print("TFLite prediction:", np.argmax(tflite_output))
print("True label:", np.argmax(test_label[0]))
