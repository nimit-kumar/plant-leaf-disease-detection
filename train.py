import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Step 1: Paths
DATA_DIR = r"C:\Users\nimit\Music\.vscode\machine_leaning\Plant_leave_diseases_dataset_with_augmentation"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
NUM_CLASSES = 39

# Step 2: Data Augmentation and Split
train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"Class names: {class_names}")

# Step 3: Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train Model
print("Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    verbose=1
)

# Step 5: Evaluate Model
loss, acc = model.evaluate(val_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {acc}')

# Step 6: Save Model PROPERLY
model.save('plant_leaf_disease_cnn_model.h5')
print("Model saved as plant_leaf_disease_cnn_model.h5")

# Also save class names for prediction
np.save('class_names.npy', class_names)
print("Class names saved as class_names.npy")
