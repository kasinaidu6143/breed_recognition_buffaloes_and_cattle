import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP & GPU MEMORY CONFIGURATION ---
print("--- Configuring GPU for memory growth ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Success: Enabled memory growth for {len(gpus)} GPU(s)")
  except RuntimeError as e:
    print(e)
else:
    print("⚠️ No GPU detected. Running on CPU.")

# --- Use your ORIGINAL dataset path ---
DATASET_PATH = r"D:\DLPROJECT\dataset"

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 10
LEARNING_RATE_FINE_TUNE = 1e-5

# --- 2. ROBUST DATA VALIDATION & LOADING ---
print("\n--- Starting Robust Data Validation ---")

if not os.path.exists(DATASET_PATH):
    print(f"FATAL ERROR: The directory was not found at '{DATASET_PATH}'")
    sys.exit()

valid_image_paths = []
labels = []
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
class_to_index = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes: {class_names}")

for class_name in class_names:
    class_dir = os.path.join(DATASET_PATH, class_name)
    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)
        try:
            raw_image = tf.io.read_file(filepath)
            tf.io.decode_image(raw_image, channels=3)
            valid_image_paths.append(filepath)
            labels.append(class_to_index[class_name])
        except Exception:
            # This will quietly skip any remaining bad files, just in case.
            pass

print(f"\n--- Validation Complete ---")
print(f"Found {len(valid_image_paths)} valid images to train on.")

if not valid_image_paths:
    print("FATAL ERROR: No valid images were found.")
    sys.exit()

# --- Create TensorFlow Datasets ---
path_ds = tf.data.Dataset.from_tensor_slices(valid_image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(labels)
image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

AUTOTUNE = tf.data.AUTOTUNE
def process_path(path, label):
    raw_image = tf.io.read_file(path)
    image = tf.io.decode_image(raw_image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

dataset_size = len(valid_image_paths)
train_size = int(0.8 * dataset_size)
image_label_ds = image_label_ds.shuffle(buffer_size=dataset_size, seed=123)
train_ds = image_label_ds.take(train_size)
val_ds = image_label_ds.skip(train_size)

# FIX: Removed .cache() from both lines below to prevent "NOT_FOUND" errors.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print("INFO: Successfully created training and validation datasets.")

# --- 3. DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
], name="data_augmentation")

# --- 4. MODEL BUILDING ---
print("INFO: Building the model...")
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# --- 5. INITIAL TRAINING ---
print("\n--- Phase 1: STARTING INITIAL TRAINING ---")
history = model.fit(train_ds, epochs=EPOCHS_INITIAL, validation_data=val_ds)

# --- 6. FINE-TUNING ---
print("\n--- Phase 2: STARTING FINE-TUNING ---")
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
total_epochs = EPOCHS_INITIAL + EPOCHS_FINE_TUNE
history_fine_tune = model.fit(train_ds, epochs=total_epochs,
                              initial_epoch=history.epoch[-1],
                              validation_data=val_ds)

# --- 7. EVALUATION & VISUALIZATION ---
print("\n--- TRAINING COMPLETE. VISUALIZING RESULTS. ---")
acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history.history['loss'] + history_fine_tune.history['loss']
val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=EPOCHS_INITIAL-1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=EPOCHS_INITIAL-1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig("training_results.png")
plt.show()

# --- 8. SAVE THE FINAL MODEL ---
print("\n--- SAVING THE FINAL, TRAINED MODEL ---")
model.save("indian_cattle_breed_recognizer.h5")
print("✅ Model saved successfully.")




