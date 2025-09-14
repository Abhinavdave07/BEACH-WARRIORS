# Step 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 2: Define paths and parameters
# IMPORTANT: Update this path to where you stored your dataset on your computer!
dataset_path = 'C:/Users/mrabh/Downloads/waste/garbage_classification' # Example Windows path
# or on macOS/Linux: dataset_path = '/Users/YourName/Documents/garbage_classification'

# Model parameters
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

# Step 3: Load the data
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get the class names that Keras found
class_names = train_ds.class_names
print(f"\nFound the following classes: {class_names}")

# Step 4: Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Step 5: Build and compile the CNN Model
num_classes = len(class_names)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Show the model architecture
model.summary()

# Step 6: Train the model!
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Step 7: Visualize and evaluate results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

loss, accuracy = model.evaluate(val_ds)
print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")

train_loss, train_accuracy = model.evaluate(train_ds)
print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Training Loss: {train_loss:.4f}")

model.save('garbage_classifier.keras')
print("Model saved successfully as waste_classifier.keras!")