# Image Classification with VGG16

This project performs multi-class image classification using a pre-trained **VGG16** model with light fine-tuning. The model uses data augmentation, class weighting for imbalanced datasets, and provides detailed performance metrics including accuracy, precision, recall, F1-score, specificity, confusion matrix, and a classification report.

---

## Dataset

The dataset should follow a folder structure compatible with `flow_from_directory`:

```
dataset_path/
    class_1/
        img1.jpg
        img2.jpg
        ...
    class_2/
        img1.jpg
        img2.jpg
        ...
```

* 20% of the data is used for validation.
* Images are resized to `(160, 160)` for VGG16 input.

---

## Configuration

```python
dataset_path = r"C:\Users\rsshr\OneDrive\Desktop\TARP PEPPER\archive"
img_size = (160, 160)
batch_size = 32
```

---

## Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)
```

---

## Model Architecture

* Base: **VGG16** pretrained on ImageNet (top layers removed)
* First 10 layers frozen, rest trainable
* Custom classifier head:

  * Global Average Pooling
  * Batch Normalization
  * Dense(256, ReLU)
  * Dropout(0.3)
  * Output layer with softmax (number of classes)

```python
import tensorflow as tf

base_model = tf.keras.applications.VGG16(
    weights='imagenet', include_top=False, input_shape=(160, 160, 3)
)

for layer in base_model.layers[:10]:
    layer.trainable = False
for layer in base_model.layers[10:]:
    layer.trainable = True

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
```

---

## Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
```

---

## Compile and Callbacks

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
]
```

---

## Training

```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    class_weight=class_weights,
    callbacks=callbacks
)
```

---

## Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_pred_probs = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

acc = accuracy_score(y_true, y_pred_classes)
print("\nFinal Accuracy:", round(acc*100, 2), "%")

prec_w = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
rec_w = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
f1_w = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
prec_m = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
rec_m = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
f1_m = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)

cm = confusion_matrix(y_true, y_pred_classes)
specificity_per_class = []
for i in range(len(cm)):
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
    fp = np.sum(cm[:, i]) - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_per_class.append(specificity)
specificity = np.mean(specificity_per_class)

metrics_summary = {
    "Accuracy": acc,
    "Precision (Weighted)": prec_w,
    "Recall (Weighted)": rec_w,
    "F1-Score (Weighted)": f1_w,
    "Precision (Macro)": prec_m,
    "Recall (Macro)": rec_m,
    "F1-Score (Macro)": f1_m,
    "Specificity": specificity
}

print("\n================ Performance Metrics ================\n")
for k, v in metrics_summary.items():
    print(f"{k:25s}: {v:.4f}")

print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys())))
print("\nConfusion Matrix:\n", cm)
```

---

## Plot Accuracy and Loss

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

---

## How to Run

1. Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

2. Set the `dataset_path` variable to your dataset folder.
3. Run the script:

```bash
python train_vgg16.py
```

Adjust `batch_size`, `img_size`, and `learning_rate` as needed based on your dataset and GPU availability.

---

## Notes

* Suitable for small to medium-sized datasets.
* Can be extended to VGG19 or deeper fine-tuning.
* Includes automatic handling of class imbalance via class weights.
