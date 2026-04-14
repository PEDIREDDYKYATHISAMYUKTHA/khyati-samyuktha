import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ================================================================
#  PATHS
# ================================================================

# Original dataset paths
orig_train = "datasets/training_set"
orig_test  = "datasets/testing_set"

# Clean dataset paths (face-cropped — created automatically below)
train_dir  = "datasets/clean_training"
test_dir   = "datasets/clean_testing"


# ================================================================
#  STEP 1: CLEAN DATASET — Auto crop face from every image
# ================================================================

import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face_and_save(src_dir, dst_dir):

    total   = 0
    cropped = 0
    skipped = 0

    classes = [d for d in os.listdir(src_dir)
               if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:

        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        images = [f for f in os.listdir(src_cls)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"  {cls}: {len(images)} images...")

        for img_file in images:

            total    += 1
            src_path  = os.path.join(src_cls, img_file)
            dst_path  = os.path.join(dst_cls, img_file)

            if os.path.exists(dst_path):
                cropped += 1
                continue

            img = cv2.imread(src_path)
            if img is None:
                skipped += 1
                continue

            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                pad = int(max(w, h) * 0.25)
                ih, iw = img.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(iw, x + w + pad)
                y2 = min(ih, y + h + pad)
                face = cv2.resize(img[y1:y2, x1:x2], (224, 224))
                cv2.imwrite(dst_path, face)
                cropped += 1
            else:
                cv2.imwrite(dst_path, cv2.resize(img, (224, 224)))
                skipped += 1

    print(f"  ✅ Total: {total} | Cropped: {cropped} | No face: {skipped}")

print("\n" + "="*55)
print("STEP 1: Cleaning dataset...")
print("="*55)
print("Training set:")
crop_face_and_save(orig_train, train_dir)
print("Testing set:")
crop_face_and_save(orig_test, test_dir)
print("✅ Done!\n")


# ================================================================
#  DATA GENERATORS
# ================================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("\n✅ Classes   :", train_data.class_indices)
print(f"✅ Training  : {train_data.samples} images")
print(f"✅ Validation: {val_data.samples} images")
n_cls = train_data.num_classes


# ================================================================
#  BUILD MODEL
# ================================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = BatchNormalization()(x)
x      = Dense(256, activation="relu")(x)
x      = Dropout(0.6)(x)          # FIX: higher dropout prevents Oval bias
x      = Dense(128, activation="relu")(x)
x      = Dropout(0.5)(x)
output = Dense(n_cls, activation="softmax")(x)

model  = Model(inputs=base_model.input, outputs=output)
print(f"✅ Model params: {model.count_params():,}\n")


# ================================================================
#  PHASE 1 — Train top layers
# ================================================================

# FIX: Label smoothing stops model from being overconfident on Oval
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

print("="*55)
print("PHASE 1: Training top layers (10 epochs)...")
print("="*55)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# ================================================================
#  PHASE 2 — Fine-tune last 40 layers
# ================================================================

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00003),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        "face_shape_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n" + "="*55)
print("PHASE 2: Fine-tuning last 40 layers (60 epochs)...")
print("="*55)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=60,
    callbacks=callbacks
)


# ================================================================
#  RESULTS
# ================================================================

best_val = max(history.history["val_accuracy"]) * 100

print("\n" + "="*55)
print(f"✅ Training complete!")
print(f"✅ Best validation accuracy : {best_val:.2f}%")
print(f"✅ Model saved              : face_shape_model.h5")
print("="*55)

# Per-class accuracy
print("\nPer-class accuracy:")
val_data.reset()
preds    = model.predict(val_data, verbose=0)
pred_cls = np.argmax(preds, axis=1)
true_cls = val_data.classes
names    = list(val_data.class_indices.keys())

for i, name in enumerate(names):
    mask    = true_cls == i
    correct = np.sum(pred_cls[mask] == i)
    total   = np.sum(mask)
    acc     = correct / total * 100 if total > 0 else 0
    status  = "✅" if acc >= 70 else "⚠️"
    print(f"  {status} {name:<10}: {correct}/{total} = {acc:.1f}%")