#!/usr/bin/env python3
# --- FINGERPRINT RECOGNITION + SPOOF DETECTION (HIGH ACCURACY VERSION) ---

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras import layers
import hashlib
from PIL import Image

# ----------------- CONFIG -----------------
DATASET_PATH = r"D:\code\Projects\biometric-template-gen\data\CASIA-dataset"  # ✅ change if needed
IMG_HEIGHT, IMG_WIDTH = 224, 224          # Higher resolution for better detail
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
EXTEND_EPOCHS = 15
AE_EPOCHS = 3
TARGET_CLASS_COUNT = 150
MODEL_PATH = "high_accuracy_fingerprint_model.keras"
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Optional: use mixed precision if your GPU supports it
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("⚙ Mixed precision enabled for faster training.")
except:
    pass

# ----------------- GPU SETUP -----------------
print("🔍 Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Using GPU: {[gpu.name for gpu in gpus]}")
else:
    print("⚠ No GPU found. Using CPU (slower training).")

# ----------------- FILTER CLASSES -----------------
def get_limited_subfolders(path, max_classes=TARGET_CLASS_COUNT):
    classes = sorted(os.listdir(path))
    selected = random.sample(classes, min(max_classes, len(classes)))
    print(f"📦 Using {len(selected)} out of {len(classes)} total classes.")
    return [os.path.join(path, c) for c in selected]

allowed_classes = set([os.path.basename(p) for p in get_limited_subfolders(DATASET_PATH)])

# ----------------- DATA GENERATORS -----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=list(allowed_classes),
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=list(allowed_classes),
    subset='validation',
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes
print(f"✅ Training samples: {train_gen.samples}, Classes used: {num_classes}")

# ----------------- IMAGE HASH FUNCTION -----------------
def compute_image_hash(image_array):
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    return hashlib.sha256(image.tobytes()).hexdigest()

# ----------------- AUTOENCODER (for SPOOF DETECTION) -----------------
def build_autoencoder(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inp)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    encoded = layers.Flatten()(x)
    encoded = layers.Dense(256, activation='relu')(encoded)

    x = layers.Dense((IMG_HEIGHT//8)*(IMG_WIDTH//8)*128, activation='relu')(encoded)
    x = layers.Reshape((IMG_HEIGHT//8, IMG_WIDTH//8, 128))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    decoded = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)

    model = Model(inp, decoded, name="autoencoder")
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

print("🔧 Building autoencoder...")
autoencoder = build_autoencoder()

print("🧠 Training Autoencoder...")
def autoencoder_data_gen(gen):
    for batch_x, _ in gen:
        yield batch_x, batch_x  # input = output

autoencoder.fit(
    autoencoder_data_gen(val_gen),
    steps_per_epoch=len(val_gen),
    epochs=AE_EPOCHS,
    verbose=1
)

# ----------------- COMPUTE RECONSTRUCTION THRESHOLD -----------------
print("📏 Computing AE reconstruction threshold...")
recon_errors = []
val_gen.reset()
for batch_x, _ in val_gen:
    recon = autoencoder.predict(batch_x, verbose=0)
    err = np.mean(np.square(batch_x - recon), axis=(1, 2, 3))
    recon_errors.extend(err.tolist())
    if len(recon_errors) >= val_gen.samples:
        break

THRESHOLD = np.mean(recon_errors) + 3*np.std(recon_errors)
print(f"🧾 AE Threshold set at {THRESHOLD:.6f}")

# ----------------- CLASSIFIER (ResNet50 → 2048 features) -----------------
print("🧩 Building classifier (ResNet50 base with fine-tuning)...")

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Phase 1: Freeze most layers for stable learning
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------- CALLBACKS -----------------
class SpoofCheckLogger(Callback):
    def _init_(self, val_gen, ae_model, threshold):
        super()._init_()
        self.val_gen = val_gen
        self.ae_model = ae_model
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        images, _ = next(self.val_gen)
        first = images[0]
        img_hash = compute_image_hash(first)
        recon = self.ae_model.predict(first[np.newaxis, ...], verbose=0)
        mse = np.mean(np.square(first - recon))
        status = "⚠ SPOOF DETECTED" if mse > self.threshold else "✅ GENUINE"
        print(f"\nEpoch {epoch+1}: AE MSE={mse:.6f}, {status}, Hash={img_hash[:10]}...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
    SpoofCheckLogger(val_gen, autoencoder, THRESHOLD)
]

# ----------------- PHASE 1: FROZEN TRAINING -----------------
print("🚀 Starting Phase 1 training (frozen ResNet layers)...")
history = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

val_loss, val_acc = model.evaluate(val_gen)
print(f"\n📊 Phase 1 Accuracy: {val_acc*100:.2f}%")

# ----------------- PHASE 2: UNFREEZE ALL + FINE-TUNE -----------------
print("🎯 Fine-tuning full ResNet (unfreezing all layers)...")
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS + EXTEND_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ----------------- SAVE MODEL -----------------
print(f"💾 Saving final model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("✅ Model saved successfully.")

# ----------------- FINAL EVALUATION -----------------
final_loss, final_acc = model.evaluate(val_gen)
print(f"\n🏁 Final Validation Accuracy: {final_acc*100:.2f}% | Loss: {final_loss:.4f}")