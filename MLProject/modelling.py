# modelling.py untuk MLflow Project & GitHub Actions
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

# Matikan warnings untuk cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("="*60)
print("MLFLOW PROJECT: BUNGA CLASSIFICATION TRAINING")
print("="*60)
print(f"Parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")

# Konfigurasi
DATASET_PATH = "datasetbunga_preprocessing"
IMG_SIZE = (224, 224)

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset tidak ditemukan di {DATASET_PATH}")
    sys.exit(1)

# Load data
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Jumlah kelas: {NUM_CLASSES}")

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=args.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup MLflow
mlflow_dir = os.path.join(os.getcwd(), "mlflow_experiments")
os.makedirs(mlflow_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlflow_dir}")

experiment_name = f"CI_Bunga_Classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)

# Autolog TANPA parameter log_tensorboard
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="GitHub_Actions_Run") as run:
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("num_classes", NUM_CLASSES)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
    
    print("\n🚀 Memulai training...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"\n✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✅ Validation Loss: {val_loss:.4f}")
    
    mlflow.log_metric("final_val_accuracy", val_accuracy)
    mlflow.log_metric("final_val_loss", val_loss)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    mlflow.log_artifact('training_history.png')
    plt.close()
    
    model.save("bunga_classification_model.h5")
    mlflow.log_artifact("bunga_classification_model.h5")
    
    print(f"\n✅ Run ID: {run.info.run_id}")
    print(f"✅ Experiment: {experiment_name}")

print("\n🎉 Training completed successfully!")