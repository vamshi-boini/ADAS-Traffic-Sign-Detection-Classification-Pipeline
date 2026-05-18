#!/usr/bin/env python
"""
Fixed Training Script - Train all three models with proper model saving
FIXED: Each model saved to correct file, no overwrites
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time

# Setup
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("\n" + "="*70)
print("FIXED MODEL TRAINING - ALL THREE MODELS")
print("="*70 + "\n")

# Configuration
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20

# Load data loader
from utils.data_loader import GTSRBDataLoader

print("Initializing data loader...")
data_loader = GTSRBDataLoader('data/GTSRB_Dataset', target_size=(64, 64))

# Load class names
class_names = data_loader.load_class_names()
NUM_CLASSES = len(class_names)

print(f"[OK] Dataset loaded: {NUM_CLASSES} classes\n")

# Define model builders (from original train_all_models.py)
def build_custom_cnn(input_shape=(64, 64, 3), num_classes=43):
    """Build custom CNN architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_mobilenetv2(num_classes=43):
    """Build MobileNetV2 with transfer learning"""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_resnet50(num_classes=43):
    """Build ResNet50 with transfer learning"""
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(scale=255.0),
        tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Validation split
train_gen, val_gen = data_loader.get_train_validation_generators(batch_size=BATCH_SIZE)

print(f"[OK] Training samples: {train_gen.samples}")
print(f"[OK] Validation samples: {val_gen.samples}\n")

# Storage for comparison
results = {
    'model': [],
    'train_acc': [],
    'val_acc': [],
    'train_time': [],
    'file_path': [],
    'parameters': []
}

# ============================================================================
# MODEL 1: CUSTOM CNN
# ============================================================================
print("="*70)
print("MODEL 1: CUSTOM CNN")
print("="*70 + "\n")

print("Building Custom CNN...")
cnn_model = build_custom_cnn(input_shape=(64, 64, 3), num_classes=NUM_CLASSES)
cnn_model.summary()

print("\nTraining Custom CNN...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {train_gen.samples // BATCH_SIZE}")

start_time = time.time()
cnn_history = cnn_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1
)
cnn_train_time = time.time() - start_time

# Get final metrics
cnn_train_acc = cnn_history.history['accuracy'][-1]
cnn_val_acc = cnn_history.history['val_accuracy'][-1]
cnn_params = cnn_model.count_params()

# Save Custom CNN - FIXED PATH
cnn_save_path = 'models/cnn_custom_trained.h5'
cnn_model.save(cnn_save_path, overwrite=True)
print(f"\n[OK] Custom CNN saved to: {cnn_save_path}")
print(f"  Training accuracy: {cnn_train_acc:.4f}")
print(f"  Validation accuracy: {cnn_val_acc:.4f}")
print(f"  Training time: {cnn_train_time:.2f} seconds ({cnn_train_time/60:.1f} minutes)")
print(f"  Parameters: {cnn_params:,}\n")

results['model'].append('Custom CNN')
results['train_acc'].append(cnn_train_acc)
results['val_acc'].append(cnn_val_acc)
results['train_time'].append(cnn_train_time)
results['file_path'].append(cnn_save_path)
results['parameters'].append(cnn_params)

# ============================================================================
# MODEL 2: MobileNetV2 (TRANSFER LEARNING)
# ============================================================================
print("="*70)
print("MODEL 2: MobileNetV2 (TRANSFER LEARNING)")
print("="*70 + "\n")

# Reload data for fresh generators
print("Reloading data for MobileNetV2...")
train_gen, val_gen = data_loader.get_train_validation_generators(batch_size=BATCH_SIZE)

print("Building MobileNetV2...")
mobilenet_model = build_mobilenetv2(num_classes=NUM_CLASSES)
mobilenet_model.summary()

print("\nTraining MobileNetV2...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {train_gen.samples // BATCH_SIZE}")

start_time = time.time()
mobilenet_history = mobilenet_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1
)
mobilenet_train_time = time.time() - start_time

# Get final metrics
mobilenet_train_acc = mobilenet_history.history['accuracy'][-1]
mobilenet_val_acc = mobilenet_history.history['val_accuracy'][-1]
mobilenet_params = mobilenet_model.count_params()

# Save MobileNetV2 - FIXED PATH (DIFFERENT FROM CNN)
mobilenet_save_path = 'models/mobilenetv2_trained.h5'
mobilenet_model.save(mobilenet_save_path, overwrite=True)
print(f"\n[OK] MobileNetV2 saved to: {mobilenet_save_path}")
print(f"  Training accuracy: {mobilenet_train_acc:.4f}")
print(f"  Validation accuracy: {mobilenet_val_acc:.4f}")
print(f"  Training time: {mobilenet_train_time:.2f} seconds ({mobilenet_train_time/60:.1f} minutes)")
print(f"  Parameters: {mobilenet_params:,}\n")

results['model'].append('MobileNetV2')
results['train_acc'].append(mobilenet_train_acc)
results['val_acc'].append(mobilenet_val_acc)
results['train_time'].append(mobilenet_train_time)
results['file_path'].append(mobilenet_save_path)
results['parameters'].append(mobilenet_params)

# ============================================================================
# MODEL 3: ResNet50 (TRANSFER LEARNING)
# ============================================================================
print("="*70)
print("MODEL 3: ResNet50 (TRANSFER LEARNING)")
print("="*70 + "\n")

# Reload data for fresh generators
print("Reloading data for ResNet50...")
train_gen, val_gen = data_loader.get_train_validation_generators(batch_size=BATCH_SIZE)

print("Building ResNet50...")
resnet_model = build_resnet50(num_classes=NUM_CLASSES)
resnet_model.summary()

print("\nTraining ResNet50...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {train_gen.samples // BATCH_SIZE}")

start_time = time.time()
resnet_history = resnet_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1
)
resnet_train_time = time.time() - start_time

# Get final metrics
resnet_train_acc = resnet_history.history['accuracy'][-1]
resnet_val_acc = resnet_history.history['val_accuracy'][-1]
resnet_params = resnet_model.count_params()

# Save ResNet50 - FIXED PATH (DIFFERENT FROM OTHERS)
resnet_save_path = 'models/resnet50_trained.h5'
resnet_model.save(resnet_save_path, overwrite=True)
print(f"\n[OK] ResNet50 saved to: {resnet_save_path}")
print(f"  Training accuracy: {resnet_train_acc:.4f}")
print(f"  Validation accuracy: {resnet_val_acc:.4f}")
print(f"  Training time: {resnet_train_time:.2f} seconds ({resnet_train_time/60:.1f} minutes)")
print(f"  Parameters: {resnet_params:,}\n")

results['model'].append('ResNet50')
results['train_acc'].append(resnet_train_acc)
results['val_acc'].append(resnet_val_acc)
results['train_time'].append(resnet_train_time)
results['file_path'].append(resnet_save_path)
results['parameters'].append(resnet_params)

# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - COMPARISON")
print("="*70 + "\n")

# Results table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
csv_path = 'outputs/training_comparison_fixed.csv'
results_df.to_csv(csv_path, index=False)
print(f"\n[OK] Results saved to: {csv_path}")

# Recommend best model
best_idx = np.argmax(results_df['val_acc'].values)
best_model_name = results_df.loc[best_idx, 'model']
best_val_acc = results_df.loc[best_idx, 'val_acc']

print(f"\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"   File: {results_df.loc[best_idx, 'file_path']}")
print(f"\n   Use this model for inference and deployment!")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Validation accuracy comparison
ax = axes[0, 0]
bars = ax.bar(results_df['model'], results_df['val_acc'] * 100, color=['#667eea', '#764ba2', '#f093fb'])
ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax.set_title('Model Comparison - Validation Accuracy', fontweight='bold')
ax.set_ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Training accuracy
ax = axes[0, 1]
ax.bar(results_df['model'], results_df['train_acc'] * 100, color=['#667eea', '#764ba2', '#f093fb'])
ax.set_ylabel('Training Accuracy (%)', fontweight='bold')
ax.set_title('Model Comparison - Training Accuracy', fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Training time
ax = axes[1, 0]
ax.bar(results_df['model'], results_df['train_time'], color=['#667eea', '#764ba2', '#f093fb'])
ax.set_ylabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Model Comparison - Training Time', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Speed vs Accuracy
ax = axes[1, 1]
scatter = ax.scatter(results_df['train_time'], results_df['val_acc'] * 100, 
                    s=results_df['parameters']/10000, 
                    c=['#667eea', '#764ba2', '#f093fb'],
                    alpha=0.7, edgecolors='black', linewidth=2)
for idx, row in results_df.iterrows():
    ax.annotate(row['model'], 
               (row['train_time'], row['val_acc']*100),
               xytext=(5, 5), textcoords='offset points',
               fontweight='bold')
ax.set_xlabel('Training Time (seconds)', fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax.set_title('Speed vs Accuracy (bubble size = parameters)', fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_comparison_fixed.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Visualization saved to: outputs/training_comparison_fixed.png")
plt.close()

print("\n" + "="*70)
print("[SUCCESS] TRAINING COMPLETE - ALL MODELS TRAINED AND SAVED")
print("="*70)
print(f"\nNext step: Test predictions with evaluate_and_deploy.py")
print(f"Command: python evaluate_and_deploy.py\n")
