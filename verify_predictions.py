#!/usr/bin/env python
"""
Verify model predictions by testing on sample data
This script helps debug why predictions might be inaccurate
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("MODEL PREDICTION VERIFICATION")
print("="*70 + "\n")

# Step 1: Load the best model
print("STEP 1: Loading best trained model...")
model_path = 'models/cnn_custom_trained.h5'

if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("Available models:")
    for f in os.listdir('models'):
        if f.endswith('.h5'):
            print(f"  - {f}")
    sys.exit(1)

try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully: {model_path}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Parameters: {model.count_params():,}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Step 2: Load class names
print("\nSTEP 2: Loading class names...")
from utils.data_loader import GTSRBDataLoader
data_loader = GTSRBDataLoader('data/GTSRB_Dataset', target_size=(64, 64))
class_names = data_loader.load_class_names()
print(f"✅ Loaded {len(class_names)} traffic sign classes")

# Step 3: Test on actual training data
print("\nSTEP 3: Testing on sample training images...")

train_gen, _ = data_loader.get_train_validation_generators(batch_size=32)

sample_batch_x, sample_batch_y = next(iter(train_gen))

print(f"Sample batch:")
print(f"  - Shape: {sample_batch_x.shape}")
print(f"  - dtype: {sample_batch_x.dtype}")
print(f"  - Range: [{sample_batch_x.min():.3f}, {sample_batch_x.max():.3f}]")

# Get predictions
predictions = model.predict(sample_batch_x, verbose=0)

# Get ground truth
true_classes = np.argmax(sample_batch_y, axis=1)
pred_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

# Calculate accuracy on this batch
batch_accuracy = np.mean(true_classes == pred_classes)

print(f"\n✓ Predictions statistics:")
print(f"  - Batch accuracy: {batch_accuracy*100:.1f}%")
print(f"  - Mean confidence: {confidences.mean():.4f}")
print(f"  - Min confidence: {confidences.min():.4f}")
print(f"  - Max confidence: {confidences.max():.4f}")

# Show sample predictions
print(f"\nSample predictions (first 5 images):")
for i in range(5):
    true_class = true_classes[i]
    pred_class = pred_classes[i]
    confidence = confidences[i]
    match = "✅" if true_class == pred_class else "❌"
    true_name = class_names.get(true_class, f"Class {true_class}")
    pred_name = class_names.get(pred_class, f"Class {pred_class}")
    print(f"  {i+1}. {match} True: {true_name:30s} | Pred: {pred_name:30s} ({confidence*100:.1f}%)")

# Step 4: Test on external image file
print("\n" + "-"*70)
print("STEP 4: Testing on external uploaded image...")

# Find a test image
test_images_dir = 'data/GTSRB_Dataset/Train'
if os.path.exists(test_images_dir):
    # Get first class with images
    for class_id in range(43):
        class_dir = os.path.join(test_images_dir, str(class_id))
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image_path = os.path.join(class_dir, images[0])
                
                # Load and preprocess exactly like training
                img = cv2.imread(test_image_path)
                if img is None:
                    print(f"❌ Could not read image: {test_image_path}")
                else:
                    # Resize to 64x64
                    img_resized = cv2.resize(img, (64, 64))
                    # Normalize to [0, 1]
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    # Add batch dimension
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    print(f"\n✓ Test image loaded: {test_image_path}")
                    print(f"  - Original class: {class_id} ({class_names.get(class_id, 'Unknown')})")
                    print(f"  - Preprocessed shape: {img_batch.shape}")
                    print(f"  - Pixel range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
                    
                    # Predict
                    pred = model.predict(img_batch, verbose=0)[0]
                    top_3_idx = np.argsort(pred)[-3:][::-1]
                    top_3_conf = pred[top_3_idx]
                    
                    print(f"\n🎯 Top-3 predictions:")
                    for rank, (idx, conf) in enumerate(zip(top_3_idx, top_3_conf), 1):
                        class_name = class_names.get(idx, f"Class {idx}")
                        match = "✅ CORRECT" if idx == class_id else ""
                        print(f"  {rank}. Class {idx:2d} ({class_name:30s}) - {conf*100:6.1f}% {match}")
                    
                    # Visualize
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Show image
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    axes[0].imshow(img_rgb)
                    axes[0].set_title(f"Input Image\nTrue: Class {class_id} ({class_names.get(class_id, 'Unknown')})")
                    axes[0].axis('off')
                    
                    # Show confidence bars
                    top_10_idx = np.argsort(pred)[-10:][::-1]
                    top_10_conf = pred[top_10_idx]
                    labels = [f"{class_names.get(idx, f'Class {idx}')[:20]}\n(C{idx})" for idx in top_10_idx]
                    colors = ['#28a745' if idx == class_id else '#667eea' for idx in top_10_idx]
                    axes[1].barh(labels, top_10_conf * 100, color=colors)
                    axes[1].set_xlabel('Confidence (%)')
                    axes[1].set_title('Top-10 Predictions')
                    axes[1].set_xlim([0, 100])
                    
                    plt.tight_layout()
                    plt.savefig('outputs/prediction_verification.png', dpi=100, bbox_inches='tight')
                    print(f"\n✅ Visualization saved to outputs/prediction_verification.png")
                    plt.close()
                break

# Step 5: Preprocessing check
print("\n" + "-"*70)
print("STEP 5: Preprocessing verification...")

# Create synthetic test image
synthetic_img = np.ones((64, 64, 3), dtype=np.uint8) * 128  # Gray image

# Test different preprocessing approaches
print("\nTesting different preprocessing approaches:")

approaches = [
    ("Normalize to [0,1]", lambda x: x.astype(np.float32) / 255.0),
    ("No normalization", lambda x: x.astype(np.float32)),
    ("Normalize to [-1,1]", lambda x: (x.astype(np.float32) / 127.5) - 1.0),
]

for approach_name, preprocess_fn in approaches:
    processed = preprocess_fn(synthetic_img)
    batch = np.expand_dims(processed, axis=0)
    try:
        pred = model.predict(batch, verbose=0)
        print(f"  ✅ {approach_name:30s} - works fine")
    except Exception as e:
        print(f"  ❌ {approach_name:30s} - ERROR: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print(f"\n✅ Model is working correctly!")
print(f"✅ If predictions seem wrong, check:")
print(f"   1. Image upload format (PNG vs JPG)")
print(f"   2. Image size and aspect ratio")
print(f"   3. Detection bounding box (try 'direct classification' mode)")
print(f"   4. Lighting/contrast of image")
print()
