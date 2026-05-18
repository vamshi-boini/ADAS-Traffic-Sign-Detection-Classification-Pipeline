# ADAS Traffic Sign Pipeline - Implementation Guide

## 📋 Complete Setup & Execution Instructions

This guide walks you through building and deploying the complete ADAS two-stage pipeline project.

## Phase 1: Environment Setup (5 minutes)

### Step 1.1: Install Python & Dependencies

```bash
# Navigate to project directory
cd c:\Users\asus\OneDrive\Desktop\Guvi_Final_Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Expected Output:**
- Successfully installed 11+ packages (TensorFlow, Keras, OpenCV, etc.)
- No errors or conflicting dependencies

### Step 1.2: Verify Installation

```python
python -c "
import tensorflow as tf
import cv2
import numpy as np
print('✓ TensorFlow version:', tf.__version__)
print('✓ OpenCV version:', cv2.__version__)
print('✓ GPU Available:', tf.config.list_physical_devices('GPU'))
"
```

## Phase 2: Dataset Preparation (15 minutes)

### Step 2.1: Download GTSRB Dataset

1. Visit: http://benchmark.ini.rub.de/gtsrb_dataset.html
2. Download:
   - `GTSRB_Final_Training_Images.zip` (43 classes, 50,000+ images)
   - `GTSRB_Final_Test_Images.zip` (43 classes, 12,600+ images)

### Step 2.2: Extract Dataset

```bash
# Create data directory structure
mkdir data\GTSRB_Dataset

# Extract downloaded files
# Training set:
unzip GTSRB_Final_Training_Images.zip -d data\GTSRB_Dataset\Train

# Test set:
unzip GTSRB_Final_Test_Images.zip -d data\GTSRB_Dataset\Test
```

### Step 2.3: Verify Dataset Structure

```bash
# Check structure
tree data\GTSRB_Dataset / F

# Expected output:
# data/GTSRB_Dataset/
# ├── Train/
# │   ├── 0/  (Speed limit 20 km/h)
# │   ├── 1/  (Speed limit 30 km/h)
# │   ├── ...
# │   └── 42/ (End of speed limit)
# └── Test/
#     ├── 0/ ... 42/
```

**Expected Stats:**
- Train images: ~39,000
- Test images: ~12,600
- Classes: 43
- Total: ~51,600 images

## Phase 3: Run Jupyter Notebook (2-3 hours)

### Step 3.1: Start Jupyter

```bash
# Navigate to project directory
cd c:\Users\asus\OneDrive\Desktop\Guvi_Final_Project

# Start Jupyter
jupyter notebook
```

This opens browser to `http://localhost:8888`

### Step 3.2: Execute Notebook Sections

**Section 1: Dataset Loading & EDA** (10 minutes)
- Cell: Import libraries and load GTSRB dataset
- Output: Class distribution chart, sample images, imbalance analysis
- **Key Insight**: 2.5× imbalance - speed limit signs overfrepresented
- Saved: `outputs/01_class_distribution.png`, `02_sample_images.png`

**Section 2: Data Preprocessing & Augmentation** (15 minutes)
- Cell: Create ImageDataGenerator with augmentation
- Cell: Visualize augmentation effects (rotation, shift, zoom, brightness)
- Explanation: Why horizontal flips are excluded
- Saved: `outputs/03_data_augmentation.png`

**Section 3: Sign Detection & Cropping Pipeline** (15 minutes)
- Cell: Create test road scene images with embedded signs
- Cell: Test detection on 5 full scenes
- Visualization: Original → Bounding box → Cropped (3-column layout)
- Discussion: Limitations of color thresholding vs. YOLO/SSD
- Saved: `outputs/04_detection_pipeline.png`

**Section 4: Custom CNN Architecture Training** (45 minutes)
- Cell: Build 3 configurations (Lightweight, Medium, Heavy)
- Cell: Train all 3 models with EarlyStopping
- Cell: Compare training history (accuracy & loss curves)
- Select best configuration for trade-off analysis
- Saved: `outputs/05_cnn_training_history.png`, `cnn_configurations.csv`
- **Models Saved**: `models/cnn_lightweight.h5`, `cnn_medium.h5`, `cnn_heavy.h5`

**Section 5: MobileNetV2 Transfer Learning** (20 minutes)
- Cell: Load pre-trained MobileNetV2, freeze base, add classification head
- Cell: Train for up to 15 epochs
- Cell: Measure inference time on 100 test images
- **Metric Recorded**: Val accuracy, test accuracy, training time, model size, inference time
- **Model Saved**: `models/mobilenetv2_final.h5`

**Section 6: ResNet50 Transfer Learning** (30 minutes)
- Cell: Load pre-trained ResNet50, freeze base, add classification head
- Cell: Train for up to 15 epochs (may take longer)
- Cell: Measure inference time using same protocol as MobileNetV2
- **Metric Recorded**: Val accuracy, test accuracy, training time, model size, inference time
- **Model Saved**: `models/resnet50_final.h5`

**Section 7: Speed vs. Accuracy Trade-off Analysis** (15 minutes)
- Cell: Compile comparison DataFrame (Model, Accuracy, Inference Time, Size, Training)
- Cell: Create dual-axis chart (bars for accuracy, line for inference time)
- Cell: **ADAS DEPLOYMENT RECOMMENDATION** (data-driven decision)
  - Edge deployment: MobileNetV2 (fast + accurate enough)
  - Cloud batch: ResNet50 (highest accuracy)
- Saved: `outputs/06_speed_vs_accuracy_tradeoff.png`, `speed_vs_accuracy_tradeoff.csv`

**Section 8: Model Evaluation** (20 minutes)
- Cell: Generate predictions on full test set
- Cell: Classification report (Precision, Recall, F1 per class)
- Cell: Create 43×43 confusion matrix heatmap
- Cell: Identify most confused class pairs
- Cell: Display 9 test images with predictions (green=correct, red=incorrect)
- Saved: `outputs/07_confusion_matrix.png`, `08_sample_predictions.png`, `classification_report.txt`

**Section 9: End-to-End Pipeline Demonstration** (10 minutes)
- Cell: Show 2 test cases of complete pipeline
- Visualization: Full scene → Detected bounding box → Cropped sign → Predicted label
- Display: Top-3 predictions with confidence scores
- Saved: `outputs/09_end_to_end_pipeline.png`

**Total Notebook Execution Time**: 2-3 hours (depends on hardware)

### Step 3.3: Save Notebook

```bash
# Notebook auto-saves, but manually save with Ctrl+S
# All outputs saved in outputs/ directory
# All models saved in models/ directory
```

## Phase 4: Deploy Streamlit Application (10 minutes)

### Step 4.1: Launch Streamlit App

```bash
# From project root directory
streamlit run app/app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 4.2: Test Application

1. Open browser to `http://localhost:8501`
2. Upload a road scene image (JPG/PNG)
3. See results:
   - **Stage 1**: Full scene, detected bbox, cropped sign
   - **Stage 2**: Predicted class, confidence, top-3 predictions
   - **Timing**: Detection + classification time in ms

### Step 4.3: Test with Sample Images

Use these classes for testing:
- **Class 0**: Speed limit 20
- **Class 14**: Stop sign
- **Class 17**: No entry
- **Class 25**: Road work
- **Class 32**: Go straight or right

## Phase 5: Deploy to Cloud (Optional)

### Option 1: Streamlit Community Cloud

1. Push code to GitHub
2. Sign in to https://share.streamlit.io
3. Create new app
4. Select repository and `app/app.py`
5. Deploy!
6. Get live URL: `https://[username]-[appname].streamlit.app`

### Option 2: Docker Deployment

```bash
# Build Docker image
docker build -t adas-sign-recognition .

# Run container
docker run -p 8501:8501 adas-sign-recognition

# Access at http://localhost:8501
```

## 📊 Expected Results Summary

### Model Performance

| Model | Val Acc. | Test Acc. | Inference | Size | Recommendation |
|-------|----------|-----------|-----------|------|-----------------|
| Custom CNN | ~90% | ~88% | 25ms | 8MB | Baseline |
| MobileNetV2 | ~93% | ~92% | 35ms | 14MB | **Edge Vehicles** ✓ |
| ResNet50 | ~95% | ~94% | 85ms | 98MB | **Cloud Processing** ✓ |

### Key Metrics

**Detection Pipeline:**
- Success rate: 80-90% (depends on sign color)
- Average detection time: 5-10ms

**Classification Accuracy (Test Set):**
- Macro average: 92-94%
- Weighted average: 93-95%
- Per-class range: 75-99%

**Most Challenging Classes:**
- Speed limit signs (similar appearance)
- Similar-colored warning signs
- Heavily occluded signs

## 🔍 Verification Checklist

- [ ] All dependencies installed without errors
- [ ] GTSRB dataset downloaded and extracted (51,600 images)
- [ ] Notebook runs without errors (2-3 hours)
- [ ] All output PNG files generated (9 files)
- [ ] CSV comparison files created (2 files)
- [ ] All model files saved (6 .h5 files)
- [ ] Streamlit app launches successfully
- [ ] Image upload and processing works
- [ ] Detection visualization displays correctly
- [ ] Classification results show confidence scores
- [ ] Top-3 predictions display correctly

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Verify all packages installed
pip list | grep tensorflow

# Reinstall if needed
pip install --upgrade tensorflow
```

### Issue: GPU not detected

**Solution:**
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# If empty list, use CPU (will be slower)
# Training still works, just slower
```

### Issue: Dataset not found

**Solution:**
```bash
# Verify dataset path
ls data/GTSRB_Dataset/Train/0/

# Should show image files
# If error, re-check extraction path
```

### Issue: Model training very slow

**Solution:**
- This is normal for ResNet50 (~30 minutes on CPU)
- Use GPU if available (100× faster)
- Can stop early with Ctrl+C and use checkpoint

### Issue: Streamlit app slow

**Solution:**
```bash
# Clear cache
streamlit cache clear

# Or limit batch size in app.py
# Or use smaller test set for inference
```

## 📚 Learning Outcomes

After completing this project, you understand:

1. **Two-Stage ADAS Architecture** - Detection + Classification pipeline
2. **Real-World Constraints** - Speed vs. Accuracy trade-offs for deployment
3. **Data Science Workflow** - EDA → Preprocessing → Modeling → Evaluation
4. **Transfer Learning** - Pre-trained models and fine-tuning strategies
5. **Model Comparison** - Quantitative framework for model selection
6. **Production ML** - Deployment considerations, containerization, web apps
7. **Evaluation Metrics** - Beyond accuracy (confusion matrix, per-class metrics)
8. **Business Thinking** - How recommendation changes with deployment context

## 🎓 Further Improvements

Try these enhancements:

1. **Better Detection**: Replace color thresholding with YOLOv5
2. **Ensemble**: Vote between MobileNetV2 and ResNet50
3. **Attention**: Add channel/spatial attention mechanisms
4. **Mobile App**: Deploy on iOS/Android using TensorFlow Lite
5. **Video Input**: Process dashcam video frame-by-frame
6. **Real-time Alerts**: Warn driver of detected signs
7. **Data Collection**: Gather your own road scene images
8. **Fine-tuning**: Use domain-specific dashcam training data

## 📝 Project Deliverables Checklist

- [x] Jupyter Notebook with complete pipeline
- [x] Exploratory Data Analysis (9 visualizations)
- [x] Custom CNN (3 configurations compared)
- [x] MobileNetV2 transfer learning
- [x] ResNet50 transfer learning
- [x] Speed vs. Accuracy trade-off analysis
- [x] Model Evaluation (confusion matrix, classification report)
- [x] End-to-end pipeline demonstration
- [x] Streamlit deployment app
- [x] Docker containerization
- [x] README documentation
- [x] Requirements.txt
- [x] Utility modules (detection, models)

## ✅ Success Criteria

Your project is complete when:
1. Notebook runs without errors
2. All models trained and compared
3. Trade-off analysis documented with numbers
4. Streamlit app deployed and functional
5. Clear deployment recommendation provided
6. README explains all components
7. Code is documented and reproducible

---

**Estimated Total Time**: 4-5 hours (mostly training time)

**Hardware Requirements**:
- CPU: 4+ cores
- RAM: 8GB+ (16GB recommended)
- GPU: Optional but 10× faster
- Storage: 20GB+ for dataset + models

**Ready to build? Start with Phase 1!**
