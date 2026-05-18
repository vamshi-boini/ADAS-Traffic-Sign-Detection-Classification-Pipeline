# ADAS Traffic Sign Detection & Classification Pipeline

## 🚗 Project Overview

A **complete two-stage computer vision pipeline** for real-world autonomous vehicle traffic sign recognition:
- **Stage 1**: Detects and crops traffic signs from full road scene images
- **Stage 2**: Classifies detected signs into 43 GTSRB categories

This project replicates how production ADAS systems (Tesla, Waymo, Mobileye) actually work - not just classifying pre-cropped images, but detecting signs in raw camera frames.

## 🎯 Business Relevance

### Use Cases
1. **Autonomous Vehicles** - Speed limit enforcement, warning detection, no-entry alerts
2. **Smart City Infrastructure** - Automated road sign inventory from dashcam footage
3. **Navigation/Mapping** - Real-world scene processing for Google Maps/HERE/TomTom
4. **Fleet Management** - Insurance claims analysis from dashcam footag

## 📊 Key Features

### 1. Complete ADAS Pipeline
- ✓ Full road scene image → Sign detection → Classification
- ✓ End-to-end demonstration on real test cases
- ✓ Production-ready two-stage architecture

### 2. Multiple Model Architectures
- ✓ Custom CNN (3 configurations with ablation study)
- ✓ MobileNetV2 (lightweight, edge deployment)
- ✓ ResNet50 (high-accuracy, cloud deployment)

### 3. Speed vs. Accuracy Trade-off Analysis
- ✓ Measured inference time per image
- ✓ Model size comparison
- ✓ Data-driven deployment recommendation

### 4. Comprehensive Evaluation
- ✓ Classification report (Precision, Recall, F1-Score per class)
- ✓ 43×43 Confusion Matrix heatmap
- ✓ Class imbalance analysis
- ✓ Most confused sign pairs identification

### 5. Streamlit Deployment App
- ✓ Interactive web interface
- ✓ Real-time image upload and processing
- ✓ Visualization of detection + classification results
- ✓ Confidence scores and top-3 predictions

## 📁 Project Structure

```
Guvi_Final_Project/
├── ADAS_Traffic_Sign_Pipeline.ipynb      # Main notebook (all analysis)
├── app/
│   └── app.py                             # Streamlit deployment app
├── utils/
│   ├── detection.py                      # Detection & cropping functions
│   └── models.py                         # Model building utilities
├── data/
│   └── GTSRB_Dataset/                    # Dataset directory (to be populated)
│       ├── Train/
│       └── Test/
├── models/
│   ├── cnn_lightweight.h5                # Saved custom CNN models
│   ├── cnn_medium.h5
│   ├── cnn_heavy.h5
│   ├── mobilenetv2_final.h5              # MobileNetV2
│   └── resnet50_final.h5                 # ResNet50
├── outputs/
│   ├── 01_class_distribution.png
│   ├── 02_sample_images.png
│   ├── 03_data_augmentation.png
│   ├── 04_detection_pipeline.png
│   ├── 05_cnn_training_history.png
│   ├── 06_speed_vs_accuracy_tradeoff.png
│   ├── 07_confusion_matrix.png
│   ├── 08_sample_predictions.png
│   ├── 09_end_to_end_pipeline.png
│   ├── cnn_configurations.csv
│   ├── speed_vs_accuracy_tradeoff.csv
│   └── classification_report.txt
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Clone/navigate to project directory
cd Guvi_Final_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

1. Download GTSRB dataset from http://benchmark.ini.rub.de/gtsrb_dataset.html
2. Extract to `data/GTSRB_Dataset/` with structure:
   ```
   data/GTSRB_Dataset/
   ├── Train/  (43 subdirectories: 0-42)
   └── Test/   (43 subdirectories: 0-42)
   ```

### 3. Run Main Notebook

```bash
jupyter notebook ADAS_Traffic_Sign_Pipeline.ipynb
```

Execute all cells to:
- Perform EDA on all 43 sign classes
- Train 3 custom CNN configurations
- Fine-tune MobileNetV2 and ResNet50
- Generate Speed vs. Accuracy comparison
- Evaluate on test set
- Demonstrate end-to-end pipeline

### 4. Launch Streamlit App

```bash
streamlit run app/app.py
```

Open browser to `http://localhost:8501`

## 📊 Key Results Summary

### Model Comparison Table

| Model | Val Accuracy | Test Accuracy | Inference (ms) | Size (MB) | Use Case |
|-------|-------------|---------------|----------------|-----------|----------|
| Custom CNN (Medium) | ~90% | ~88% | 25ms | ~8 | Baseline |
| MobileNetV2 | ~93% | ~92% | 35ms | 14 | **Edge deployment** |
| ResNet50 | ~95% | ~94% | 85ms | 98 | **Cloud processing** |

### ADAS Deployment Recommendation

**For Onboard Vehicle Edge Devices** (NVIDIA Jetson, Qualcomm Snapdragon):
- **Recommended:** MobileNetV2
- **Rationale:**
  - ✓ Inference: 35ms < 100ms latency budget (30 FPS camera)
  - ✓ Accuracy: 92% meets safety requirements
  - ✓ Size: 14MB fits in onboard memory
  - ✓ Optimized for edge hardware

**For Cloud Batch Processing** (Insurance, Fleet Management):
- **Recommended:** ResNet50
- **Rationale:**
  - ✓ Higher accuracy: 94% reduces false positives
  - ✓ No real-time latency constraint
  - ✓ Minimize risk of false accusations in claim analysis

## 🔍 Sample Outputs

### 1. Class Distribution Analysis
- Identifies severe class imbalance (2.5:1 ratio)
- Speed limit signs are over-represented
- Justifies use of class weights in training

### 2. Data Augmentation Visualization
- Shows rotation, shift, zoom, brightness effects
- Justified by real-world driving conditions
- Explains why horizontal flips are excluded

### 3. Sign Detection Pipeline Test
- 5 test road scenes with embedded signs
- Shows original, detected bounding box, cropped output
- Demonstrates detection limitations and trade-offs

### 4. Speed vs. Accuracy Trade-off Chart
- Dual-axis visualization (accuracy bars + speed line)
- Makes deployment decision visually clear
- Supports quantitative recommendation

### 5. Confusion Matrix Heatmap
- 43×43 matrix showing which signs are confused
- Identifies most challenging class pairs
- Guides future model improvements

### 6. End-to-End Pipeline Demonstration
- Full scene image → Detected sign → Predicted label
- Shows complete two-stage flow
- Demonstrates production-ready system

## 🛠️ Technical Implementation

### Detection Module (`utils/detection.py`)

```python
cropped_sign, bbox = detect_and_crop_sign(image, target_size=(64, 64))
```

**Algorithm:**
1. Convert BGR to HSV (lighting-robust)
2. Create masks for red/yellow hues
3. Morphological operations (close, open)
4. Contour detection and filtering (area, aspect ratio)
5. Extract bounding box and crop
6. Resize to 64×64 for classifier

### Model Architectures (`utils/models.py`)

**Custom CNN:**
```
Conv2D(32, 3×3) → MaxPool(2×2) → Conv2D(64, 3×3) → MaxPool(2×2)
→ Flatten() → Dense(neurons, ReLU) → Dropout(0.3) → Dense(43, softmax)
```

**Transfer Learning:**
- MobileNetV2 base (pre-trained ImageNet)
- GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(43)
- Frozen base layers, fine-tune classification head only

## 📈 Performance Metrics

### Evaluation on Test Set
- **Accuracy**: Per-class precision calculated
- **Precision**: True positives vs. all predictions
- **Recall**: True positives vs. all ground truth
- **F1-Score**: Harmonic mean of precision & recall
- **Confusion Matrix**: 43×43 misclassification patterns

### Most Confused Sign Pairs
Examples from confusion matrix:
- Speed limit signs (similar appearance)
- Warning triangles (yellow color similarity)
- Prohibition signs (red round shape)

## 🌐 Streamlit Deployment

### Features
- ✓ Image upload interface
- ✓ Real-time detection visualization
- ✓ Classification with confidence scores
- ✓ Top-3 prediction display
- ✓ Processing time metrics
- ✓ Production-ready UI

### Deploy to Cloud

**Streamlit Community Cloud:**
```bash
# Create GitHub repository
# Connect to Streamlit Community Cloud
# Select app/app.py as main file
# Deploy!
```

**Docker Deployment:**
```bash
docker build -t adas-sign-recognition .
docker run -p 8501:8501 adas-sign-recognition
```

## 📚 Dataset Information

**GTSRB (German Traffic Sign Recognition Benchmark)**
- 43 traffic sign categories
- 50,000+ training images
- ~12,600 test images
- Real-world traffic scenes
- Class imbalance: 2.5:1 ratio

**Sign Categories:**
- Speed limits (20-130 km/h)
- Prohibition signs (no entry, no stopping, etc.)
- Danger warnings (pedestrian, curve, etc.)
- Information/directional signs (blue)
- Other mandatory/regulatory signs

## 🔧 Training Configuration

### Data Preprocessing
- Input size: 64×64 pixels (balance: detail vs. speed)
- Normalization: [0, 1] pixel values
- Train/Val split: 80/20 with fixed seed
- Class mode: Categorical (one-hot encoding)

### Data Augmentation (Training only)
- Rotation: ±15° (vehicle angle)
- Shift: ±10% (position in frame)
- Zoom: ±10% (distance variation)
- Brightness: 90-120% (lighting conditions)
- NO horizontal flip (left ≠ right directional signs!)

### Training Strategy
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 30 (with EarlyStopping, patience=3)
- Batch size: 32
- Callbacks: EarlyStopping, ModelCheckpoint

## ⚠️ Known Limitations & Future Improvements

### Current Limitations
1. **Detection**: Colour thresholding works well for red/yellow, struggles with blue
2. **Speed Limit Signs**: Similar appearance makes them hard to distinguish
3. **Edge Cases**: Heavily occluded or damaged signs may fail
4. **Weather**: Reflection/wet signs not optimized

### Future Improvements
1. Replace color thresholding with YOLO/SSD object detector
2. Use ensemble voting (combine MobileNetV2 + ResNet50)
3. Add attention mechanism to focus on key sign features
4. Fine-tune on harder examples (difficult-to-classify pairs)
5. Mobile app deployment (Android/iOS)
6. Real-time video processing support

## 📖 How to Use This Project

### For Learning
- Study the notebook sections in order
- Understand each stage (EDA → Preprocessing → Modeling → Evaluation)
- Review speed vs. accuracy trade-off framework
- Adapt for your own datasets

### For Production
- Load trained models from `models/` directory
- Use detection + classification functions in your pipeline
- Adapt detection thresholds for your camera/lighting
- Deploy via Streamlit or Docker

### For Research
- Compare different architectures
- Test on larger/different datasets
- Experiment with detection algorithms
- Publish results and improvements

## 💡 Key Insights

1. **Real ADAS are Two-Stage**: Detection + Classification always
2. **Speed vs. Accuracy**: Trade-off depends on deployment context
3. **Class Imbalance**: Affects all thresholds and decisions
4. **Augmentation Matters**: Simulates real driving conditions
5. **End-to-End Testing**: Critical for production readiness

## 📝 Citation

If you use this project in research, please cite:

```
ADAS Traffic Sign Detection & Classification Pipeline
A Complete Two-Stage Real-World Computer Vision System
Dataset: GTSRB (http://benchmark.ini.rub.de/gtsrb_dataset.html)
```

## 📞 Support & Questions

- Check the notebook for detailed explanations
- Review code comments in `utils/` modules
- Test with sample images in `outputs/`
- Experiment with hyperparameters and report results

---

**Built with TensorFlow, Keras, OpenCV, and Streamlit**

**Project demonstrates: ADAS development, transfer learning, speed-accuracy trade-offs, and production ML deployment**
