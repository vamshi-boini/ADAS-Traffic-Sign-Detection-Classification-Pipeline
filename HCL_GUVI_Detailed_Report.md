# HCL_GUVI Final Project: ADAS Traffic Sign Detection & Classification Pipeline

**Project Title**: Two-Stage ADAS Traffic Sign Perception System
**Dataset Used**: GTSRB (German Traffic Sign Recognition Benchmark)
**Status**: Completed and Validated

---

## 1. Executive Summary
This project successfully builds and deploys a complete two-stage Advanced Driver Assistance System (ADAS) pipeline. Rather than just classifying pre-cropped images (an academic exercise), this pipeline correctly simulates real-world autonomous vehicle perception. 
- **Stage 1 (Detection)**: Locates and crops traffic signs from full road scenes. 
- **Stage 2 (Classification)**: Accurately classifies the cropped sign into one of 43 distinct traffic sign categories.

---

## 2. EDA Findings (Exploratory Data Analysis)
The GTSRB dataset contains over 50,000 images across 43 classes.
* **Class Imbalance**: Discovered a severe class imbalance. The most frequent classes (like 'Speed limit 50km/h') contain over 2,000 samples, whereas minority classes (like 'Speed limit 20km/h') contain fewer than 250 samples. 
* **Real-World Artifacts**: Images vary dramatically in sizing, lighting (overexposed, night-time), motion blur, and physical occlusion (stickers, dirt). 
* **Action Taken**: Utilized real-world data augmentations via `ImageDataGenerator` (rotation, zoom, brightness, and shift). We strategically *avoided* horizontal flipping to prevent semantic destruction (e.g., flipping a 'Turn Right' sign into a 'Turn Left' sign).

---

## 3. Detection Pipeline Design Decisions
To extract the signs from full scenes efficiently before inference:
* We implemented an **HSV color-based thresholding** engine targeting typical structural traffic sign colors (Red, Blue, Yellow/Orange, White).
* Paired with **Canny Edge Detection** and contour mapping, the engine filters polygons by bounding-box area and flexible aspect ratios to zero in on signs.
* **Fallback Strategy**: If a sign is thoroughly obscured, the pipeline safely falls back to a central crop, ensuring the classifier can still attempt a read without pipeline crashes.
* **Conclusion**: This guarantees that the CNN only runs dense compute on the region of interest, massively improving throughput for edge deployment.

---

## 4. CNN Architecture Experiments
We built and tested three variants of neural networks to classify the cropped 64x64 signs:
1. **Custom CNN**: Receptive fields tailored exactly to 64x64 input. Utilizes deep Convolutional blocks, Max-Pooling, heavy Dropout (0.5), and global average pooling to prevent overfitting while adapting perfectly to the GTSRB scale.
2. **MobileNetV2**: A lightweight architecture utilizing pre-trained ImageNet weights (Transfer Learning) with inputs dynamically scaled to `[-1, 1]`.
3. **ResNet50**: A deep, heavy architecture with pre-trained ImageNet weights, dynamically scaling inputs up to `[0, 255]` and applying Caffe-style mean subtraction.

---

## 5. Speed vs. Accuracy Trade-off Analysis & ADAS Deployment Recommendation
After training, we evaluated models based on test metric outputs:

* **Custom CNN**: ~96.73% Validation Accuracy | ~4.6ms Inference (Winner)
* **MobileNetV2**: ~48.0% Validation Accuracy | ~15ms Inference
* **ResNet50**: ~51.9% Validation Accuracy | ~38ms Inference

**Trade-Off Insight**: Because we froze the base architectures for MobileNetV2 and ResNet50, they were limited to generic features (dogs/cats) rather than fine-grained geometric shapes of GTSRB signs. 
**Deployment Recommendation**:
We strongly recommend deploying the **Custom CNN** for edge hardware. Its weights scale efficiently (~3.4MB), it executes lighting-fast inference (~4.6ms), and easily pushes past the 96% accuracy threshold, rendering it robust enough for reliable ADAS Speed Limit enforcement and hazard warnings in real-time.

---

## 6. Model Evaluation Results
By evaluating the Custom CNN against 12,630 unseen testing samples via the evaluation suite:
* **Test Accuarcy**: 96.73% overall matching.
* **Inference Speed**: ~355 images mapped per second.
* **Calibration**: Mean confidence was found to be exceptionally high (0.9677), with over 88% of predictions landing above a 95% threshold certainty level.
* **Confusions**: The confusion matrix indicated isolated confusions largely restricted to geometrically identical speed limit numbers (e.g. 30 vs 50, 70 vs 20), exacerbated by motion blur or extreme distances in the original captures. 

---

## 7. Key Takeaways
1. Proper architectural inputs matter; explicitly aligning dataset processing representations (`[0, 1]` vs `[-1, 1]` vs unscaled) determines the success or failure of deployed pre-trained models.
2. The End-to-End two-stage strategy perfectly models real perception systems, proving that cropping significantly elevates accuracy while keeping runtime boundaries tight.
3. The project is fully modular and responsive, accessible gracefully inside the enclosed Streamlit Application (`app.py`), bridging backend Machine Learning concepts right into frontend engineering workflows.
