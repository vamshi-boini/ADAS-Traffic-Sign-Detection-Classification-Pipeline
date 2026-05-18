# Models Directory

## 📦 Trained Models Storage

This directory will contain the trained Keras models after running the notebook.

### Expected Files:
- `cnn_lightweight.h5` - Custom CNN (2 hidden layers, 128 neurons)
- `cnn_medium.h5` - Custom CNN (3 hidden layers, 256 neurons)
- `cnn_heavy.h5` - Custom CNN (4 hidden layers, 512 neurons)
- `mobilenetv2_final.h5` - MobileNetV2 transfer learning model
- `resnet50_final.h5` - ResNet50 transfer learning model

## 🔄 How These Are Created:

Run the Jupyter notebook:
```bash
jupyter notebook ADAS_Traffic_Sign_Pipeline.ipynb
```

The notebook will:
1. Load training data from `data/GTSRB_Dataset/Train/`
2. Build models using utilities in `utils/models.py`
3. Train each model on the dataset
4. Save trained models here as `.h5` files

## ⚡ File Sizes (Approximate):
- Custom CNN: 5-15 MB
- MobileNetV2: 14 MB
- ResNet50: 98 MB

**Total: ~130-150 MB after training**

## 📊 Model Performance (Expected):
| Model | Accuracy | Inference Time |
|-------|----------|-----------------|
| Custom CNN (Light) | ~86% | 10ms |
| Custom CNN (Medium) | ~90% | 25ms |
| Custom CNN (Heavy) | ~91% | 40ms |
| MobileNetV2 | ~92% | 35ms |
| ResNet50 | ~94% | 85ms |

