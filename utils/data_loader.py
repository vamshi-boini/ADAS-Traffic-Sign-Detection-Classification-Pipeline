"""
Complete data loading module for ADAS GTSRB traffic sign project.
Handles both training (class subdirectories) and test (flat) data structures.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class GTSRBDataLoader:
    """Complete data loader for GTSRB dataset with proper handling of file structures."""
    
    def __init__(self, data_path='data/GTSRB_Dataset', target_size=(64, 64)):
        """
        Initialize data loader.
        
        Args:
            data_path: Root path to GTSRB dataset
            target_size: Target image size (H, W)
        """
        self.data_path = os.path.abspath(data_path)
        self.target_size = target_size
        self.train_path = os.path.join(self.data_path, 'Train')
        self.test_path = os.path.join(self.data_path, 'Test')
        self.meta_file = os.path.join(self.data_path, 'Meta.csv')
        
    def load_class_names(self):
        """Load class names from metadata or create default mapping."""
        try:
            meta_df = pd.read_csv(self.meta_file)
            if 'SignName' in meta_df.columns:
                return dict(zip(meta_df['ClassId'], meta_df['SignName']))
            else:
                # Create default class names based on ClassId
                class_names_default = {
                    0: 'Speed limit 20',
                    1: 'Speed limit 30',
                    2: 'Speed limit 50',
                    3: 'Speed limit 60',
                    4: 'Speed limit 70',
                    5: 'Speed limit 80',
                    6: 'End of speed limit 80',
                    7: 'Speed limit 100',
                    8: 'Speed limit 120',
                    9: 'No passing',
                    10: 'No passing trucks',
                    11: 'Right-of-way',
                    12: 'Priority road',
                    13: 'Yield',
                    14: 'Stop',
                    15: 'No entry',
                    16: 'Prohibited all',
                    17: 'Prohibited trucks',
                    18: 'Mandatory direction straight',
                    19: 'Mandatory turn right',
                    20: 'Mandatory turn left',
                    21: 'Mandatory straight or right',
                    22: 'Mandatory straight or left',
                    23: 'Mandatory right',
                    24: 'Mandatory left',
                    25: 'Mandatory go round right',
                    26: 'Mandatory go round left',
                    27: 'End of prohibited zone',
                    28: 'End of no passing',
                    29: 'End of no passing trucks',
                    30: 'Speed limit zone start',
                    31: 'Speed limit zone end',
                    32: 'Pedestrian crossing',
                    33: 'Railway level crossing',
                    34: 'Cyclists crossing',
                    35: 'Ice/snow hazard',
                    36: 'Wild animals crossing',
                    37: 'End of all restrictions',
                    38: 'Turn right ahead',
                    39: 'Turn left ahead',
                    40: 'Ahead only',
                    41: 'Go straight or right',
                    42: 'Go straight or left'
                }
                return class_names_default
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
            # Return default mapping
            return {i: f'Class {i}' for i in range(43)}
    
    def get_train_validation_generators(self, batch_size=32, validation_split=0.2):
        """
        Get training and validation data generators using ImageDataGenerator.
        Uses class subdirectories: Train/0, Train/1, etc.
        
        Args:
            batch_size: Batch size for generators
            validation_split: Fraction of training data to use for validation
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        print(f"Loading training and validation data from: {self.train_path}")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42
        )
        
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {validation_generator.samples}")
        print(f"✓ Number of classes: {len(train_generator.class_indices)}")
        
        return train_generator, validation_generator
    
    def load_test_data_flat(self, batch_size=32):
        """
        Load test data from flat directory structure.
        Test images are stored flat, not in class subdirectories.
        
        Args:
            batch_size: Batch size for batching
        
        Returns:
            Tuple of (test_generator, test_image_files)
        """
        print(f"Loading test data from flat directory: {self.test_path}")
        
        # Get list of test images
        test_images = sorted([
            f for f in os.listdir(self.test_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))
        ])
        
        print(f"✓ Found {len(test_images)} test images")
        
        # Generator function for flat directory
        def test_image_generator():
            for idx in range(0, len(test_images), batch_size):
                batch_files = test_images[idx:idx+batch_size]
                batch_images = []
                batch_filenames = []
                
                for img_file in batch_files:
                    try:
                        img_path = os.path.join(self.test_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.target_size)
                        img_array = np.array(img) / 255.0
                        batch_images.append(img_array)
                        batch_filenames.append(img_file)
                    except Exception as e:
                        print(f"Warning: Error loading {img_file}: {e}")
                        continue
                
                if batch_images:
                    yield np.array(batch_images), np.array(batch_filenames)
        
        return test_image_generator(), test_images
    
    def validate_structure(self):
        """Validate dataset directory structure."""
        print("\nValidating dataset structure...")
        
        train_exists = os.path.exists(self.train_path)
        test_exists = os.path.exists(self.test_path)
        meta_exists = os.path.exists(self.meta_file)
        
        print(f"  Train path exists: {train_exists}")
        print(f"  Test path exists: {test_exists}")
        print(f"  Meta file exists: {meta_exists}")
        
        if train_exists:
            train_classes = len([d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))])
            print(f"  Train classes: {train_classes}")
        
        if test_exists:
            test_images = len([f for f in os.listdir(self.test_path) if f.lower().endswith(('.png', '.jpg', '.ppm'))])
            print(f"  Test images (flat): {test_images}")
        
        return train_exists and test_exists and meta_exists


# Standalone test function for quick validation
def quick_test():
    """Quick validation of data loading."""
    print("="*60)
    print("GTSRB Data Loader - Quick Test")
    print("="*60)
    
    loader = GTSRBDataLoader()
    
    # Validate structure
    valid = loader.validate_structure()
    if not valid:
        print("✗ Dataset structure validation failed!")
        return False
    
    print("\n" + "="*60)
    print("Testing Training/Validation Data Loading")
    print("="*60)
    
    # Test train/val generators
    try:
        train_gen, val_gen = loader.get_train_validation_generators(batch_size=32)
        print("✓ Training/validation generators loaded successfully!")
        
        # Test first batch
        x_batch, y_batch = next(train_gen)
        print(f"  Train batch shape: {x_batch.shape}")
        print(f"  Train labels shape: {y_batch.shape}")
        print(f"  Image pixel range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return False
    
    print("\n" + "="*60)
    print("Testing Test Data Loading")
    print("="*60)
    
    # Test test data generator
    try:
        test_gen, test_files = loader.load_test_data_flat(batch_size=32)
        print("✓ Test data generator created successfully!")
        print(f"  Total test images: {len(test_files)}")
        
        # Test first batch
        x_test, files = next(test_gen)
        print(f"  Test batch shape: {x_test.shape}")
        print(f"  Test batch filenames: {files[:3]}")
        print(f"  Image pixel range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All data loading tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
