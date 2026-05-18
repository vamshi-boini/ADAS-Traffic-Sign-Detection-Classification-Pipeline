"""
Model Building Utilities for CNN Architecture
This module provides functions to build and compile custom CNN configurations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2, ResNet50
from typing import Tuple


def build_custom_cnn(
    num_hidden_layers: int,
    neurons: int,
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001
) -> Model:
    """
    Build a custom CNN from scratch using Keras Sequential API.
    
    Architecture:
    - Shared backbone: Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> 
      MaxPool(2x2) -> Flatten
    - Configurable dense layers with dropout and L2 regularization
    - Output: Dense(43, softmax) for 43 sign classes
    
    Args:
        num_hidden_layers: Number of hidden dense layers
        neurons: Number of neurons in each hidden layer
        input_shape: Input image shape
        num_classes: Number of output classes (43 for GTSRB)
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization weight for dense layers
        
    Returns:
        Compiled Keras Model
    """
    
    model = Sequential([
        # Shared backbone
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
    ])
    
    # Add configurable hidden layers
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    dropout_rate: float = 0.3
) -> Model:
    """
    Build MobileNetV2 transfer learning model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras Model
    """
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Build classification head with [0,1] to [-1,1] preprocessing mapping
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(scale=2.0, offset=-1.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_resnet50(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    dropout_rate: float = 0.3
) -> Model:
    """
    Build ResNet50 transfer learning model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras Model
    """
    
    # Load pre-trained ResNet50
    base_model = ResNet50(
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Build classification head with [0,1] to [0,255] and Caffe BGR/Mean preprocessing
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(scale=255.0),
        layers.Lambda(tf.keras.applications.resnet50.preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
