"""
Sign Detection & Cropping Module
This module handles the first stage of the ADAS pipeline: detecting and cropping
traffic signs from full road scene images using colour-based thresholding.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def detect_and_crop_sign(
    image: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    verbose: bool = False,
    fallback_to_center: bool = True
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect and crop a traffic sign from a full road scene image.
    
    Uses multiple robust detection strategies:
    1. HSV color thresholding (red, yellow, blue, white)
    2. Adaptive edge detection with Canny
    3. Contour analysis with flexible filtering
    4. Fallback to 64x64 center crop when no sign is detected
    
    Args:
        image: Input image (BGR format from OpenCV or RGB)
        target_size: Output size for cropped sign (default: 64x64)
        verbose: Print debug information
        
    Returns:
        Tuple of (cropped_sign, bounding_box) or (None, None) if detection fails
    """
    
    # Approach 1: Multi-color HSV detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Very permissive color ranges
    masks = []
    
    # Red (very permissive - low saturation threshold)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    masks.append(cv2.inRange(hsv, lower_red1, upper_red1))
    
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    masks.append(cv2.inRange(hsv, lower_red2, upper_red2))
    
    # Yellow/Orange
    lower_yellow = np.array([5, 30, 30])
    upper_yellow = np.array([50, 255, 255])
    masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    
    # Blue
    lower_blue = np.array([80, 30, 30])
    upper_blue = np.array([140, 255, 255])
    masks.append(cv2.inRange(hsv, lower_blue, upper_blue))
    
    # White (low saturation, high value)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    masks.append(cv2.inRange(hsv, lower_white, upper_white))
    
    # Combine all color masks
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Approach 2: Edge detection for boundary detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Canny edge detection with low thresholds
    edges = cv2.Canny(enhanced, 30, 100)
    
    # Combine color mask and edges
    combined_mask = cv2.bitwise_or(combined_mask, edges)
    
    # Aggressive morphological operations
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel1, iterations=3)
    # Remove noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    # Dilate to fill holes
    combined_mask = cv2.dilate(combined_mask, kernel1, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        if verbose:
            print("No contours found")
        if fallback_to_center:
            # fallback crop to center area strongly when no detection
            h, w = image.shape[:2]
            min_side = min(h, w)
            center_x, center_y = w // 2, h // 2
            crop_size = min_side // 2
            x = max(0, center_x - crop_size // 2)
            y = max(0, center_y - crop_size // 2)
            x2 = min(w, x + crop_size)
            y2 = min(h, y + crop_size)
            cropped = image[y:y2, x:x2]
            if cropped.size == 0:
                return None, None
            resized = cv2.resize(cropped, target_size)
            return resized, (x, y, x2 - x, y2 - y)
        return None, None
    
    # Very lenient contour filtering
    valid_contours = []
    img_area = image.shape[0] * image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Accept any contour with reasonable area
        if area < 50 or area > img_area * 0.7:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum size check
        if w < 8 or h < 8:
            continue
        
        # Very flexible aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if 0.2 < aspect_ratio < 5:  # Much more permissive
            valid_contours.append((area, x, y, w, h))
    
    if not valid_contours:
        if verbose:
            print("No valid contours after filtering")
        return None, None
    
    # Pick largest contour
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    _, x, y, w, h = valid_contours[0]
    
    # Add generous padding
    padding = int(0.2 * max(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop
    cropped = image[y:y+h, x:x+w]
    
    if cropped.size == 0:
        if verbose:
            print("Cropped region is empty")
        return None, None
    
    # Resize
    resized = cv2.resize(cropped, target_size)
    
    if verbose:
        print(f"Sign detected at: x={x}, y={y}, w={w}, h={h}")
    
    return resized, (x, y, w, h)


def draw_bounding_box(
    image: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image: Input image (BGR format)
        bounding_box: (x, y, w, h) bounding box coordinates
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with drawn bounding box
    """
    image_copy = image.copy()
    x, y, w, h = bounding_box
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy
