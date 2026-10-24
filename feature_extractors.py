import cv2
import numpy as np

def extract_sift_features(image_array, max_keypoints=50):
    """
    Extracts SIFT features (128-dim descriptor).
    Returns an average pooled 128-dim vector.
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kp, des = sift.detectAndCompute(image_array, None)
    
    if des is None or len(des) == 0:
        return np.zeros(128)
    
    # Average pooling to get a fixed-size representation
    return np.mean(des, axis=0)

def extract_harris_corners(image_array):
    """
    Extracts Harris Corners.
    Returns the count of strong corners.
    """
    # Ensure it's float32
    if image_array.dtype != np.float32:
        gray = np.float32(image_array)
    else:
        gray = image_array
        
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value
    corners_count = np.sum(dst > 0.01 * dst.max())
    return np.array([corners_count])

def extract_all_features(image_array):
    """
    Combines SIFT and Harris features.
    Returns a 129-dim feature vector (128 SIFT + 1 Harris).
    """
    sift_desc = extract_sift_features(image_array)
    harris_features = extract_harris_corners(image_array)
    
    combined = np.concatenate([sift_desc, harris_features])
    return combined
