import cv2
import numpy as np
from PIL import Image, ImageFilter
import pytesseract

def preprocess_image(image_path, size=(256, 256)):
    """
    Load image, apply Grayscale, Gaussian blur, and resize.
    Returns both PIL image and numpy array (for cv2).
    """
    try:
        img = Image.open(image_path).convert('L')
        img = img.filter(ImageFilter.GaussianBlur(1))
        img = img.resize(size)
        return img, np.array(img)
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None, None

def extract_text_ocr(image_path):
    """
    Extract text using Tesseract OCR.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return ""

def evaluate_model(y_true, y_pred):
    """
    Returns accuracy and F1 score.
    """
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return acc, f1
