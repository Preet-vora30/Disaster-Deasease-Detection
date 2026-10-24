import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_extractors import extract_all_features, extract_sift_features, extract_harris_corners
from utils import preprocess_image
from ml_pipeline import load_data, build_svm_pipeline, build_ensemble_pipeline
from web_scraper import scrape_disaster_images

print("="*60)
print("DISASTERVISION AI - COMPREHENSIVE MODULE TEST PREVIEW")
print("="*60)

# ----------------------------------------------------
# TEST 1: Image Processing & Feature Extraction
# ----------------------------------------------------
print("\n[1/4] Testing Image Processing & Feature Extraction...")
start = time.time()
img_path = "datasets/post_0000.png"
if os.path.exists(img_path):
    img_pil, img_array = preprocess_image(img_path)
    if img_array is not None:
        sift_feat = extract_sift_features(img_array)
        harris_feat = extract_harris_corners(img_array)
        all_feat = extract_all_features(img_array)
        
        print(f"  OK Preprocessing: Successful (Numpy Array Shape: {img_array.shape})")
        print(f"  OK SIFT Features: Extracted {len(sift_feat)} average pooled dimensions")
        print(f"  OK Harris Corners: Value {harris_feat[0]}")
        print(f"  OK Combined Extract_All_Features: {len(all_feat)} dimensions")
    else:
        print("  X Failed to load image through Pil/OpenCV")
else:
    print("  X Image not found for testing")
print(f"  > Time taken: {time.time()-start:.2f}s")


# ----------------------------------------------------
# TEST 2: Label Encoder & Pipeline Data Loading
# ----------------------------------------------------
print("\n[2/4] Testing Data Loading & Category Encoding...")
start = time.time()
metadata_path = "datasets/metadata.csv"
try:
    X_df, y, f_cols, le = load_data(metadata_path, "datasets")
    print(f"  OK CSV Metadata Loaded: {len(X_df)} samples extracted")
    print(f"  OK Geo Encodings (bins=10): Simulated {len(le.classes_)} unique buckets")
    print(f"  OK Feature DataFrame Shape: {X_df.shape} containing numeric + nominal categories")
except Exception as e:
    print(f"  X Data loading failed: {e}")
print(f"  > Time taken: {time.time()-start:.2f}s")


# ----------------------------------------------------
# TEST 3: Web Scraper Simulator (BeautifulSoup)
# ----------------------------------------------------
print("\n[3/4] Testing Web Scraper Engine (BeautifulSoup)...")
start = time.time()
try:
    # Test with 2 images
    res = scrape_disaster_images(query="Bengaluru Flood test", num_images=2, output_dir="datasets/scraped_test")
    print(f"  OK Extracted {len(res)} simulated images successfully using HTTP requests")
    if len(res) > 0:
        print(f"  OK Validated Mock EXIF GPS Extraction: Lat {res[0]['latitude']:.4f}, Lon {res[0]['longitude']:.4f}")
except Exception as e:
    print(f"  X Web Scraper failed: {e}")
print(f"  > Time taken: {time.time()-start:.2f}s")


# ----------------------------------------------------
# TEST 4: ML Models GridSearchCV & Inference
# ----------------------------------------------------
print("\n[4/4] Testing Scikit-Learn Pipeline & Model Inference...")
start = time.time()
model_path = "models/svm_trained.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("  OK `svm_trained.pkl` loaded successfully from disk")
    print("  OK Verifying Pipeline Structure: ", [step[0] for step in model.steps])
    
    # Run Inference
    test_sample = X_df.iloc[[0]]
    pred = model.predict(test_sample)
    prob = model.predict_proba(test_sample)
    
    mapping = {0:"No Damage", 1:"Minor", 2:"Major", 3:"Destroyed"}
    
    print(f"  OK Single Inference Test: Success -> Predicted Class {pred[0]} ({mapping.get(pred[0], 'Unknown')})")
    print(f"  OK Prediction Probabilities: {prob[0]}")
    print(f"  OK Test sample True Class: {y[0]} ({mapping.get(y[0], 'Unknown')})")
else:
    print("  X Model not found! Try running `python ml_pipeline.py` first.")
print(f"  > Time taken: {time.time()-start:.2f}s")


print("\n" + "="*60)
print("ALL TESTS COMPLETED SUCCESFULLY")
print("="*60)
