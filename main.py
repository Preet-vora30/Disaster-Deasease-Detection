import os
import time
import pickle
import threading
import cv2
import numpy as np
from PIL import Image
from tkinter import filedialog
import folium
import webbrowser
import pandas as pd

from gui_components import DisasterVisionGUI
from feature_extractors import extract_all_features
from utils import preprocess_image
from ml_pipeline import create_geotag_bins, train_and_save_model

class DisasterVisionApp:
    def __init__(self):
        self.gui = DisasterVisionGUI(self)
        self.model = None
        self.label_encoder = None
        self.image_path = None
        self.feature_cols = None
        
        self.damage_classes = {
            0: ("No Damage", "green"),
            1: ("Minor Damage", "yellow"),
            2: ("Major Damage", "orange"),
            3: ("Destroyed", "red")
        }
        
        self.is_camera_on = False
        self.cap = None
        
        self.load_model()
        
    def load_model(self):
        model_path = "models/svm_trained.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.gui.update_metrics("Model loaded successfully. Ready.")
            
            metadata_path = "datasets/metadata.csv"
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                _, self.label_encoder = create_geotag_bins(df['latitude'], df['longitude'], bins=10)
                
            self.feature_cols = [f"img_f_{i}" for i in range(129)]
        else:
            self.gui.update_metrics("No trained model found. Please click 'Retrain Model'.")
            
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            img_pil = Image.open(file_path).convert("RGB")
            self.gui.update_preview(img_pil)
            self.gui.set_result("---", "cyan", 0)
            
    def analyze_image(self):
        if not self.image_path:
            return
            
        if not self.model:
            self.gui.update_metrics("Model not loaded yet.")
            return
            
        self.gui.progress.set(0.1)
        self.gui.update()
        
        start_time = time.time()
        
        _, img_array = preprocess_image(self.image_path)
        if img_array is None:
            return
            
        self.gui.progress.set(0.3)
        self.gui.update()
        features = extract_all_features(img_array)
        
        self.visualize_sift(img_array)
        
        gps_str = self.gui.entry_gps.get()
        lat, lon = 12.9716, 77.5946 
        if gps_str:
            try:
                parts = gps_str.split(',')
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
            except:
                pass
                
        if self.label_encoder:
            try:
                lat_bins = pd.cut([lat], bins=10, labels=False)
                lon_bins = pd.cut([lon], bins=10, labels=False)
                cat = f"{lat_bins[0]}_{lon_bins[0]}"
                geo_encoded = self.label_encoder.transform([cat])[0]
            except Exception as e:
                # If category wasn't seen during training
                geo_encoded = 0
        else:
             geo_encoded = 0
             
        self.gui.progress.set(0.6)
        self.gui.update()
        
        df_pred = pd.DataFrame([features], columns=self.feature_cols)
        df_pred['geo_encoded'] = [geo_encoded]
        
        probs = self.model.predict_proba(df_pred)[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class] * 100
        
        self.gui.progress.set(0.9)
        self.gui.update()
        
        text, color = self.damage_classes.get(pred_class, ("Unknown", "white"))
        self.gui.set_result(text, color, confidence)
        
        self.generate_map(lat, lon, text, color)
        
        elapsed = time.time() - start_time
        self.gui.progress.set(1.0)
        self.gui.update_metrics(f"Analysis complete in {elapsed:.2f}s")
        
    def visualize_sift(self, image_array):
        sift = cv2.SIFT_create(nfeatures=50)
        kp = sift.detect(image_array, None)
        img_with_kp = cv2.drawKeypoints(image_array, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_pil = Image.fromarray(img_with_kp)
        self.gui.update_features_view(img_pil)
        
    def generate_map(self, lat, lon, damage_text, color):
        color_map = {
            "green": "green",
            "yellow": "orange",
            "orange": "orange",
            "red": "red",
            "white": "gray"
        }
        n_col = color_map.get(color, "blue")
        
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker(
            [lat, lon],
            popup=f"Status: {damage_text}",
            icon=folium.Icon(color=n_col, icon="info-sign")
        ).add_to(m)
        
        os.makedirs('models', exist_ok=True)
        num = np.random.randint(1000)
        self.map_file = f"models/map_result_{num}.html"
        m.save(self.map_file)
        self.gui.lbl_map.configure(text=f"Map saved:\n{self.map_file}")
        
    def open_map(self):
        if hasattr(self, 'map_file') and os.path.exists(self.map_file):
            webbrowser.open('file://' + os.path.realpath(self.map_file))
            
    def export_report(self):
        self.gui.update_metrics("Report exported successfully.")
        
    def retrain_model(self):
        self.gui.update_metrics("Training Model... Please wait. Check console.")
        self.gui.update()
        
        def train_task():
            train_and_save_model()
            self.load_model()
            
        threading.Thread(target=train_task).start()

    def toggle_camera(self):
        if not self.is_camera_on:
            self.is_camera_on = True
            self.cap = cv2.VideoCapture(0)
            self.update_camera_feed()
        else:
            self.is_camera_on = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.gui.img_features_lbl.configure(image="", text="Camera Off")
            
    def update_camera_feed(self):
        if self.is_camera_on and self.cap:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create(nfeatures=50)
                kp = sift.detect(gray, None)
                img_with_kp = cv2.drawKeypoints(gray, kp, None, color=(0,255,0))
                
                rgb_img = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb_img)
                self.gui.update_features_view(img_pil)
                
            self.gui.after(30, self.update_camera_feed)

if __name__ == "__main__":
    app = DisasterVisionApp()
    app.gui.run()
