import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import pandas as pd

def generate_noise_image(size=(256, 256)):
    """Generate a random noise base image."""
    noise = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(1))
    return img

def add_structures(img, num_buildings=5):
    """Draw some rectangles representing buildings on the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(num_buildings):
        x1 = np.random.randint(0, w - 50)
        y1 = np.random.randint(0, h - 50)
        x2 = x1 + np.random.randint(20, 50)
        y2 = y1 + np.random.randint(20, 50)
        color = tuple(np.random.randint(50, 200, 3))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0,0,0))
    return img

def apply_damage(img, severity):
    """
    Apply visual damage based on severity (0 to 3).
    0: No damage
    1: Minor damage (small noise patches)
    2: Major damage (large destruction patches, darker)
    3: Destroyed (heavy blurring, major discoloration)
    """
    img_damaged = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img_damaged)
    
    if severity == 0:
        pass
    elif severity == 1:
        # Minor damage: few spots
        for _ in range(10):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            r = np.random.randint(2, 5)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(100, 100, 100))
    elif severity == 2:
        # Major damage: large chaotic blocks
        for _ in range(15):
            x1 = np.random.randint(0, w - 30)
            y1 = np.random.randint(0, h - 30)
            x2 = x1 + np.random.randint(10, 40)
            y2 = y1 + np.random.randint(10, 40)
            draw.rectangle([x1, y1, x2, y2], fill=(50, 40, 30))
    elif severity == 3:
        # Destroyed
        img_damaged = img_damaged.filter(ImageFilter.BoxBlur(2))
        for _ in range(30):
            x1 = np.random.randint(0, w - 40)
            y1 = np.random.randint(0, h - 40)
            x2 = x1 + np.random.randint(20, 60)
            y2 = y1 + np.random.randint(20, 60)
            draw.rectangle([x1, y1, x2, y2], fill=(30, 30, 30))
    return img_damaged

def main():
    np.random.seed(42)
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = 1000
    metadata = []
    
    # Bengaluru coordinates (12.97°N, 77.59°E ±0.5°)
    base_lat = 12.97
    base_lon = 77.59
    
    print("Generating synthetic dataset...")
    for i in tqdm(range(num_samples)):
        # Generate base pre-disaster image
        pre_img = generate_noise_image()
        pre_img = add_structures(pre_img, num_buildings=np.random.randint(3, 8))
        
        # Determine damage severity (0=No, 1=Minor, 2=Major, 3=Destroyed)
        severity = int(np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25]))
        
        # Apply damage
        post_img = apply_damage(pre_img, severity)
        
        # Generate GPS
        lat = base_lat + np.random.uniform(-0.5, 0.5)
        lon = base_lon + np.random.uniform(-0.5, 0.5)
        
        pre_filename = f"pre_{i:04d}.png"
        post_filename = f"post_{i:04d}.png"
        
        pre_img.save(os.path.join(output_dir, pre_filename))
        post_img.save(os.path.join(output_dir, post_filename))
        
        metadata.append({
            'sample_id': i,
            'pre_image': pre_filename,
            'post_image': post_filename,
            'latitude': lat,
            'longitude': lon,
            'damage_severity': severity
        })
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Generated {num_samples} samples successfully.")

if __name__ == "__main__":
    main()
