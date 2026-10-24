import time
import threading
from PIL import Image
import os
import sys

# Needed for screenshot
try:
    import pyautogui
except ImportError:
    pass

from main import DisasterVisionApp

def auto_driver(app):
    print("Waiting for GUI to initialize...")
    time.sleep(3)
    
    print("Simulating Image Upload (post_0042.png)...")
    app.image_path = "datasets/post_0042.png"
    if os.path.exists(app.image_path):
        img_pil = Image.open(app.image_path).convert("RGB")
        app.gui.update_preview(img_pil)
        app.gui.set_result("---", "cyan", 0)
    else:
        print(f"Error: {app.image_path} not found.")
    
    time.sleep(1)
    
    print("Simulating GPS Input: 12.91, 77.62...")
    app.gui.entry_gps.delete(0, 'end')
    app.gui.entry_gps.insert(0, "12.91, 77.62")
    time.sleep(1)
    
    print("Simulating ANALYZE click...")
    # Just call analyze_image directly
    app.analyze_image()
    time.sleep(2)
    
    print("\n[PREVIEW RESULT]")
    res_text = app.gui.lbl_result.cget("text")
    conf_text = app.gui.lbl_confidence.cget("text")
    metrics = app.gui.lbl_metrics.cget("text")
    map_txt = app.gui.lbl_map.cget("text")
    
    print(f"Predicted Damage Level: {res_text}")
    print(f"Confidence Level: {conf_text}")
    print(f"Performance Metrics: {metrics}")
    print(f"Generated Map Path: {map_txt}")
    
    try:
        from PIL import ImageGrab
        print("\nTaking screenshot of the screen...")
        screenshot = ImageGrab.grab()
        artifact_dir = r"C:\Users\hp5cd\.gemini\antigravity\brain\eb828595-c0d1-47b1-b735-94410659c76b"
        os.makedirs(artifact_dir, exist_ok=True)
        filename = os.path.join(artifact_dir, "app_preview.png")
        screenshot.save(filename)
        print(f"Screenshot saved to: {filename}")
    except Exception as e:
        print(f"Could not take screenshot: {e}")
    
    # Wait a bit then close the app
    time.sleep(3)
    app.gui.quit()
    print("Test finished successfully.")
    
    # Brute exit to ensure clean closing
    os._exit(0)

if __name__ == "__main__":
    app = DisasterVisionApp()
    threading.Thread(target=auto_driver, args=(app,), daemon=True).start()
    app.gui.run()
