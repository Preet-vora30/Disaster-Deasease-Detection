import customtkinter as ctk
import tkinter as tk

class DisasterVisionGUI(ctk.CTk):
    def __init__(self, main_app_controller):
        super().__init__()
        
        self.controller = main_app_controller
        self.title("DisasterVision AI")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._build_ui()
        
    def _build_ui(self):
        # Grid setup
        self.grid_columnconfigure(0, weight=1) # Left
        self.grid_columnconfigure(1, weight=2) # Center
        self.grid_columnconfigure(2, weight=1) # Right
        self.grid_rowconfigure(0, weight=4) # Main Panels
        self.grid_rowconfigure(1, weight=1) # Bottom
        
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._build_bottom_panel()
        
    def _build_left_panel(self):
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_upload = ctk.CTkLabel(self.left_frame, text="Upload Image", font=("Arial", 16, "bold"))
        self.lbl_upload.pack(pady=20)
        
        self.btn_upload = ctk.CTkButton(self.left_frame, text="Select Image(s)", command=self.controller.upload_image)
        self.btn_upload.pack(pady=10)
        
        self.img_preview_lbl = ctk.CTkLabel(self.left_frame, text="No Image", width=256, height=256, fg_color="gray30")
        self.img_preview_lbl.pack(pady=10)
        
        self.lbl_gps = ctk.CTkLabel(self.left_frame, text="GPS Coordinates (Lat, Lon):")
        self.lbl_gps.pack(pady=5)
        
        self.entry_gps = ctk.CTkEntry(self.left_frame, placeholder_text="12.9716, 77.5946")
        self.entry_gps.pack(pady=5)
        
        # Pulsing button
        self.btn_analyze = ctk.CTkButton(self.left_frame, text="ANALYZE", fg_color="#E74C3C", hover_color="#C0392B", 
                                        font=("Arial", 18, "bold"), command=self.controller.analyze_image)
        self.btn_analyze.pack(pady=30)
        
        self.progress = ctk.CTkProgressBar(self.left_frame, mode="determinate")
        self.progress.set(0)
        self.progress.pack(pady=10, fill="x", padx=20)
        
    def _build_center_panel(self):
        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_center = ctk.CTkLabel(self.center_frame, text="Feature Analysis (SIFT & Matches)", font=("Arial", 16, "bold"))
        self.lbl_center.pack(pady=10)
        
        self.img_features_lbl = ctk.CTkLabel(self.center_frame, text="Keypoints Viz", width=400, height=400, fg_color="gray30")
        self.img_features_lbl.pack(pady=10)
        
        self.btn_camera = ctk.CTkButton(self.center_frame, text="Toggle Live Camera", command=self.controller.toggle_camera)
        self.btn_camera.pack(pady=20)

    def _build_right_panel(self):
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        self.lbl_prediction = ctk.CTkLabel(self.right_frame, text="Damage Prediction", font=("Arial", 16, "bold"))
        self.lbl_prediction.pack(pady=10)
        
        self.lbl_result = ctk.CTkLabel(self.right_frame, text="---", font=("Arial", 28, "bold"), text_color="cyan")
        self.lbl_result.pack(pady=20)
        
        self.lbl_confidence = ctk.CTkLabel(self.right_frame, text="Confidence: --%", font=("Arial", 14))
        self.lbl_confidence.pack(pady=5)
        
        # Folium map placeholder text
        self.lbl_map = ctk.CTkLabel(self.right_frame, text="Generated Map: map.html", fg_color="gray20", width=250, height=200)
        self.lbl_map.pack(pady=20)
        
        self.btn_open_map = ctk.CTkButton(self.right_frame, text="Open Interactive Map", command=self.controller.open_map)
        self.btn_open_map.pack(pady=10)

    def _build_bottom_panel(self):
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        
        self.lbl_metrics = ctk.CTkLabel(self.bottom_frame, text="Metrics will appear here after model loading.", font=("Arial", 14))
        self.lbl_metrics.pack(side="left", padx=20, pady=20)
        
        self.btn_export = ctk.CTkButton(self.bottom_frame, text="Export Info", command=self.controller.export_report)
        self.btn_export.pack(side="right", padx=20, pady=20)
        
        self.btn_train = ctk.CTkButton(self.bottom_frame, text="Retrain Model", command=self.controller.retrain_model)
        self.btn_train.pack(side="right", padx=10, pady=20)
        
    def update_preview(self, img_pil):
        # We need to maintain reference
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(256, 256))
        self.img_preview_lbl.configure(image=img_ctk, text="")
        self.img_preview_lbl.image = img_ctk
        
    def update_features_view(self, img_pil):
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(400, 400))
        self.img_features_lbl.configure(image=img_ctk, text="")
        self.img_features_lbl.image = img_ctk
        
    def update_metrics(self, text):
        self.lbl_metrics.configure(text=text)

    def set_result(self, text, color, confidence):
        self.lbl_result.configure(text=text, text_color=color)
        self.lbl_confidence.configure(text=f"Confidence: {confidence:.1f}%")

    def run(self):
        self._pulse_button()
        self.mainloop()
        
    def _pulse_button(self):
        current_color = self.btn_analyze.cget("fg_color")
        next_color = "#C0392B" if current_color == "#E74C3C" else "#E74C3C"
        self.btn_analyze.configure(fg_color=next_color)
        self.after(800, self._pulse_button)
