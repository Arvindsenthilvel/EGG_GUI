import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\mearv\Downloads\egg_cnn_model.h5")  # Update this path!

# GUI App
class EGGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EGG Signal Abnormality Detection")
        self.root.geometry("400x300")

        self.label = tk.Label(root, text="Upload an EGG signal CSV file", font=("Arial", 14))
        self.label.pack(pady=20)

        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.upload_csv)
        self.upload_btn.pack(pady=10)

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)

        self.data = None

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] != 3 or df.shape[0] != 2400:
                messagebox.showerror("Shape Error", f"Expected shape (2400, 3), but got {df.shape}")
                return
            self.data = df.values.astype(np.float32)
            self.label.config(text="File loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def predict(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please upload a CSV file first.")
            return
        try:
            # Reshape to match model input: (1, 2400, 3)
            input_data = np.expand_dims(self.data, axis=0)
            prediction = model.predict(input_data)
            result = "Postprandial (1)" if prediction[0][0] > 0.5 else "Fasting (0)"
            self.result_label.config(text=f"Prediction: {result}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Something went wrong:\n{e}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EGGApp(root)
    root.mainloop()



