import tkinter as tk
from tkinter import filedialog
import torch
from PIL import Image, ImageTk
import numpy as np
import xarray as xr
from climatenet.models import CGNet
from climatenet.utils.data import ClimateDatasetLabeled
from climatenet.utils.utils import Config
from os import path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Project App")

        # Model initialization (replace with your model loading logic)
        self.model = CGNet(model_path="./models/1. Weighted Jaccard-loss/")  # Assuming your model is named CGNet

        # Create a button to open a file dialog
        self.open_button = tk.Button(root, text="Choose NetCDF File", command=self.choose_file)
        self.open_button.pack(pady=20)

        # Create a label to display the selected file path
        self.file_label = tk.Label(root, text="")
        self.file_label.pack(pady=10)

        # Create a button to perform prediction
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=20)

        # Create an image display area
        self.figure, self.ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.Orthographic(-80, 35)})
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def choose_file(self):
        # Open a file dialog to select a NetCDF file
        file_path = filedialog.askopenfilename(title="Select a NetCDF file")
        folder_path = path.dirname(file_path)
        # Update the label with the selected file path
        self.file_label.config(text=f"Selected File: {file_path}")
        self.netcdf_file_path = file_path
        self.folder_path = folder_path

    def predict(self):
        if hasattr(self, 'netcdf_file_path'):
            # Load the NetCDF file
            dataset = xr.open_mfdataset(self.netcdf_file_path)

            # Create a ClimateDatasetLabeled instance (assuming ClimateDatasetLabeled extends Dataset)
            test_dataset = ClimateDatasetLabeled(self.folder_path, config=Config("./models/1. Weighted Jaccard-loss/config.json"))  # Replace 'config' with your actual configuration

            # Perform prediction
            predictions = self.model.predict(test_dataset)

            # Display the result (you may customize this part based on your needs)
            self.display_result(predictions)
        else:
            print("Please choose a NetCDF file first.")
    
    def map_image(self, image, lon=-80, lat=35):
        self.ax.clear()
        

    def display_result(self, predictions):
        # Convert predictions to a format suitable for display
        # (modify this based on the structure of your predictions)
        image = predictions[0]
        self.map_image(image=image)

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()

    # Create an instance of the app
    app = MLApp(root)

    # Run the main loop
    root.mainloop()

