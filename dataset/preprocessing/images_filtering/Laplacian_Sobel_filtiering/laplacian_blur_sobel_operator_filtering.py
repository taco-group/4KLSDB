import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

patch_size = 240
flat_threshold = 800
Laplacian_Threshold_low = 150
Laplacian_Threshold_high = 8000

def sobel_variance(image):
    # Calculate Sobel gradients in x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    height, width = magnitude.shape

    total_processed_patch = 0
    flat_region_count = 0

    # Process image in non-overlapping patches
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = magnitude[i:min(i + patch_size, height), j:min(j + patch_size, width)]
            total_processed_patch += 1
            var_sobel = np.var(patch)
            if var_sobel < flat_threshold:
                flat_region_count += 1

    flat_ratio = flat_region_count / total_processed_patch if total_processed_patch > 0 else 0
    # If more than 50% of patches are flat, return True
    return flat_ratio >= 0.5

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

if __name__ == "__main__":
    file_path = ""  # Path to your CSV with image URLs
    df = pd.read_csv(file_path, engine="python", dtype=str)

    # List to store rows that pass quality checks
    good_rows = []

    for idx, row in df.iterrows():
        url_link = row['url']
        try:
            response = requests.get(url_link, stream=True, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error downloading {url_link}: {e}")
            continue

        # Open image using PIL and convert to RGB
        im = Image.open(response.raw).convert("RGB")
        # Convert PIL image to numpy array
        img_np = np.array(im)
        # Convert RGB to grayscale (using COLOR_RGB2GRAY, since image is in RGB)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        result_lap = laplacian_variance(gray_img)
        result_sobel = sobel_variance(gray_img)
        
        print(f"Image {idx} - Laplacian Variance: {result_lap}, Flat detection: {result_sobel}")
        
        # Check thresholds: if image is too blurry or has too many flat regions, skip it
        if result_lap <= Laplacian_Threshold_low or result_lap >= Laplacian_Threshold_high or result_sobel:
            print(" --> Image rejected based on quality thresholds.")
            continue  # Skip this row
        else:
            print(" --> Image accepted.")
            good_rows.append(row)
    
    # Create a new DataFrame with the filtered rows and save to a new CSV file
    filtered_df = pd.DataFrame(good_rows)
    filtered_df.to_csv("filtered_images.csv", index=False)
