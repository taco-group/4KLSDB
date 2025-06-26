import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
import logging
import os
import signal
import sys

# === Configuration ===
patch_size = 240
flat_threshold = 100
Laplacian_Threshold_low = 100
Laplacian_Threshold_high = 10000

CHECKPOINT_INTERVAL = 50000
CHECKPOINT_FILE = "checkpoint.csv"
PROGRESS_FILE = "progress_checkpoint.txt"
CSV_INPUT_FILE = ""  # your input CSV file

# === Logging Setup ===
logging.basicConfig(
    filename='image_processing_third_trail.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# === Graceful Shutdown Handler ===
def graceful_shutdown(signum, frame):
    logging.info("Received termination signal. Saving progress before exit.")
    save_checkpoint(current_index, good_rows)
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)   # handle Ctrl+C
signal.signal(signal.SIGTERM, graceful_shutdown)  # handle termination signal

# === Checkpoint Saving Function ===
def save_checkpoint(idx, good_rows):
    try:
        pd.DataFrame(good_rows).to_csv(CHECKPOINT_FILE, index=False)
        with open(PROGRESS_FILE, "w") as pf:
            pf.write(str(idx))
        logging.info(f"Checkpoint saved at image index: {idx}")
    except Exception as e:
        logging.error(f"Error saving checkpoint at index {idx}: {e}")

# === Image Quality Functions ===
def sobel_variance(image):
    try:
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
        # If more than 50% of patches are flat, consider it as low-quality
        return flat_ratio >= 0.65
    except Exception as e:
        logging.error(f"Error in sobel_variance calculation: {e}")
        return True  # Assume flat (reject) if error occurs

def laplacian_variance(image):
    try:
        return cv2.Laplacian(image, cv2.CV_64F).var()
    except Exception as e:
        logging.error(f"Error in laplacian_variance calculation: {e}")
        return 0  # Return a value that will likely cause the image to be skipped

# === Main Processing ===
if __name__ == "__main__":
    try:
        # Read input CSV containing image URLs
        df = pd.read_csv(CSV_INPUT_FILE, engine="python", dtype=str)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Initialize or resume checkpoint
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as pf:
                last_index = int(pf.read().strip())
            logging.info(f"Resuming from index {last_index + 1}")
        except Exception as e:
            logging.error(f"Error reading progress file: {e}")
            last_index = -1
    else:
        last_index = -1

    if os.path.exists(CHECKPOINT_FILE):
        try:
            good_rows_df = pd.read_csv(CHECKPOINT_FILE, dtype=str)
            good_rows = good_rows_df.to_dict('records')
            logging.info(f"Loaded {len(good_rows)} previously accepted images.")
        except Exception as e:
            logging.error(f"Error reading checkpoint file: {e}")
            good_rows = []
    else:
        good_rows = []

    total_images = len(df)
    current_index = last_index  # will update in loop

    try:
        # Process images starting from the next index
        for idx in range(last_index + 1, total_images):
            current_index = idx  # update current index for checkpointing and signal handling
            row = df.iloc[idx]
            url_link = row.get('url', None)
            if not url_link:
                logging.warning(f"Row {idx} missing URL. Skipping.")
                continue

            try:
                # Use a context manager to ensure the connection is properly closed
                with requests.get(url_link, stream=True, timeout=10) as response:
                    response.raise_for_status()
                    # Open image using PIL and convert to RGB
                    im = Image.open(response.raw).convert("RGB")
            except Exception as e:
                logging.error(f"Error downloading or opening image at index {idx} ({url_link}): {e}")
                continue

            try:
                # Convert image to numpy array and then to grayscale
                img_np = np.array(im)
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                logging.error(f"Error processing image at index {idx}: {e}")
                continue

            try:
                result_lap = laplacian_variance(gray_img)
                result_sobel = sobel_variance(gray_img)
            except Exception as e:
                logging.error(f"Error in quality checks for image at index {idx}: {e}")
                continue

            logging.info(f"Image {idx} - Laplacian Variance: {result_lap}, Flat detection: {result_sobel}")

            # Check quality thresholds and decide acceptance
            if result_lap <= Laplacian_Threshold_low or result_lap >= Laplacian_Threshold_high or result_sobel:
                logging.info(f"Image {idx} rejected based on quality thresholds.")
            else:
                logging.info(f"Image {idx} accepted.")
                good_rows.append(row)

            # Save checkpoint every CHECKPOINT_INTERVAL images processed
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(idx, good_rows)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught. Saving current progress...")
        save_checkpoint(current_index, good_rows)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error at image index {current_index}: {e}")
        save_checkpoint(current_index, good_rows)
        sys.exit(1)

    # Final save after processing all images
    save_checkpoint(current_index, good_rows)
    logging.info("Processing complete. Final checkpoint saved.")
