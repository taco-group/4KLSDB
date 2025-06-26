import pandas as pd
import os
from glob import glob

# Define the directory containing parquet files
parquet_dir = ''  # Update this to the correct relative path to your parquet files
output_csv_dir = ''  # Update this to the correct relative path for output
output_csv = os.path.join(output_csv_dir, '4K_PD12M.csv')

# Ensure the output directory exists
os.makedirs(output_csv_dir, exist_ok=True)

# Filtering function
def filter_images(row):
    w = row['width']
    h = row['height']
    url = row['url']
    caption = row['caption']
   
    
    aspect_ratio = w / h
    resolution = h * w
    tested_side = max(h, w)
    
    if (
        (resolution >= 3840 * 2160) and
        (tested_side >= 3840) and
        (0.6 <= aspect_ratio <= 1.5)
    ):
        return {'width': w, 'height': h, 'url': url, 'caption':caption}
    else:
        return None

# List all parquet files in the directory
parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))

# Process each parquet file
filtered_results = []

for file in parquet_files:
    print(f"Processing {file}...")
    df = pd.read_parquet(file)
    
    # Apply the filtering function to each row
    filtered = df.apply(filter_images, axis=1)
    
    # Drop rows that didn't pass the filter (i.e., `None`)
    filtered = filtered.dropna().tolist()
    
    # Append results
    filtered_results.extend(filtered)

# Convert results into a DataFrame
filtered_df = pd.DataFrame(filtered_results)

# Save to CSV
filtered_df.to_csv(output_csv, index=False)

print(f"Filtered results saved to {output_csv}")
