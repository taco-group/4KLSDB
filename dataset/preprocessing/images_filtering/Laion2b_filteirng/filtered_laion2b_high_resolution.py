import pandas as pd
import os
from glob import glob

# Define the directory containing input parquet files
parquet_dir = ''  # Update this if needed

# Define output directory for filtered parquet files
output_parquet_dir = ''  # Update as needed
os.makedirs(output_parquet_dir, exist_ok=True)

def filter_images(row):
    w = row['WIDTH']
    h = row['HEIGHT']
    url = row['URL']
    caption = row['TEXT']
    aesthetic = row['aesthetic']
    
    aspect_ratio = w / h
    resolution = h * w
    tested_side = max(h, w)

    if (
        (resolution >= 3840 * 2160) and
        (tested_side >= 3840) and
        (0.6 <= aspect_ratio <= 1.5)
    ):
        return {'width': w, 'height': h, 'url': url, 'caption': caption, 'aesthetic_score':aesthetic}
    else:
        return None

# List all parquet files in the directory
parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))

for file in parquet_files:
    print(f"Processing {file}...")
    df = pd.read_parquet(file)
    
    # Apply the filtering function to each row
    filtered = df.apply(filter_images, axis=1).dropna().tolist()
    
    # If there are filtered results, save them
    if filtered:
        filtered_df = pd.DataFrame(filtered)

        # Construct a unique output filename based on the input file name
        base_name = os.path.basename(file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        output_parquet_file = os.path.join(output_parquet_dir, f"{base_name_no_ext}_filtered.parquet")

        filtered_df.to_parquet(output_parquet_file, index=False)
        print(f"Filtered results saved to {output_parquet_file}")
    else:
        print(f"No results matched the criteria for {file}.")
