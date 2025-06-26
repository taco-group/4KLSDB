import pandas as pd
import os
from glob import glob

# Define the directory containing input parquet files
parquet_dir = ''  # Update this if needed

# Define output directory for filtered parquet files
output_parquet_dir = ''  # Update as needed
os.makedirs(output_parquet_dir, exist_ok=True)

error_num = 0
total_num = 0

def filter_images(row):
    global error_num
    global total_num
    w = row['width']
    h = row['height']
    url = row['url']
    caption = row['caption']
    aesthetic = row['aesthetic_score']
    shot_type = row['shot_distance']
    style = row['style']

    if (shot_type == 'Error' or shot_type == 'error' or style == 'Error' or style == 'error'):
        error_num += 1
        total_num += 1
        return None
    else:
        total_num += 1
        return {'width': w, 'height': h, 'url': url, 'caption': caption, 'aesthetic_score':aesthetic, 'shot_type':shot_type, 'style':style, 'aespect_ratio': w/h} 

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

print(f"Total number of images processed: {total_num} with {error_num} errors.")
