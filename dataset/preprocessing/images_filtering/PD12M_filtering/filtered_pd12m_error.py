import pandas as pd
import os

# Path to the single CSV file
csv_file = ''  # Update with the actual path to your input file

# Directory for the filtered CSV file
output_csv_dir = ''  # Update as needed
os.makedirs(output_csv_dir, exist_ok=True)

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

    # Check for errors
    if (shot_type == 'Error' or shot_type == 'error' or 
        style == 'Error' or style == 'error'):
        error_num += 1
        total_num += 1
        return None
    else:
        total_num += 1
        return {
            'width': w, 
            'height': h, 
            'url': url, 
            'caption': caption,
            'aesthetic_score': aesthetic, 
            'shot_type': shot_type, 
            'style': style,
            'aespect_ratio': w/h
        }

print(f"Processing {csv_file}...")
# Read the CSV file
df = pd.read_csv(csv_file)

# Apply the filtering function to each row
filtered = df.apply(filter_images, axis=1).dropna().tolist()

if filtered:
    filtered_df = pd.DataFrame(filtered)

    # Construct output filename
    base_name = os.path.basename(csv_file)
    base_name_no_ext = os.path.splitext(base_name)[0]
    output_csv_file = os.path.join(output_csv_dir, f"{base_name_no_ext}_filtered.csv")

    # Save the filtered data to CSV
    filtered_df.to_csv(output_csv_file, index=False)
    print(f"Filtered results saved to {output_csv_file}")
else:
    print(f"No results matched the criteria for {csv_file}.")

print(f"Total number of images processed: {total_num} with {error_num} errors.")
