import pandas as pd
import os

# Path to the single CSV file
csv_file = ''  # Update with the actual path to your input file

# Directory for the filtered CSV file
output_csv_dir = ''  # Update as needed
os.makedirs(output_csv_dir, exist_ok=True)

# Read the entire CSV, preserving all columns
df = pd.read_csv(csv_file)

# Convert shot_distance and style columns to lowercase strings for the comparison
df['shot_distance'] = df['shot_distance'].astype(str).str.lower()
df['style'] = df['style'].astype(str).str.lower()

# Filter rows: keep only those where both shot_distance and style != 'unknown'
df_filtered = df[(df['shot_distance'] != 'unknown') & (df['style'] != 'unknown')]

# Prepare output filename
base_name = os.path.basename(csv_file)
base_name_no_ext = os.path.splitext(base_name)[0]
output_csv_file = os.path.join(output_csv_dir, f"{base_name_no_ext}_without_unknown.csv")

# Write the filtered DataFrame to a new CSV
df_filtered.to_csv(output_csv_file, index=False)

# Print some stats
total_num = len(df)
filtered_num = len(df_filtered)
error_num = total_num - filtered_num

print(f"Processed {csv_file}")
print(f"Total rows: {total_num}")
print(f"Filtered rows: {filtered_num}")
print(f"Rows removed (unknown shot_distance or style): {error_num}")
print(f"Filtered CSV saved to {output_csv_file}")
