import pandas as pd

df = pd.read_csv("combined.csv")

# Compute the 20th percentile thresholds for each score
quality_threshold = df['quality_score'].quantile(0.20)
aesthetic_threshold = df['aesthetic_score'].quantile(0.20)

# Filter the DataFrame to only include images with both scores above the thresholds
filtered_df = df[
    (df['quality_score'] >= quality_threshold) & 
    (df['aesthetic_score'] >= aesthetic_threshold)
]

# Optionally, save the filtered DataFrame to a new CSV
filtered_df.to_csv('filtered_images.csv', index=False)
