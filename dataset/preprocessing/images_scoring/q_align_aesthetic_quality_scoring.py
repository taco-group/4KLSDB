import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "q-future/one-align", 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

def get_image_scores(url):
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/115.0.0.0 Safari/537.36")
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Ensure request was successful

        img = Image.open(response.raw).convert("RGB")
        # Optionally, you can resize if needed:
        # img = img.resize((512, 512), Image.LANCZOS)

        quality_score = model.score([img], task_="quality", input_="image")
        aesthetic_score = model.score([img], task_="aesthetics", input_="image")

        return quality_score, aesthetic_score
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None, None

# Read the CSV file (it must have a column named 'url')
input_csv = ""
df = pd.read_csv(input_csv, engine="python", dtype=str)

total_rows = len(df)
print(f"Total rows to process: {total_rows}")

# Preallocate lists for the scores (ensuring we have a slot for every row)
quality_scores = [None] * total_rows
aesthetic_scores = [None] * total_rows

checkpoint_interval = 20000
processed_count = 0

for idx, row in df.iterrows():
    url = row['url']
    q_score, a_score = get_image_scores(url)
    
    # Save the scores by index
    quality_scores[idx] = q_score
    aesthetic_scores[idx] = a_score

    processed_count += 1

    # Print progress every 1000 images (adjust as needed)
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_rows}")

    # Checkpoint every checkpoint_interval images
    if processed_count % checkpoint_interval == 0:
        df.loc[:processed_count - 1, "quality_score"] = [
            score.cpu().item() if score is not None else None
            for score in quality_scores[:processed_count]
        ]
        df.loc[:processed_count - 1, "aesthetic_score"] = [
            score.cpu().item() if score is not None else None
            for score in aesthetic_scores[:processed_count]
        ]

        output_csv = f"checkpoint_{processed_count}_second_cropped.csv"
        df.to_csv(output_csv, index=False)
        print(f"Checkpoint saved at {processed_count} images.")

# After processing, if the last checkpoint wasn't exactly at the end, save the final CSV.
if processed_count % checkpoint_interval != 0:
    df["quality_score"] = [
        score.cpu().item() if score is not None else None
        for score in quality_scores
    ]
    df["aesthetic_score"] = [
        score.cpu().item() if score is not None else None
        for score in aesthetic_scores
    ]

    output_csv = f"" # update with your desired output filename
    df.to_csv(output_csv, index=False)
    print("Final results saved.")

print(f"Finished processing {processed_count} images.")
