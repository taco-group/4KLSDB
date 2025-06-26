import os
import json

# Path to your images directory (adjust this path accordingly)
images_dir = ''

# List all image files (you can filter by file extension)
allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif')
files = [f for f in os.listdir(images_dir) if f.lower().endswith(allowed_extensions)]

# Build a list of image metadata dictionaries
images_metadata = []
for idx, filename in enumerate(sorted(files), start=1):
    image_path = f"/hr_images/{filename}"  # This URL will work with Flask's static files
    images_metadata.append({
        "id": idx,
        "url": image_path,
        "quality": None  # Initially, no quality decision has been made
    })

# Save the metadata to a JSON file
with open('images.json', 'w') as json_file:
    json.dump(images_metadata, json_file, indent=4)

print(f"Generated JSON file with {len(images_metadata)} entries.")
