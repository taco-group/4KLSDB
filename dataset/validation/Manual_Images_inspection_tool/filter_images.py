import json
import os
import shutil  # For moving files
from tqdm import tqdm  # For progress bars (requires pip install tqdm)

def filter_low_quality_images(json_file_path, image_base_dir, low_quality_dest_dir, action='move'):
    """
    Filter images marked as 'low' quality based on JSON metadata file.

    Args:
        json_file_path (str): Path to the JSON metadata file.
        image_base_dir (str): Base directory containing the actual image files.
        low_quality_dest_dir (str): Destination directory for low quality images (only needed when action='move').
        action (str): Operation to perform: 'move' (move), 'list' (list only), 'delete' (delete, use with caution!).
                      Default is 'move'.
    """

    # --- Safety checks ---
    if action not in ['move', 'list', 'delete']:
        print(f"Error: Invalid action '{action}'. Please choose 'move', 'list', or 'delete'.")
        return
    if action == 'move' and not low_quality_dest_dir:
        print("Error: 'low_quality_dest_dir' must be provided when action is 'move'.")
        return
    if action == 'delete':
        confirm = input(f"Warning: Action 'delete' will permanently remove files from '{image_base_dir}'.\n"
                       f"Are you sure you want to continue? (Type 'yes' to confirm): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
    # --- End safety checks ---

    # 1. Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:  # Specify utf-8 encoding for robustness
            metadata = json.load(f)
        print(f"Successfully loaded {len(metadata)} metadata entries from '{json_file_path}'.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: Cannot parse JSON in '{json_file_path}'. Please check file format.")
        return
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # 2. Identify low quality files and build paths
    low_quality_files_to_process = []
    low_quality_filenames_in_json = []
    print("Scanning metadata for low quality images...")

    for entry in tqdm(metadata, desc="Scanning JSON"):
        # Use .get() to safely access keys and avoid KeyError
        quality = entry.get('quality')
        url = entry.get('url')

        if quality == 'low':
            if not url:
                # print(f"Warning: Entry ID {entry.get('id', 'N/A')} marked as low but missing 'url'.")
                continue  # Skip entries without URL

            # Extract filename
            filename = os.path.basename(url)
            if not filename:
                # print(f"Warning: Cannot extract filename from URL '{url}' (ID: {entry.get('id', 'N/A')}).")
                continue  # Skip entries where filename cannot be extracted
                
            low_quality_filenames_in_json.append(filename)  # Record filenames marked in JSON

            # Build full actual path of the file
            full_image_path = os.path.join(image_base_dir, filename)

            # Check if file actually exists
            if os.path.isfile(full_image_path):
                low_quality_files_to_process.append(full_image_path)
            # else:
            #     # File marked as low in JSON but doesn't exist in actual folder, print warning (optional)
            #     print(f"Warning: File '{filename}' marked as low quality in JSON not found in directory '{image_base_dir}'.")

    num_json_low = len(low_quality_filenames_in_json)
    num_files_found = len(low_quality_files_to_process)
    print(f"\nFound {num_json_low} entries marked as 'low' quality in JSON.")
    print(f"Found {num_files_found} corresponding actual files in directory '{image_base_dir}'.")

    if num_files_found == 0:
        print("No low quality image files found to process.")
        return

    # 3. Execute operation
    if action == 'list':
        print("\nThe following low quality image files were found in the directory:")
        for file_path in low_quality_files_to_process:
            print(os.path.basename(file_path))

    elif action == 'move':
        print(f"\nPreparing to move {num_files_found} low quality files to '{low_quality_dest_dir}'...")
        os.makedirs(low_quality_dest_dir, exist_ok=True)  # Ensure destination directory exists
        moved_count = 0
        error_count = 0
        for file_path in tqdm(low_quality_files_to_process, desc="Moving files"):
            try:
                destination_path = os.path.join(low_quality_dest_dir, os.path.basename(file_path))
                # Avoid overwriting files that already exist in destination folder
                if not os.path.exists(destination_path):
                    shutil.move(file_path, destination_path)
                    moved_count += 1
                else:
                    print(f"Skipped: File '{os.path.basename(file_path)}' already exists in destination directory.")
                    # If you want to overwrite, remove the if not os.path.exists(...) check
            except Exception as e:
                print(f"\nError moving file '{os.path.basename(file_path)}': {e}")
                error_count += 1
        print(f"\nMove operation completed. Successfully moved {moved_count} files, {error_count} errors occurred.")

    elif action == 'delete':
        print(f"\nPreparing to delete {num_files_found} low quality files from '{image_base_dir}'...")
        deleted_count = 0
        error_count = 0
        for file_path in tqdm(low_quality_files_to_process, desc="Deleting files"):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"\nError deleting file '{os.path.basename(file_path)}': {e}")
                error_count += 1
        print(f"\nDelete operation completed. Successfully deleted {deleted_count} files, {error_count} errors occurred.")


# --- Configuration Section ---
if __name__ == "__main__":
    # 1. Set your JSON file path
    JSON_FILE = ''

    # 2. Set the folder path containing actual image files (like 228228.jpg)
    #    Even if URLs in JSON are '/hr_images/...', you need to fill in the real system path here
    IMAGE_FOLDER = ''

    # 3. Choose the operation to perform:
    #    'move'   - Move low quality images to the 'LOW_QUALITY_FOLDER' below (recommended, safest)
    #    'list'   - Only print the names of low quality image files found, don't move or delete
    #    'delete' - **Permanently delete** low quality images (please be very careful!)
    ACTION_TO_PERFORM = 'move'

    # 4. If ACTION_TO_PERFORM = 'move', set the destination folder path
    #    Can be a subdirectory of IMAGE_FOLDER, or a completely different path
    LOW_QUALITY_FOLDER = ''
    # Example: LOW_QUALITY_FOLDER = '/path/to/separate/low_quality_storage'

    # --- Run processing function ---
    filter_low_quality_images(
        json_file_path=JSON_FILE,
        image_base_dir=IMAGE_FOLDER,
        low_quality_dest_dir=LOW_QUALITY_FOLDER if ACTION_TO_PERFORM == 'move' else None,
        action=ACTION_TO_PERFORM
    )