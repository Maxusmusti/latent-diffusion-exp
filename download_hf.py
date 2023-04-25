import os
import json
from io import BytesIO
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from PIL import Image
from datasets import load_dataset
import argparse
from tqdm import tqdm

"""
download_hf.py

This script downloads images and their metadata from a Hugging Face dataset using the specified parameters.
It supports datasets with image bytes or URLs and saves the images and metadata in the specified output directory.
It uses parallel workers for faster downloading and processes the dataset without loading all the data at once.

Usage:
    python download_hf.py --dataset_name <dataset_name> [--output_dir <output_dir>] [--rows <rows>] [--workers <workers>] [--image_bytes_key <image_bytes_key>] [--url_key <url_key>] [--metadata_columns <metadata_columns>...]

Arguments:
    --dataset_name       Name of the dataset to download from Hugging Face (required)
    --output_dir         Directory to save the downloaded images and metadata (default: "images") (optional)
    --rows               Number of rows to download (default: -1, which means all rows) (optional)
    --workers            Number of concurrent workers for downloading images, which speeds up the process (default: 4) (optional)
    --image_bytes_key    Key for image bytes in the dataset, if present (optional)
    --url_key            Key for image URL in the dataset, if present (optional)
    --metadata_columns   List of metadata column names to save (optional)

Examples:
    For laion/laion-art dataset:
    python download_hf.py --dataset_name laion/laion-art --url_key URL --metadata_columns TEXT WIDTH HEIGHT similarity LANGUAGE hash pwatermark punsafe aesthetic --rows 1000 --output_dir laion_art_images --workers 8

    For huggan/wikiart dataset:
    python download_hf.py --dataset_name huggan/wikiart --image_bytes_key image --metadata_columns artist genre style --rows 1000 --output_dir wikiart_images --workers 8
"""

def download_image_and_metadata(row, output_dir, metadata_columns, image_bytes_key, url_key, idx):
    try:
        # Check if the dataset contains image bytes directly
        if image_bytes_key in row:
            img = row[image_bytes_key]
        # Check if the dataset contains image URLs
        elif url_key in row:
            with urlopen(row[url_key], timeout=5) as response:
                r = BytesIO(response.read())
                img = Image.open(r)
        else:
            return False

        # Save the image and metadata to the output directory
        filename = os.path.join(output_dir, f"{idx:07d}")
        with open(f"{filename}.jpg", "wb") as f:
            img.save(f, "JPEG")

        metadata = {key: row[key] for key in metadata_columns}
        with open(f"{filename}.json", "w") as f:
            json.dump(metadata, f)
        return True
    except Exception as e:
        print(f"Error downloading image {idx}: {e}")
        return False

# Main function to handle downloading images and metadata
def process_downloads(output_dir, rows, dataset_name, image_bytes_key, url_key, metadata_columns, workers):
    output_dir, rows, dataset_name, image_bytes_key, url_key, metadata_columns

    os.makedirs(output_dir, exist_ok=True)

    # Load the specified dataset and slice it to the desired number of rows
    print("Initializing dataset...")
    dataset = load_dataset(dataset_name, split=f"train", streaming=(rows > -1))
    if rows > -1:
        dataset = itertools.islice(dataset, rows)

    success_count = 0

    # Use ThreadPoolExecutor to download images in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Initialize the progress bar and counters
        progress_bar = tqdm(total=rows, desc="Downloading images", unit="image")
        futures = []

        # Submit the download tasks one by one, updating the progress bar as each task is submitted
        for idx, row in enumerate(dataset):
            future = executor.submit(download_image_and_metadata, row, args.output_dir, args.metadata_columns, args.image_bytes_key, args.url_key, idx)
            futures.append(future)
            progress_bar.update(1)
            if idx > rows and rows > -1:
                break

        # Reset the progress bar for processing the download results
        progress_bar.reset()
        progress_bar.set_description("Processing results")

        success_count = 0

        # Process the download results as they complete
        for future in as_completed(futures):
            if future.result():
                success_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"success_rate": f"{100 * success_count / progress_bar.n:.2f}%"})

        # Close the progress bar
        progress_bar.close()

    # Print the percentage of successful downloads
    success_percentage = (success_count / len(futures)) * 100
    print(f"{success_percentage:.2f}% of downloads were successful.")
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download images and metadata from a Hugging Face dataset")

  parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to download from Hugging Face")
  parser.add_argument("--output_dir", type=str, default="images", help="Directory to save the downloaded images and metadata")
  parser.add_argument("--rows", type=int, default=-1, help="Number of rows to download (default: -1, which means all rows)")
  parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers for downloading images (default: 4)")
  parser.add_argument("--image_bytes_key", type=str, default=None, help="Key for image bytes in the dataset, if present")
  parser.add_argument("--url_key", type=str, default=None, help="Key for image URL in the dataset, if present")
  parser.add_argument("--metadata_columns", type=str, nargs="*", help="List of metadata column names to save (optional)")

  args = parser.parse_args()
  process_downloads(args.output_dir, args.rows, args.dataset_name, args.image_bytes_key, args.url_key, args.metadata_columns, args.workers)
