from datasets import load_dataset
import os
from PIL import Image

# Define the dataset repository and the local directories for images and masks
dataset_repo = "deep-plants/AGM_HS"
image_dir = "C:\\Users\\Stell\\Desktop\\DVA 309\\Images10"
mask_dir = "C:\\Users\\Stell\\Desktop\\DVA 309\\masks10"

os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Load the dataset from Hugging Face Hub
dataset = load_dataset(dataset_repo)

# Function to retrieve dataset entries
def get_dataset_entries(dataset):
  
    return dataset['train']

# Function to download and save images and masks using the fields from the dataset entries
from itertools import count

# Modify the download_and_save function to use a sequential ID
def download_and_save(dataset_entries, image_dir, mask_dir):
    id_generator = count(start=1)  # Starts counting from 1
    for entry in dataset_entries:
        image = entry['image']
        mask = entry['mask']
        image_id = next(id_generator)  # Get the next ID in the sequence

        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(mask_dir, f"{image_id}_mask.png")

        image.save(image_path)
        mask.save(mask_path)


dataset_entries = get_dataset_entries(dataset)


download_and_save(dataset_entries, image_dir, mask_dir)



