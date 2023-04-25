"""
This module creates a Pytorch DataLoader2 for Laion-5B (https://laion.ai/blog/laion-5b/) and contains related utility functions.
Laion-5B is a massive dataset popular for training large models like diffusion models.
LAION released the laion2B-en-joined dataset (https://huggingface.co/datasets/laion/laion2B-en-joined) which we will use here. 

References for the code:
- https://pytorch.org/data/beta/dataloader2.html
- https://pytorch.org/data/main/examples.html
- https://pytorch.org/data/main/generated/torchdata.datapipes.iter.HuggingFaceHubReader.html
- https://pytorch.org/data/main/generated/torchdata.datapipes.iter.Slicer.html
- https://pytorch.org/data/main/generated/torchdata.datapipes.iter.Dropper.html
- https://pytorch.org/data/main/generated/torchdata.datapipes.map.Batcher.html
"""

import numpy as np
from matplotlib import pyplot as plt
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import HuggingFaceHubReader
from io import BytesIO
import requests
import PIL
from PIL import Image
import time
import torchvision.transforms as T
from transformers import ViTFeatureExtractor
import os
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")

def load_image_from_url(URL):
    """
    Loads an image from URL using requests package and BytesIO package. Returns image as tensor
        Performs preprocessing so that the channels are normalized and image is resized to 224x224.
    URL: the URL from which to load the image
    """
    try:
        call = requests.get(URL, timeout=5) # timeout of 5 seconds so we don't hang indefinitely
        image = Image.open(BytesIO(call.content))
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values[0]
        return pixel_values
    except:
        return None
    
    
def quality_filters(x):
    """
    Returns whether the entry in the dataset meets quality filters: no watermark and is sfw
        Threshold for watermark: 0.8
        Threshold for nsfw: 0.5
    x: Row in dataset
    """
    return x["pwatermark"] is not None and x["pwatermark"] < 0.8 and x["punsafe"] is not None and x["punsafe"] < 0.5


def get_iterdatapipe(path):
    """
    Loads the IterDataPipe from the specified HuggingFace path and returns a DataPipe
    path: the HuggingFace path following https://huggingface.co/datasets/
    """
    data = HuggingFaceHubReader(path) # returns an iterable HuggingFace dataset
    data = data.filter(quality_filters) # filters out images with watermark and images that are unsafe
    data = data.shuffle().sharding_filter() # allows DataPipe to be sharded
    data = data.slice(index=["TEXT", "URL"]) # get columns by index
    data = data.map(fn=load_image_from_url, input_col="URL", output_col="IMAGE") # load each image and put it in a new "IMAGE" column
    data = data.filter(filter_fn=lambda x: x is not None, input_col="IMAGE") # filter out rows with images that couldn't be loaded
    data = data.drop("URL") # drop this column since we loaded images and don't need URL anymore
    data = data.batch(20) # Create mini-batches of data of size 20
    return data


# TODO: limit datset size
def get_data_loader_2(path):
    """
    Creates a data loader from the given HuggingFace path
    path: the HuggingFace path following https://huggingface.co/datasets/
    """
    dataset = get_iterdatapipe(path)
    reading_service = MultiProcessingReadingService(num_workers=4) # Spawns 4 worker processes to load data from the DataPipe
    data_loader = DataLoader2(dataset, reading_service=reading_service) # Create data loader with the 4 worker processes
    return data_loader


def unnormalize_tensor(tensor, mean, std):
    """
    Un-normalizes a tensor with given mean and standard-deviation. (For displaying purposes)
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
    

def view_entry(entry, debug=False):
    if not debug:
        return
    
    label, tensor_image = entry["TEXT"], entry["IMAGE"]
    try:
        print("\t\t", label)
        tensor_to_image = T.Compose([T.ToPILImage()])
        plt.imshow(tensor_to_image(unnormalize_tensor(tensor_image, np.array(feature_extractor.image_mean), np.array(feature_extractor.image_std))))
        plt.savefig('./image.png')
        plt.clf()
        time.sleep(0.2)
    except PIL.UnidentifiedImageError:
        print("corrupted")



image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the short side to 256, maintaining the aspect ratio
    transforms.CenterCrop(224),     # Crop the center square of size 224x224 from the resized image
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(           # Normalize the image tensor using mean and std values for ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ImageMetaDataset(Dataset):
    """
    A custom dataset class for loading images and their metadata from a specified directory.

    Attributes:
        root_dir (str): Path to the directory containing the images and metadata.
        transform (callable, optional): Optional transform to be applied to the image.
        image_files (List[str]): List of image file names in the directory.
        metadata_files (List[str]): List of metadata file names in the directory.
    """

    def __init__(self, root_dir, transform=image_transforms):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.metadata_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]

        # Ensure the image and metadata files are in the same order
        self.image_files.sort()
        self.metadata_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        metadata_path = os.path.join(self.root_dir, self.metadata_files[idx])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if self.transform: # if you are looking to use default collate option from dataloader, make sure to specify this and change and create equal shape tensors
            image = self.transform(image)
        return image, metadata

def load_data(dataset_path, batch_size = 4):
    train_data = ImageMetaDataset(dataset_path, resolution=256)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_loader

if __name__ == "__main__":
    data_loader = get_data_loader("laion/laion2B-en-joined")

    for i, batch in enumerate(data_loader):
        print("Batch", i)
        for j, entry in enumerate(batch):
            print("\tEntry", j)
            view_entry(entry, debug=True)
