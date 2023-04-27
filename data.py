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
from io import BytesIO
import requests
import PIL
from PIL import Image
import time
import torchvision.transforms as T
import os
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations
    
    
def quality_filters(x):
    """
    Returns whether the entry in the dataset meets quality filters: no watermark and is sfw
        Threshold for watermark: 0.8
        Threshold for nsfw: 0.5
    x: Row in dataset
    """
    return x["pwatermark"] is not None and x["pwatermark"] < 0.8 and x["punsafe"] is not None and x["punsafe"] < 0.5


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

    def __init__(self, root_dir, resolution = 256):
        self.root_dir = root_dir
        self.resolution = resolution

        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.metadata_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]

        # Ensure that each jpg has a json and vice-versa
        matches = set([i[:-4] for i in self.image_files]).intersection(set([i[:-5] for i in self.metadata_files]))
        self.image_files = [i+'.jpg' for i in matches]
        self.metadata_files = [i+'.json' for i in matches]

        # Ensure the image and metadata files are in the same order
        self.image_files.sort()
        self.metadata_files.sort()

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.resolution)
        self.cropper = albumentations.CenterCrop(height=self.resolution, width=self.resolution)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return len(self.image_files)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = self.preprocess_image(img_path)

        metadata_path = os.path.join(self.root_dir, self.metadata_files[idx])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return image, metadata

def load_data(dataset_path, batch_size, resolution, num_workers):
    train_data = ImageMetaDataset(dataset_path, resolution=resolution)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader
