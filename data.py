"""
This module creats a Pytorch DataLoader2 for Laion-5B (https://laion.ai/blog/laion-5b/) and contains related utility functions.
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


def load_image_from_url(URL):
    """
    Loads an image from URL using requests package and BytesIO package
    URL: the URL from which to load the image
    """
    try:
        call = requests.get(URL, timeout=5) # timeout of 5 seconds so we don't hang indefinitely
        return np.array(Image.open(BytesIO(call.content)))
    except:
        return None
    

def get_dataset(path):
    """
    Loads the dataset from the specified HuggingFace path and returns a DataPipe
    path: the HuggingFace path following https://huggingface.co/datasets/
    """
    data = HuggingFaceHubReader(path) # returns an iterable HuggingFace dataset
    data = data.shuffle().sharding_filter() # allows DataPipe to be sharded
    data = data.slice(index=["TEXT", "URL"]) # get columns by index
    data = data.map(fn=load_image_from_url, input_col="URL", output_col="IMAGE") # load each image and put it in a new "IMAGE" column
    data = data.filter(filter_fn=lambda x: x is not None, input_col="IMAGE") # filter out rows with images that couldn't be loaded
    data = data.drop("URL") # drop this column since we loaded images and don't need URL anymore
    data = data.batch(20) # Create mini-batches of data of size 20
    return data


def get_data_loader(path):
    """
    Creates a data loader from the given HuggingFace path
    path: the HuggingFace path following https://huggingface.co/datasets/
    """
    dataset = get_dataset(path)
    reading_service = MultiProcessingReadingService(num_workers=4) # Spawns 4 worker processes to load data from the DataPipe
    data_loader = DataLoader2(dataset, reading_service=reading_service) # Create data loader with the 4 worker processes
    return data_loader


def view_entry(entry, debug=False):
    if not debug:
        return
    
    label, image = entry["TEXT"], entry["IMAGE"]
    try:
        print("\t\t", label)
        plt.imshow(image)
        plt.savefig('./image.png')
        plt.clf()
        time.sleep(0.25)
    except PIL.UnidentifiedImageError:
        print("corrupted")


if __name__ == "__main__":
    data_loader = get_data_loader("laion/laion2B-en-joined")

    for i, batch in enumerate(data_loader):
        print("Batch", i)
        for j, entry in enumerate(batch):
            print("\tEntry", j)
            view_entry(entry, debug=False)
