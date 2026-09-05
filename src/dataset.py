import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

class CaptionDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        \"\"\"
        Args:
            data_dir (string): Directory with all the images.
            annotations_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        \"\"\"
        self.data_dir = data_dir
        self.transform = transform
        
        # TODO: Load annotations (e.g., COCO or Flickr8k format)
        self.annotations = [] # Placeholder
        
        # TODO: Build vocabulary
        self.vocab = {} # Placeholder

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # TODO: Load image, apply transforms, convert caption to tensor
        pass

def get_dataloader(data_dir, annotations_file, batch_size, transform=None):
    dataset = CaptionDataset(data_dir, annotations_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
