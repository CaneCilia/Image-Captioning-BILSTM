import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

class MockCaptionDataset(Dataset):
    def __init__(self, size=100, max_length=15, feature_dim=512, vocab_size=1000):
        self.size = size
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return mock YOLO features and a random caption sequence
        features = torch.randn(self.feature_dim)
        caption = torch.randint(1, self.vocab_size, (self.max_length,))
        return features, caption

def get_dataloader(data_dir, annotations_file, batch_size, transform=None):
    # Using mock dataset so it runs out of the box
    dataset = MockCaptionDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
