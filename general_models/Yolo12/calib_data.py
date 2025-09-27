# by yhpark 2025-9-3
from datasets import load_dataset  
from torch.utils.data import DataLoader
import torch
import os 
import random
import numpy as np 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def set_random_seed(random_seed = 42):
   print("[set random seeds]")
   torch.manual_seed(random_seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(random_seed)
   random.seed(random_seed)

def dataset_load(batch_size, transform, data_type, num_workers=0):
    set_random_seed()
    shuffle = False
    if data_type == 'train':
        shuffle = True

    # Load the dataset test  
    dataset = load_dataset("ilee0022/ImageNet100", cache_dir=f"{CUR_DIR}/dataset")

    test_dataset = dataset[data_type].with_transform(lambda ex: {"image": ex["image"], "label": ex["label"]})

    def collate_fn(batch):
        images = torch.stack([transform(ex["image"].convert("RGB")) for ex in batch])
        labels = torch.tensor([ex["label"] for ex in batch])
        return images, labels  # tuple, not dict

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return test_loader