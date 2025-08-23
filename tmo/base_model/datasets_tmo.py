from datasets import load_dataset  
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 
import cv2
import numpy as np
import os 
import json

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def dataset_load(batch_size, transform, data_type):
    shuffle = False
    if data_type == 'train':
        shuffle = True

    # Load the dataset test  
    dataset = load_dataset("ilee0022/ImageNet100", cache_dir=f"{CUR_DIR}/dataset")

    test_dataset = dataset[data_type].with_transform(lambda ex: {"image": ex["image"], "label": ex["label"]})

    def collate_fn(batch):
        images = torch.stack([transform(ex["image"].convert("RGB")) for ex in batch])
        labels = torch.tensor([ex["label"] for ex in batch])
        return {"image": images, "label": labels}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count()-2, pin_memory=True, collate_fn=collate_fn)

    return test_loader


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    batch_size = 64
    train_loader = dataset_load(batch_size, transform, 'train')
    val_loader = dataset_load(batch_size, transform, 'validation')

    batch = next(iter(train_loader))
    print(batch["image"].shape)  # torch.Size([64, 3, 224, 224])
    print(batch["label"][:5])

    test_loader = dataset_load(batch_size, transform, 'test')

    batch = next(iter(test_loader))
    print(batch["image"].shape)  # torch.Size([64, 3, 224, 224])
    print(batch["label"][:5])

    # Load the dataset  
    dataset = load_dataset("ilee0022/ImageNet100", cache_dir=f"{CUR_DIR}/dataset")

    print(dataset)              
    print(dataset["train"][10])  

    # Access a sample from the dataset  
    image = dataset["train"][11]["image"]
    label = dataset["train"][11]["label"]
    text = dataset["train"][11]["text"]
    print(f"Label: {label}")
    print(f"text: {text}")
    
    with open(f"{CUR_DIR}/dataset/label2text.json", "r", encoding="utf-8") as f:
        label2text = json.load(f)
    print(label2text[str(label)])

    # PIL → numpy, RGB → BGR
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print(img_cv2.shape)

    cv2.imshow("sample", img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_dir_path = f'{CUR_DIR}/test'
    if not os.path.exists(image_dir_path):
        os.makedirs(image_dir_path)
    cv2.imwrite(f"{image_dir_path}/output2.png", img_cv2)