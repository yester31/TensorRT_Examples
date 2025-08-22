import torch
import torch.nn as nn
import random
import timm
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import os
import re
from collections import OrderedDict

def set_random_seed(random_seed = 42):
   print(f"[TRT_E] set random seed: {random_seed}")
   torch.manual_seed(random_seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(random_seed)
   random.seed(random_seed)

# Function to load the model
def load_model(num_classes, dropout_p=None):
    model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    if dropout_p is not None:
        model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),         # 30% 
        nn.Linear(model.fc.in_features, num_classes)  # num_classes
    )
    return model

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Model prediction and loss calculation
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backpropagation and optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # 배치 단위 진행도 출력
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {running_loss/(batch_idx+1):.4f}, Acc: {correct_train / total_train:.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val
    return val_loss, val_acc

# Function to save the plot as a file
def save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path, suffix=''):
    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # Save the graph as a file
    plot_path = os.path.join(train_progress_dir_path, f'loss_acc_{suffix}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

# Function to save the best model
def save_checkpoint(best_val_acc, model, epoch, val_acc, checkpoint_dir_path, suffix = ''):
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = os.path.join(checkpoint_dir_path, f'best_model{suffix}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")
    return best_val_acc

def test_model_topk(model, test_loader, device, k=5):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)

            total += labels.size(0)
            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1,1).expand_as(pred)).sum().item()

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    print(f"Test Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Test Top-{k} Accuracy: {top5_acc*100:.2f}%")
    return top1_acc, top5_acc

def test_model_topk_fps(model, test_loader, device, k=5, use_half=False):
    model.eval()
    if use_half:
        model.half()  # Convert model to FP16
    model = torch.compile(model)

    top1_correct = 0
    top5_correct = 0
    total = 0

    # Warm-up
    batch = next(iter(test_loader))
    dummy_input = torch.randn((batch["image"].shape), requires_grad=False).to(device)  # Create a dummy input
    if use_half:
        dummy_input = dummy_input.half()  # FP16 
    for _ in range(10):
        outputs = model(dummy_input)
    torch.cuda.synchronize()

    # FPS
    elapsed = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_half:
                images = images.half()  # FP16 

            # Forward pass
            begin = time.perf_counter()
            outputs = model(images)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - begin
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)

            total += labels.size(0)
            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1,1).expand_as(pred)).sum().item()

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    fps = total / elapsed

    print(f"[TRT_E] Test Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"[TRT_E] Test Top-{k} Accuracy: {top5_acc*100:.2f}%")
    print(f"[TRT_E] Inference FPS: {fps:.2f} samples/sec (use_half={use_half})")

    return top1_acc, top5_acc, fps

# Scheduler: Warmup + CosineAnnealing
class WarmupCosineLR:
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr=1e-5):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.cur_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.cur_lr = lr

    def get_last_lr(self):
        return [self.cur_lr]

def check_and_parse(path: str):
    # Normalize and split path
    parts = os.path.normpath(path).split(os.sep)
    
    if len(parts) < 2:
        return None, None  # no second last folder
    
    folder = parts[-2]
    
    # Check for 'd' and extract number
    match = re.search(r"d(\d+(?:\.\d+)?)", folder)
    if match:
        return True, float(match.group(1))
    else:
        return False, None

def remove_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict