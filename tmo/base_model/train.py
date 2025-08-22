# by yhpark 2024-10-15
# by yhpark 2025-8-21
# TensorRT Model Optimization PTQ example
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import wandb
from utils import *
from dataset import *
torch._inductor.config.max_autotune_gemm = False

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('[TRT_E] gpu device count : ', torch.cuda.device_count())
    print('[TRT_E] device_name : ', torch.cuda.get_device_name(0))
    
print(f"[TRT_E] using device: {DEVICE}")
print(f"[TRT_E] cpu_count: {os.cpu_count()}")
print(f"[TRT_E] multiprocessing cpu_count: {multiprocessing.cpu_count()}")
print("[TRT_E] torch version: ", torch.__version__)

# Main training loop
def main():
    set_random_seed()

    # Hyperparameter settings
    num_classes = 100
    batch_size = 256
    num_epochs = 10
    warmup = True
    warmup_epochs = 3
    dropout = True
    dropout_p = 0.3
    learning_rate = 7e-4
    best_val_acc = 0.0  # Criterion for saving the best model
    suffix = f'b{batch_size}_lr{learning_rate:.1e}'
    suffix = f"{suffix}_we{warmup_epochs}" if warmup else suffix
    suffix = f"{suffix}_d{dropout_p}" if dropout else suffix
    checkpoint_dir_path = f'{CUR_DIR}/checkpoint/{suffix}'  # Directory to save graphs and models
    train_progress_dir_path = f'{CUR_DIR}/train_progress'  # Directory to save graphs and models
    # Create a folder to store results
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    if not os.path.exists(train_progress_dir_path):
        os.makedirs(train_progress_dir_path)
        
    wandb_flag = True
    if wandb_flag :
        wandb_log = wandb.init(
            entity="yester31_me",
            project="my-awesome-project",
            dir=f"{CUR_DIR}/",
            config={
                "learning_rate": learning_rate,
                "architecture": "resnet18",
                "dataset": "Imagenet-100",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "warmup_epochs": warmup_epochs,
                "dropout_p": dropout_p,
            },
        )

    # Define data preprocessing (for ImageNet100)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),        
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    # Set up model, data, loss function, and optimizer
    print("[TRT_E] SET model, data, loss function, and optimizer")
    train_loader = dataset_load(batch_size, transform_train, 'train')
    val_loader = dataset_load(batch_size, transform_val, 'validation')
    test_loader = dataset_load(batch_size, transform_val, 'test')
    if dropout:
        model = load_model(num_classes, dropout_p).to(DEVICE)
    else:
        model = load_model(num_classes).to(DEVICE)
    # model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if warmup:
        scheduler = WarmupCosineLR(optimizer, num_epochs, warmup_epochs, min_lr=1e-5)

    # Initialize lists to store training progress
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    print("[TRT_E] START Training loop")
    for epoch in range(num_epochs):
        if warmup:
            scheduler.step(epoch)

        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print training and validation info
        print(f"[TRT_E] Epoch [{epoch+1}/{num_epochs}]")
        print(f"[TRT_E] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"[TRT_E] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f"[TRT_E] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if wandb_flag :
            wandb_log.log({
                "train_acc": train_acc, 
                "val_acc": val_acc, 
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": train_loss, 
                "val_loss": val_loss,
                })
     
        # Save the best model
        best_val_acc = save_checkpoint(best_val_acc, model, epoch, val_acc, checkpoint_dir_path)

        # Save the graph as a file
        save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path, suffix)

    print("[TRT_E] Training complete.")

    top1_acc, top5_acc = test_model_topk(model, test_loader, DEVICE, k=5)

    if wandb_flag :
        wandb_log.log({
            "top1_acc": top1_acc, 
            "top5_acc": top5_acc,})
        wandb_log.finish()

# Execute the main function
if __name__ == "__main__":
    main()
