# by yhpark 2025-8-24
# TensorRT Model Optimization NAS example
import modelopt.torch.opt as mto
import modelopt.torch.nas as mtn

import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import os
import sys
import torchvision.transforms as transforms
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from base_model.utils_tmo import *
 
set_random_seed()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Print version information for debugging purposes
print(f"[TRT_E] PyTorch version: {torch.__version__}")
print(f"[TRT_E] ONNX version: {onnx.__version__}")

@torch.no_grad()
def evaluate(model, test_loader):
    """Evaluate the model on the given test_loader and return accuracy percentage."""
    model.eval()
    correct = total = 0.0
    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        output = model(images)
        predicted = output.argmax(dim=1)
        correct += torch.sum(labels == predicted).item()
        total += len(labels)

    accuracy = 100 * correct / total
    return accuracy

def main():

    num_classes = 100

    # Load the pre-trained model
    model_path = f"{CUR_DIR}/../base_model/checkpoint/b256_lr7.0e-04_we3_d0.3/best_model.pth"
    print(f"[TRT_E] load model ({model_path})")
    dropout, dropout_p = check_and_parse(model_path)
    model = load_model(num_classes, dropout_p).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint = remove_prefix(checkpoint, "_orig_mod.")
    model.load_state_dict(checkpoint)
    model.eval()

    # 1. Convert your model via mtn.convert
    # Generate the search space for AutoNAS
    nas_model = mtn.convert(model, mode="autonas") # mode="fastnas"

    # Save the search space for future use
    mto.save(nas_model, f"{CUR_DIR}/checkpoint/modelopt_autonas_model_init.pth")

    # A single 224x224 image for computing FLOPs
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    # Looking for a subnet with at most 2 GFLOPs
    constraints = 1.5e9

    is_sat, search_space_stats = mtn.profile(nas_model, dummy_input, constraints={"flops": constraints})

    top1_acc, top5_acc = test_model_topk(nas_model, test_loader, DEVICE, k=5)

    # 2. NAS training
    # Hyperparameter settings
    batch_size = 128
    model_name = "resnet18"
    num_epochs = 20
    warmup = True
    warmup_epochs = 5
    learning_rate = 5e-4
    best_val_acc = 0.0  # Criterion for saving the best model
    best_val_acc2 = 0.0  # Criterion for saving the best model
    suffix = f"tnas_e{num_epochs}_b{batch_size}_lr{learning_rate:.1e}"
    suffix = f"{suffix}_we{warmup_epochs}" if warmup else suffix
    checkpoint_dir_path = f"{CUR_DIR}/checkpoint/{suffix}"  # Directory to save graphs and models
    train_progress_dir_path = f"{CUR_DIR}/train_progress"  # Directory to save graphs and models
    # Create a folder to store results
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    if not os.path.exists(train_progress_dir_path):
        os.makedirs(train_progress_dir_path)

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

    train_loader = dataset_load(batch_size, transform_train, 'train')
    val_loader = dataset_load(batch_size, transform_val, 'validation')
    test_loader = dataset_load(batch_size, transform_val, 'test')
    test_loader2 = dataset_load2(batch_size, transform_val, 'test')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(nas_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if warmup:
        scheduler = WarmupCosineLR(optimizer, num_epochs, warmup_epochs, min_lr=1e-4)
    # Initialize lists to store training progress
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    print("[TRT_E] START Training loop")
    for epoch in range(num_epochs):
        if warmup:
            scheduler.step(epoch)
        # Training
        train_loss, train_acc = train_one_epoch(nas_model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = validate(nas_model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print training and validation info
        print(f"[TRT_E] Epoch [{epoch+1}/{num_epochs}]")
        print(f"[TRT_E] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"[TRT_E] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f"[TRT_E] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        best_val_acc = save_checkpoint(best_val_acc, nas_model, epoch, val_acc, checkpoint_dir_path)

        # Save the searched model 
        if val_acc > best_val_acc2:
            best_val_acc2 = val_acc
            mto.save(nas_model, f"{checkpoint_dir_path}/autonas_f{constraints}_best_model.pth")

        # Save the graph as a file
        save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path, suffix)

    print("Train complete.")

    checkpoint = torch.load(f'{checkpoint_dir_path}/best_model.pth', map_location=DEVICE)
    nas_model.load_state_dict(checkpoint)
    nas_model = nas_model.to(DEVICE)

    top1_acc, top5_acc = test_model_topk(nas_model, test_loader, DEVICE, k=5)

    # 3. Subnet architecture search via mtn.search
    # Wrap your original validation function to only take the model as input.
    # This function acts as the score function to rank models.
    def score_func(model):
        return evaluate(model, val_loader)

    # search_res (dict) contains state_dict / stats of the searcher
    searched_model, search_res = mtn.search(
        model=nas_model,
        constraints={"flops": constraints},# Looking for a subnet with at most 2 GFLOPs
        dummy_input=dummy_input,
        config={
            "data_loader": test_loader2,
            "score_func": score_func,
            "checkpoint": f"{checkpoint_dir_path}/autonas_f{constraints}_searched_model.pth",
        },
    )

    # evaluate the NAS model
    top1_acc, top5_acc = test_model_topk(searched_model, test_loader, DEVICE, k=5)

    # 4. Fine-tuning
    print("Fine-tune...")
    # Hyperparameter settings
    num_epochs = 10
    learning_rate = 3e-4
    best_val_acc = 0.0  # Criterion for saving the best model
    best_val_acc2 = 0.0  # Criterion for saving the best model
    suffix = f'fnas_e{num_epochs}_b{batch_size}_lr{learning_rate:.1e}'
    checkpoint_dir_path = f'{CUR_DIR}/checkpoint/{suffix}'  # Directory to save graphs and models
    train_progress_dir_path = f'{CUR_DIR}/train_progress'  # Directory to save graphs and models
    # Create a folder to store results
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    if not os.path.exists(train_progress_dir_path):
        os.makedirs(train_progress_dir_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(searched_model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize lists to store training progress
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    print("[TRT_E] START Training loop")
    for epoch in range(num_epochs):

        # Training
        train_loss, train_acc = train_one_epoch(searched_model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = validate(searched_model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print training and validation info
        print(f"[TRT_E] Epoch [{epoch+1}/{num_epochs}]")
        print(f"[TRT_E] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"[TRT_E] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f"[TRT_E] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        best_val_acc = save_checkpoint(best_val_acc, searched_model, epoch, val_acc, checkpoint_dir_path)

        # Save the searched model 
        if val_acc > best_val_acc2:
            best_val_acc2 = val_acc
            mto.save(searched_model, f"{checkpoint_dir_path}/autonas_f{constraints}_best_model.pth")

        # Save the graph as a file
        save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path, suffix)

    print("Fine-tune complete.")

    checkpoint = torch.load(f'{checkpoint_dir_path}/best_model.pth', map_location=DEVICE)
    searched_model.load_state_dict(checkpoint)
    searched_model = searched_model.to(DEVICE)

    top1_acc, top5_acc = test_model_topk(searched_model, test_loader, DEVICE, k=5)

    # export nas model 
    print("export nas model")
    searched_model.eval()
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_nas.onnx')
    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)

    # Get model input size from the model configuration
    dummy_input = torch.randn((1, 3, 224, 224), requires_grad=False).to(DEVICE)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            searched_model, 
            dummy_input, 
            export_model_path, 
            opset_version=19, 
            input_names=["input"], 
            output_names=["output"]
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()