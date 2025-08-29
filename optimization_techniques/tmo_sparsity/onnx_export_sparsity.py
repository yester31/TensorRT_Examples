# by yhpark 2024-10-17
# by yhpark 2025-8-23
# Sparsity model example
import modelopt.torch.sparsity as mts
import modelopt.torch.opt as mto

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

def main():
    
    batch_size = 256
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    calib_dataloader = dataset_load(batch_size, transform_test, 'test')
    
    # Load the pre-trained model
    model_path = f'{CUR_DIR}/../base_model/checkpoint/b256_lr7.0e-04_we3_d0.3/best_model.pth'
    print(f"[TRT_E] load model ({model_path})")
    num_classes = 100
    dropout, dropout_p = check_and_parse(model_path)
    model = load_model(num_classes, dropout_p).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint = remove_prefix(checkpoint, "_orig_mod.")
    model.load_state_dict(checkpoint)
    model = model.eval()

    # Hyperparameter settings
    sparsity_mode = "sparse_magnitude" # "sparsegpt" or "sparse_magnitude"
    # sparsegpt": data_loader is only required in case of data-driven sparsity
    # sparse_magnitude: sparse_magnitude does not require data_loader as it uses magnitude-based method for thresholding.
    model_name = "resnet18"
    num_epochs = 5
    learning_rate = 1e-4
    best_val_acc = 0.0  # Criterion for saving the best model
    suffix = f'{sparsity_mode}_e{num_epochs}_b{batch_size}_lr{learning_rate:.1e}'
    checkpoint_dir_path = f'{CUR_DIR}/checkpoint/{suffix}'  # Directory to save graphs and models
    train_progress_dir_path = f'{CUR_DIR}/train_progress'  # Directory to save graphs and models
    # Create a folder to store results
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    if not os.path.exists(train_progress_dir_path):
        os.makedirs(train_progress_dir_path)
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_{sparsity_mode}.onnx')
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)

    # Sparsify the model
    print("Starting sparsifying...")
    # Configure and convert for sparsity
    sparsity_config = {
        # data_loader is required for sparsity calibration
        "data_loader": calib_dataloader,
        "collect_func": lambda x: x["image"],
    }

    sparse_model = mts.sparsify(
        model,
        sparsity_mode,
        config=sparsity_config,
    )

    # Save the sparsity modelopt state
    modelopt_state_path = f"{checkpoint_dir_path}/pts_modelopt_state.pth"
    mto.save(sparse_model, modelopt_state_path)
        
    print("Fine-tune...")

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(sparse_model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize lists to store training progress
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    print("[TRT_E] START Training loop")
    for epoch in range(num_epochs):

        # Training
        train_loss, train_acc = train_one_epoch(sparse_model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = validate(sparse_model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print training and validation info
        print(f"[TRT_E] Epoch [{epoch+1}/{num_epochs}]")
        print(f"[TRT_E] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"[TRT_E] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f"[TRT_E] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        best_val_acc = save_checkpoint(best_val_acc, sparse_model, epoch, val_acc, checkpoint_dir_path)

        # Save the graph as a file
        save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path, suffix)

    print("Fine-tune complete.")
    print("load best model (sparsity)")
    model_path = f'{checkpoint_dir_path}/best_model.pth'
    checkpoint = torch.load(model_path, map_location=DEVICE)
    sparse_model.load_state_dict(checkpoint)
    sparse_model = sparse_model.eval()

    top1_acc, top5_acc = test_model_topk(sparse_model, test_loader, DEVICE, k=5)

    modelopt_state = mto.modelopt_state(sparse_model)
    torch.save({"modelopt_state": modelopt_state, "model_state_dict": sparse_model.state_dict()}, f"{checkpoint_dir_path}/sprase_model_with_state.pth",)

    # Get model input size from the model configuration
    dummy_input = torch.randn((1, 3, 224, 224), requires_grad=False).to(DEVICE)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            sparse_model, 
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


if __name__ == '__main__':
    main()