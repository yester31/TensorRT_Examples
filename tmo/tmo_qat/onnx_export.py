# by yhpark 2024-10-16
# TensorRT Model Optimization QAT example
import torch
import torch.onnx
import torch.nn as nn
import onnx
import os
import timm
import numpy as np 
import copy

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2

        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
        return conv
    else:
        return False

def fuse_bn_recursively(model):
    previous_name = None

    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name  # Initialization

        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()

        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

        previous_name = module_name

    return model

def main():
        
    class_count = 100
    model_name = "resnet18"
    export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_bf.onnx')
    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Load the pre-trained model
    model = timm.create_model(model_name=model_name, num_classes=class_count, pretrained=True).to(device)
    model_path = f'{current_directory}/../base_model/checkpoint/best_model.pth'
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode
    
    # batch folding(conv + bn -> fused conv)
    model = fuse_bn_recursively(model) 
    
    model.to(device)  # Move the model to the chosen device

    # Get model input size from the model configuration
    input_size = model.pretrained_cfg["input_size"]
    dummy_input = torch.randn(input_size, requires_grad=False).unsqueeze(0).to(device)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
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