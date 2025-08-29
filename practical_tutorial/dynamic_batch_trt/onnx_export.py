# by yhpark 2024-10-04
# TIMM ResNet18 ONNX model generation for dynamic batch size
import torch
import torch.onnx
import onnx
import os
import timm

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


def main():
    
    model_name = "resnet18"
    export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}.onnx')

    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Load the pre-trained model
    model = timm.create_model(model_name=model_name, pretrained=True)
    model.eval()  # Set model to evaluation mode
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
            output_names=["output"], 
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allow variable batch size
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()