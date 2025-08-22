# by yhpark 2024-10-15
# TensorRT Model Optimization PTQ example
import modelopt.torch.quantization as mtq

import torch
import torch.onnx
import onnx
import os
import sys
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base_model")))
import utils
import dataset

utils.set_random_seed()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Print version information for debugging purposes
print(f"[TRT_E] PyTorch version: {torch.__version__}")
print(f"[TRT_E] ONNX version: {onnx.__version__}")

def main():
    
    # Quantization need calibration data. Setup calibration data loader
    batch_size = 256
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),        
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    train_loader = dataset.dataset_load(batch_size, transform_train, 'test')
    
    num_classes = 100
    model_name = "resnet18"
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_ptq.onnx')
    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Load the pre-trained model
    model_path = f'{CUR_DIR}/../base_model/checkpoint/b256_lr7.0e-04_we3_d0.3/best_model.pth'
    print(f"[TRT_E] load model ({model_path})")
    dropout, dropout_p = utils.check_and_parse(model_path)
    model = utils.load_model(num_classes, dropout_p).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint = utils.remove_prefix(checkpoint, "_orig_mod.")
    model.load_state_dict(checkpoint)
    model = model.eval()

    # Select quantization config
    config = mtq.INT8_DEFAULT_CFG # for CNN models

    # Define forward_loop. Please wrap the data loader in the forward_loop
    def forward_loop(model):
        for i, batch in enumerate(train_loader):
            model(batch["image"].to(DEVICE))

    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, forward_loop)
    
    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(model)
    
    # Get model input size from the model configuration
    dummy_input = torch.randn((1, 3, 224, 224), requires_grad=False).to(DEVICE)  # Create a dummy input

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