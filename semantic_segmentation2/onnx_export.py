# by yhpark 2025-8-26
import torch
import onnx
import os
from BEN2 import BEN2

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Print version information for debugging purposes
print(f"[TRT_E] PyTorch version: {torch.__version__}")
print(f"[TRT_E] ONNX version: {onnx.__version__}")

def main():

    input_size = 512
    model_name = "BEN2_Base"
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_{input_size}.onnx')
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Set up model
    model_path = f"{CUR_DIR}/BEN2/checkpoint/BEN2_Base.pth"
    print(f"[TRT_E] load model ({model_path})")
    model = BEN2.BEN_Base().to(DEVICE).eval() #init pipeline
    model.loadcheckpoints(model_path)

    # Get model input size from the model configuration
    dummy_input = torch.randn((1, 3, input_size, input_size), requires_grad=False).to(DEVICE)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=19, 
            input_names=["input"], 
            output_names=["output"],
            dynamo=True
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()