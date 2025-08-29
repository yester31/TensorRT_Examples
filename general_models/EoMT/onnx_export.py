# by yhpark 2025-8-29
import torch
import onnx
import os
from transformers import EomtForUniversalSegmentation

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Print version information for debugging purposes
print(f"[TRT_E] PyTorch version: {torch.__version__}")
print(f"[TRT_E] ONNX version: {onnx.__version__}")

def main():
    model_name = "coco_panoptic_eomt_large_640"
    model_name = "coco_panoptic_eomt_small_640_2x"
    model_name = "coco_panoptic_eomt_base_640_2x"
    model_id = f"tue-mps/{model_name}"
    input_size = 640
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Set up model
    print(f"[TRT_E] load model ({model_id})")
    model = EomtForUniversalSegmentation.from_pretrained(model_id).to(DEVICE).eval()

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
            output_names=['class_queries_logits', 'masks_queries_logits', 'last_hidden_state'],
            # dynamo=True
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()