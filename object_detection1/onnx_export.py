# by yhpark 2024-10-16
# Object Detection Example with TensorRT Model Optimization
from ultralytics import YOLO
import torch.onnx
import onnx
import math
import os 

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

    # 0. prepare weight & onnx file name & image size 
    batch_size = 1
    model_name = "yolo11l.pt"
    export_model_path = os.path.join(current_directory, 'onnx', f'{os.path.splitext(model_name)[0]}_{device.type}.onnx')
    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # 1. input size
    dummy_input = torch.randn(batch_size, 3, 640, 640, requires_grad=True)

    # 2. load model
    model = YOLO(model_name).model    # yolov11 model load
    #model.fuse()
    model = model.eval()              # set evaluation mode

    # 3. generate onnx file
    with torch.no_grad():
        torch.onnx.export(model,                   # pytorch model
                        dummy_input,               # model dummy input
                        export_model_path,         # onnx model path
                        opset_version=19,          # the version of the opset
                        input_names=['input'],     # input name
                        output_names=['output'],   # output name
                        # dynamic_axes={
                        #     "input": {0: "batch_size", 2: "width", 3: "height"},
                        #     "output0": {0: "batch_size", 2: "anchors"}
                        #             }
                        )  

        print("ONNX Model exported at ", export_model_path)

    # 4. check the produced onnx file
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model check done!")

if __name__ == '__main__':
    main()