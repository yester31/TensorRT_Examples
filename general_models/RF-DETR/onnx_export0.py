# by yhpark 2025-9-27
import onnx
import os 
import sys
import torch 

sys.path.insert(1, os.path.join(sys.path[0], "rf-detr"))
from rfdetr import RFDETRBase, RFDETRNano

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    model = RFDETRNano()
    model = model.export(output_dir=save_path, infer_dir=None, simplify=True,  backbone_only=False, opset_version=20)



if __name__ == '__main__':
    main()