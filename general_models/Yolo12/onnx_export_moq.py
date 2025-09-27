# by yhpark 2025-9-3
# ONNX PTQ example
import modelopt.onnx.quantization as moq

import torch
import os
import numpy as np 
import torchvision.transforms as transforms
from calib_data import dataset_load
from onnx_export import * 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def main():

    input_h, input_w = 640, 640 
    calibration_data_path = f"{CUR_DIR}/calib_datas/calib_data_{input_h}x{input_w}.npy"
    os.makedirs(os.path.dirname(calibration_data_path), exist_ok=True)

    if os.path.exists(calibration_data_path) is False:
        batch_size = 512  
        print(f"generate calib data ({batch_size}, 3, {input_h}, {input_w})")
        transform_test = transforms.Compose([
            transforms.Resize((input_h, input_w)),             
            transforms.ToTensor(),
        ])

        # load dataset 
        test_loader = dataset_load(batch_size, transform_test, 'test')
        images, labels = next(iter(test_loader))
        calib_data = np.array(images, dtype=np.float32)
        np.save(calibration_data_path, calib_data)

    calib_data = np.load(calibration_data_path)

    model_name = "yolov12n-face"
    onnx_sim = True # True or False
    model_name = f"{model_name}_{input_h}x{input_w}"
    input_onnx_path = f"{CUR_DIR}/onnx/{model_name}.onnx"
    model_name = f"{model_name}_moq"
    export_model_path = f"{CUR_DIR}/onnx/{model_name}.onnx"
    moq.quantize(
        onnx_path=input_onnx_path,
        quantize_mode="int8",
        calibration_data=calib_data,
        calibration_method="entropy",   # max, entropy, awq_clip, rtn_dq etc.
        output_path=export_model_path,
    )

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)
    if onnx_sim :
        export_model_sim_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

    max_output_boxes = 300
    iou_threshold = 0.7
    score_threshold = 0.01
    yolo_insert_nms(
        path=export_model_sim_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_output_boxes=max_output_boxes,
    )

if __name__ == '__main__':
    main()