# by yhpark 2024-10-15
# by yhpark 2025-8-23
# ONNX PTQ example
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
import torchvision.transforms as transforms
import tensorrt as trt
import torch
import cv2
import common
from common import *
import json

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from base_model.utils_tmo import *
from base_trt.onnx2trt import get_engine, transform_cv, test_model_topk_trt, get_inference_fps

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def main():

    # Input
    img_path = f'{CUR_DIR}/../base_model/test/test_11.png'
    image = cv2.imread(img_path)  # Load image
    input_image = transform_cv(image)  # Preprocess image
    with open(f"{CUR_DIR}/../base_model/dataset/label2text.json", "r", encoding="utf-8") as f:
        label2text = json.load(f)

    batch_size = 1
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    test_loader = dataset_load(batch_size, transform_test, 'test')

    # Model and engine paths
    model_name = "resnet18"
    precision = "int8" # Choose 'int8' or 'fp32' or 'fp16'
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_optq.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}_optq.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes expected
    output_shapes = [(1, 100)]

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        top1_acc, top5_acc = test_model_topk_trt(test_loader, output_shapes, context, engine, inputs, outputs, bindings, stream)
        get_inference_fps(batch_size, context, engine, bindings, inputs, outputs, stream)

        # run test image 
        inputs[0].host = input_image
        trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) 

        # Reshape and post-process the output
        t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        t_outputs = torch.from_numpy(t_outputs[0])

        # Results
        max_tensor = t_outputs.max(dim=1)
        max_value = max_tensor[0].cpu().numpy()[0]
        max_index = max_tensor[1].cpu().numpy()[0]
        print(f'[TRT_E] max value: {max_value}')
        print(f'[TRT_E] max index: {max_index}')
        print(f'[TRT_E] max label: {label2text[str(max_index)]}')

    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
