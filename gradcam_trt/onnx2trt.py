# by yhpark 2024-10-04
# TIMM ResNet18 GradCam TensorRT example
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import cv2
import os
import numpy as np
import time
import common
from common import *
from utils import *

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Global Variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

def get_engine(onnx_file_path, engine_file_path="", precision='fp32'):
    """Load or build a TensorRT engine based on the ONNX model."""
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"[TRT_E] ONNX file {onnx_file_path} not found.")

            print(f"[TRT_E] Loading and parsing ONNX file: {onnx_file_path}")
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    raise RuntimeError("[TRT_E] Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

            for i_idx in range(network.num_inputs):
                print(f'[TRT_E] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[TRT_E] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
                
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    print(f"[TRT_E] Engine file path: {engine_file_path}")

    if os.path.exists(engine_file_path):
        print(f"[TRT_E] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    iteration = 1000
    dur_time = 0

    # Input
    img_path = os.path.join(current_directory, 'data', 'panda0.jpg')
    origin_img = cv2.imread(img_path)  # Load image
    input_image = preprocess_image(origin_img)  # Preprocess image
    
    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "resnet18"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_modified.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}_modified.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    
    # Output shapes expected
    output_shapes = [(1, 1000), (1, 512, 7, 7)]
    
    # load fc weight
    weights_fc_w_path = os.path.join(current_directory, 'onnx', f'{model_name}_fc_weights.bin') 
    weights_fc_w = np.fromfile(weights_fc_w_path, dtype=np.float32).reshape((1000, 512))

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = input_image
        
        # Warm-up        
        common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        # Inference loop
        for i in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        print(f'[TRT_E] {iteration} iterations time: {dur_time:.4f} [sec]')
        
    
    # Reshape and post-process the output
    t_outputs1 = trt_outputs[0].reshape(output_shapes[0])
    t_outputs2 = trt_outputs[1].reshape(output_shapes[1])
    
    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    max_tensor = torch.from_numpy(t_outputs1).max(dim=1)
    max_value = max_tensor[0].cpu().numpy()[0]
    max_index = max_tensor[1].cpu().numpy()[0]
    print(f'[TRT_E] Resnet18 max index: {max_index}, value: {max_value}, class name: {class_name[max_index]}')
    
    # calculate grad cam
    activations = torch.from_numpy(t_outputs2)
    target_weights = torch.from_numpy(weights_fc_w[max_index])  # [512] <- [1000, 512]
    grad_cam = torch.zeros(activations.shape[2:])  # [7, 7]
    for i, weight in enumerate(target_weights):  # [512]
        grad_cam += weight * activations[0, i, :, :]  # [7, 7] += [1] * [7, 7]

    grad_cam = grad_cam.data.numpy()
    grad_cam = np.maximum(grad_cam, 0)  # relu
    grad_cam = grad_cam / (1e-7 + np.max(grad_cam))

    # scale-up
    grad_cam_224 = cv2.resize(grad_cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_224), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    bgr_img = np.float32(origin_img) / 255
    mix_cam_ori = 0.5 * heatmap + 0.5 * bgr_img
    mix_cam_ori = mix_cam_ori / (1e-7 + np.max(mix_cam_ori))
    mix_cam_ori = np.uint8(255 * mix_cam_ori)
    
    f_name = os.path.basename(img_path).split(".")[0]
    mix_results_path = os.path.join(current_directory, 'results', f'{f_name}_mix_cam_ori.jpg')
    heatmap_results_path = os.path.join(current_directory, 'results', f'{f_name}_heatmap.jpg')
    os.makedirs(os.path.dirname(mix_results_path), exist_ok=True)
    cv2.imwrite(mix_results_path, mix_cam_ori)
    cv2.imwrite(heatmap_results_path, np.uint8(255 * heatmap))
    
    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Grad Cam Example succeeded!")


if __name__ == '__main__':
    main()
