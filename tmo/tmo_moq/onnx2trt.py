# by yhpark 2024-10-17
# ONNX PTQ example
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "../.."))

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

timing_cache = f"{current_directory}/timing.cache"

def get_engine(onnx_file_path, engine_file_path="", precision='fp32'):
    """Load or build a TensorRT engine based on the ONNX model."""
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
                    
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"[TRT] ONNX file {onnx_file_path} not found.")

            print(f"[TRT] Loading and parsing ONNX file: {onnx_file_path}")
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    raise RuntimeError("[TRT] Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            
            if precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 mode.")
            elif precision == "int8" :
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                if builder.platform_has_fast_int8 :
                    config.set_flag(trt.BuilderFlag.INT8)
                print("Using INT8 mode.")
            elif precision == "fp32":
                print("Using FP32 mode.")
            else:
                raise NotImplementedError(
                    f"Currently hasn't been implemented: {precision}."
                )

            for i_idx in range(network.num_inputs):
                print(f'[TRT_E] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[TRT_E] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
                
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            common.save_timing_cache(config, timing_cache)
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


def preprocess_image2(image):
    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    # resize 256x256
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    # center crop 224x224 
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top + 224, left:left + 224]
    # HWC -> CHW, [0, 1]
    image_tensor = image_cropped.transpose(2, 0, 1) / 255.0
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image_normalized = (image_tensor - mean) / std
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    image_normalized = np.expand_dims(image_normalized, axis=0)
    # Return as NumPy array (C-order)   
    return np.array(image_normalized, dtype=np.float32, order="C")


def main():
    iteration = 1000
    dur_time = 0

    # Input
    img_path = f'{current_directory}/../datasets/imagenet100/val/n01824575/ILSVRC2012_val_00002263.JPEG'
    img = cv2.imread(img_path)  # Load image
    input_image = preprocess_image2(img)  # Preprocess image
    
    Labels_path = f'{current_directory}/../datasets/imagenet100/Labels.json'
    Labels = read_json(Labels_path)
    Labels_path2 = f'{current_directory}/../datasets/imagenet100/class_to_idx.json'
    Labels2 = read_json(Labels_path2)

    # Model and engine paths
    ptq_onnx = True # whether this onnx from onnx_quantization.py
    model_name = "resnet18"
    if ptq_onnx:
        precision = "int8" 
        onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_moq.onnx') # batch folding before ptq
        engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}_moq.engine')
        # Average FPS: 6737.40 [fps]
    else :
        precision = "fp16" # Choose 'fp32' or 'fp16'
        onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_bf.onnx')
        engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}_bf.engine')
        # Average FPS: 4886.81 [fps] <- f16 w batch folding
        # Average FPS: 4887.49 [fps] <- f16 wo batch folding
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes expected
    output_shapes = [(1, 100)]

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = input_image
        
        # Warm-up       
        for i in range(50):
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        # Inference loop
        for i in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        print(f'[TRT_E] {iteration} iterations time: {dur_time:.4f} [sec]')
        
    
    # Reshape and post-process the output
    t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] precision: {precision}')
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
    max_value = max_tensor[0].cpu().numpy()[0]
    max_index = max_tensor[1].cpu().numpy()[0]
    
    matching_keys = [key for key, value in Labels2.items() if value == max_index]
    Labels[matching_keys[0]]
    print(f'[TRT_E] Resnet18 max index: {max_index}, value: {max_value}, class name: {matching_keys[0]}, {Labels[matching_keys[0]]}')
    print(f'[TRT_E] image path : {img_path}')
    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
