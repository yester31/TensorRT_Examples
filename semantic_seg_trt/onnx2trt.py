import os
import sys
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
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

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

def preproc_xray(img):
   """
   Function to preprocess the image.
   Includes tensor transformation, and normalization.

   Parameters:
      img (np.ndarray): Input image in 1 chanel format.

   Returns:
      torch.tensor : Preprocessed image tensor.
   """
   with torch.no_grad():
      # Convert uint8 -> float32
      img = torch.from_numpy(img).float()
      # Normalize to [0, 1]
      img /= 255.0
      # Add batch dimension (H, W) -> (1, C, H, W)
      img = img.unsqueeze(0).unsqueeze(0)
   return np.array(img, dtype=np.float32, order="C")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    iteration = 1000
    dur_time = 0

    # Input
    seg_model_input_size = 256
    image_file_name = '0f45742c4d100eeee221f8853d79c9d4.png'
    image_path = os.path.join(current_directory, 'samples', "Pneumothorax", 'TRUE', image_file_name)
    origin_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if origin_img.shape != (seg_model_input_size, seg_model_input_size):
        origin_img = cv2.resize(origin_img, dsize=(seg_model_input_size, seg_model_input_size))
        
    input_image = preproc_xray(origin_img)  # Preprocess image
    
    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "UnetPlusPlus_timm-efficientnet-b3"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes expected
    output_shapes = [(1,256, 256)]

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
    t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    
    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    threshold = 0.3
    t_outputs = np.squeeze(np.array(t_outputs))
    # 기준치 이상인 값만 255로, 나머지는 0으로 설정
    pr_masks = np.where(sigmoid(t_outputs) >= threshold, 255, 0)

    save_path_mask = os.path.join(current_directory, 'save', f'{image_file_name}_mask.jpg')
    os.makedirs(os.path.dirname(save_path_mask), exist_ok=True)
    cv2.imwrite(save_path_mask, pr_masks)

    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
