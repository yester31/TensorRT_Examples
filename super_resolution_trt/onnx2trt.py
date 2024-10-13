# by yhpark 2024-10-04
# TIMM ResNet18 ONNX model generation for dynamic input size
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

def get_engine(onnx_file_path, engine_file_path="", precision='fp32', dynamic_input_shapes=None):
    """Load or build a TensorRT engine based on the ONNX model."""
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(0) as network, \
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
                
            # The input text length is variable, so we need to specify an optimization profile.
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input = network.get_input(i)
                assert input.shape[0] == -1
                min_shape = dynamic_input_shapes[0]
                opt_shape = dynamic_input_shapes[1]
                max_shape = dynamic_input_shapes[2]
                profile.set_shape(input.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
            # for i in range(network.num_outputs):
            #     output = network.get_output(i)
            #     assert output.shape[0] == -1
            #     min_shape = dynamic_input_shapes[0]
            #     opt_shape = dynamic_input_shapes[1]
            #     max_shape = dynamic_input_shapes[2]
            #     profile.set_shape(output.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
            #     print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(output.name, min_shape, opt_shape, max_shape))
            config.add_optimization_profile(profile)
            
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

    print(f"[TRT] Engine file path: {engine_file_path}")

    if os.path.exists(engine_file_path):
        print(f"[TRT] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
    
def preprocess_image(img):
   """
   Function to preprocess the image.
   Includes color conversion, tensor transformation, and normalization.

   Parameters:
      img (np.ndarray): Input image in BGR format.

   Returns:
      np.ndarray: Preprocessed image tensor.
   """
   with torch.no_grad():
      # Convert BGR -> RGB
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # Convert HWC -> CHW and uint8 -> float32
      img = torch.from_numpy(img).permute(2, 0, 1).float()
      # Normalize to [0, 1]
      img /= 255.0
      # Add batch dimension (C, H, W) -> (1, C, H, W)
      img = img.unsqueeze(0)
      # Return as NumPy array (C-order)   
   return np.array(img, dtype=np.float32, order="C")

def main():
    iteration = 1000
    dur_time = 0

    # Input
    image_file_name = '0014.jpg'
    img_path = os.path.join(current_directory, 'Real-ESRGAN', 'inputs', image_file_name)
    img = cv2.imread(img_path)  # Load image
    input_image = preprocess_image(img)  # Preprocess image
    batch_images = np.concatenate([input_image], axis=0)
    print(batch_images.shape)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "RealESRGAN_x4plus"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    dynamic_input_shapes = [
                            [1,3,128,128],  # min
                            [1,3,256,256],  # opt
                            [1,3,512,512]   # max
                            ]
    
    # Output shapes expected(max shape size)
    output_shapes = (1, 3, input_image.shape[2]*4, input_image.shape[3]*4)
    print(output_shapes)

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
                
        #inspector = engine.create_engine_inspector()
        #inspector.execution_context = context # OPTIONAL
        #print(inspector.get_layer_information(0, trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
        #print(inspector.get_engine_information( trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the entire engine.
        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shapes, profile_idx=0)
        inputs[0].host = batch_images
        
        context.set_input_shape('input', batch_images.shape)
        
        # Warm-up        
        common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        # Inference loop
        for i in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        print(f'[TRT] {iteration} iterations time: {dur_time:.4f} [sec]')

    # # Reshape and post-process the output
    vol = 1
    for elem in output_shapes:
        vol *= elem
    new_outputs = trt_outputs[0][:vol]
    t_outputs = new_outputs.reshape(output_shapes)

    # # Results
    avg_time = dur_time / iteration
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')

    # postprocess (NCHW->NHWC, RGB->BGR, *255, ROUND, uint8)
    image = np.squeeze(np.array(t_outputs))
    image = np.clip(image, 0, 1)
    image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))
    image = (image * 255.0).round().astype(np.uint8)
    
    # 샘플 결과 출력 및 저장
    save_path = os.path.join(current_directory, 'save', f'{os.path.splitext(image_file_name)[0]}_{model_name}.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
