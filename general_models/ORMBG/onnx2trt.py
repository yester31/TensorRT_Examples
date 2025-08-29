# by yhpark 2025-8-26
# Semantic Segmentation (BEN2)
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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_engine(onnx_file_path, engine_file_path="", precision='fp32', TRT_LOGGER = trt.Logger(trt.Logger.INFO)):
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
            parser.parse_from_file(onnx_file_path)

            filename = os.path.splitext(os.path.basename(engine_file_path))[0]
            timing_cache = f"{CUR_DIR}/engine/{filename}timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(3))
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
        print(f'[MDET] Build engine ({engine_file_path})')
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        print(f'[MDET] Engine build done! ({show_build_time(build_time)})')  
        return engine

def show_build_time(build_time):      
    if build_time < 60:
        build_time_str = f"{build_time:.2f} sec"
    elif build_time < 3600:
        minutes = int(build_time // 60)
        seconds = build_time % 60
        build_time_str = f"{minutes} min {seconds:.2f} sec"
    else:
        hours = int(build_time // 3600)
        minutes = int((build_time % 3600) // 60)
        seconds = build_time % 60
        build_time_str = f"{hours} hr {minutes} min {seconds:.2f} sec"
    return build_time_str

def transform_cv(image_, size):   
    image = image_.copy() 
    # 1) BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2) Resize (1024)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    # 4) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    tensor = torch.from_numpy(image)

    # Add batch dimension (C, H, W) -> (1, C, H, W)
    tensor = np.expand_dims(tensor, axis=0)
    # Return as NumPy array (C-order)   
    return np.array(tensor, dtype=np.float32, order="C")

def get_inference_fps(batch_size, context, engine, bindings, inputs, outputs, stream):
    # CUDA Events
    err, start_event = cudart.cudaEventCreate()
    err, end_event = cudart.cudaEventCreate()

    # warmup
    for _ in range(10):
        trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  

    # Measure
    iterations = 100
    cudart.cudaEventRecord(start_event, 0)
    for _ in range(iterations):
        trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)            
    cudart.cudaEventRecord(end_event, 0)
    cudart.cudaEventSynchronize(end_event)
    err, elapsed_time_ms = cudart.cudaEventElapsedTime(start_event, end_event)
    elapsed_time_sec = elapsed_time_ms / 1000.0
    fps = (iterations * batch_size) / (elapsed_time_sec)
    avg_time = elapsed_time_ms / (iterations * batch_size)
    # Results
    print(f'[TRT_E] {iterations} iterations time: {elapsed_time_sec:.4f} [sec]')
    print(f'[TRT_E] Average FPS: {fps:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time:.2f} [msec]')

def main():

    # Input
    img_path = f'{CUR_DIR}/data/pexels-photo-5965592.png'
    image = cv2.imread(img_path)  # Load image
    h, w, c = image.shape 
    input_size = 512
    input_image = transform_cv(image, input_size)  # Preprocess image (1,3,1024,1024)
    batch_size = 1

    # Model and engine paths
    model_name = "ormbg"
    precision = "fp16" # Choose 'fp32' or 'fp16'
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_{input_size}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{input_size}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes expected
    output_shapes = [(1, input_size, input_size)] 

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        get_inference_fps(batch_size, context, engine, bindings, inputs, outputs, stream)

        # run test image 
        inputs[0].host = input_image
        trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)            

    # Reshape and post-process the output
    t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    # postprocess (CHW->HW, resize, *255, uint8)
    t_result = np.squeeze(t_outputs[0])
    t_result = cv2.resize(t_result, (w, h))
    min_value = np.min(t_result)
    max_value = np.max(t_result)
    t_result = (t_result - min_value) / (max_value - min_value)
    mask = (t_result * 255.0).astype(np.uint8)

    foreground = cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(CUR_DIR, 'save', f'{filename}_{input_size}_fg_trt.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, foreground)
    save_path = os.path.join(CUR_DIR, 'save', f'{filename}_{input_size}_mask_trt.png')
    cv2.imwrite(save_path, mask)

    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
