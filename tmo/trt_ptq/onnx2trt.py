# by yhpark 2025-8-25
# TensorRT Implicit PTQ example
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
from calibrator import EngineCalibrator

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from base_model.utils_tmo import *
from base_trt.onnx2trt import transform_cv, test_model_topk_trt, get_inference_fps, show_build_time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

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
                    
            timing_cache = f"{CUR_DIR}/engine/timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.INT8)
                print("Using INT8 mode.")
                print("Using TensorRT implict PTQ mode.")

                calib_cache = f"{CUR_DIR}/engine/cache_table.table"
                calib_data_path = f"{CUR_DIR}/calib_datas/calib_data.npy"
                if os.path.exists(calib_cache):
                    os.remove(calib_cache)
                config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    inputs = [network.get_input(i) for i in range(network.num_inputs)]
                    batch_size = inputs[0].shape[0]
                    calib_shape = inputs[0].shape
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    config.int8_calibrator.set_calibrator(batch_size, calib_shape, calib_dtype, calib_data_path)
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
    onnx_model_path = f"{CUR_DIR}/../base_trt/onnx/{model_name}.onnx"
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}_trt_ptq.engine')
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
