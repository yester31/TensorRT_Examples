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

def preprocess_image(img):
    """Preprocess the image for inference."""
    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW, uint8 -> float32
        img /= 255.0  # Normalize to [0, 1]
        img = img.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    return np.array(img, dtype=np.float32, order="C")


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
                raise FileNotFoundError(f"[TRT] ONNX file {onnx_file_path} not found.")

            print(f"[TRT] Loading and parsing ONNX file: {onnx_file_path}")
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    raise RuntimeError("[TRT] Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

            network.get_input(0).shape = [1, 3, 224, 224]
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)

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

def main():
    iteration = 1000
    dur_time = 0

    # Input
    img = cv2.imread(f'{current_directory}/data/panda0.jpg')  # Load image
    input_image = preprocess_image(img)  # Preprocess image

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "resnet18"
    onnx_model_path = f"onnx/{model_name}_{device.type}.onnx"
    engine_file_path = f"engine/{model_name}_{precision}.engine"
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes expected
    output_shapes = [(1, 1000)]

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

        print(f'[TRT] {iteration} iterations time: {dur_time:.4f} [sec]')
        
    
    # Reshape and post-process the output
    t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    # Results
    avg_time = dur_time / iteration
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')

    max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
    max_value = max_tensor[0].cpu().numpy()[0]
    max_index = max_tensor[1].cpu().numpy()[0]
    print(f'[TRT] Resnet18 max index: {max_index}, value: {max_value}, class name: {class_name[max_index]}')
    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
