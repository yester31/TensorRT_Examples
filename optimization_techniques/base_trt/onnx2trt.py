# by yhpark 2024-10-15
# by yhpark 2025-8-22
# TensorRT Model Optimization PTQ example
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))
import torchvision.transforms as transforms
import tensorrt as trt
import torch
import cv2
import os
import numpy as np
import time
import common
from common import *
import json

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from base_model.utils_tmo import *
set_random_seed()

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

            timing_cache = f"{CUR_DIR}/engine/timing.cache"
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

def transform_cv(image):    
    # 1) BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2) Resize (256, keep aspect ratio → shorter side 256)
    h, w, _ = image.shape
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_w, new_h = 256, int(h * 256 / w)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3) CenterCrop 224
    h, w, _ = image.shape
    start_x = (w - 224) // 2
    start_y = (h - 224) // 2
    image = image[start_y:start_y+224, start_x:start_x+224]

    # 4) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    tensor = torch.from_numpy(image)

    # 5) Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor = (tensor - mean) / std

    # Add batch dimension (C, H, W) -> (1, C, H, W)
    tensor = np.expand_dims(tensor, axis=0)
    # Return as NumPy array (C-order)   
    return np.array(tensor, dtype=np.float32, order="C")

def test_model_topk_trt(test_loader, output_shapes, context, engine, inputs, outputs, bindings, stream):
    top1_correct = 0
    top5_correct = 0
    total = 0

    # Warm-up
    batch = next(iter(test_loader))
    dummy_input = torch.randn((batch["image"].shape), requires_grad=False)  # Create a dummy input
    dummy_input = np.asarray(dummy_input.detach().cpu(), dtype=np.float32, order="C")
    for _ in range(10):
        common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)    
    torch.cuda.synchronize()

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            images = np.asarray(images.detach().cpu(), dtype=np.float32, order="C")
            inputs[0].host = images
            labels = batch["label"]

            # Forward pass
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) 

            t_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
            t_outputs = torch.from_numpy(t_outputs[0])
            _, pred = t_outputs.topk(5, dim=1, largest=True, sorted=True)

            total += labels.size(0)
            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1,1).expand_as(pred)).sum().item()

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total


    print(f"[TRT_E] Total Count: {total}")
    print(f"[TRT_E] Test Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"[TRT_E] Test Top-{5} Accuracy: {top5_acc*100:.2f}%")

    return top1_acc, top5_acc

def get_inference_fps(batch_size, context, engine, bindings, inputs, outputs, stream):
    # CUDA Events
    err, start_event = cudart.cudaEventCreate()
    err, end_event = cudart.cudaEventCreate()
    # Measure
    iterations = 10000
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
    precision = "fp16" # Choose 'fp32' or 'fp16'
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
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
