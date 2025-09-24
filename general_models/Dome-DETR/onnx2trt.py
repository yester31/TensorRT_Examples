import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

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
print(f"Using device: {DEVICE}")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
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

            timing_cache = f"{os.path.dirname(engine_file_path)}/{os.path.splitext(os.path.basename(engine_file_path))[0]}_timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(2))

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
            
            if dynamic_input_shapes:
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    input = network.get_input(i_idx)
                    assert input.shape[0] == -1
                    min_shape = dynamic_input_shapes[0]
                    opt_shape = dynamic_input_shapes[1]
                    max_shape = dynamic_input_shapes[2]
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                    print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
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

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (640x640)
    image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

VisDrone_CLASSES = [
    'regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 
    'bus', 'motor', 'others'
]

AITOD_CLASSES = [
    'airplane', 'bridge', 'storage tank', 'ship', 'swimming pool', 'vehicle', 'person', 'wind mill'
]

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def draw_bounding_boxes(image, boxes, labels=None, scores=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image using OpenCV.
    
    Parameters:
    - image: The image on which to draw the bounding boxes (NumPy array).
    - boxes: List of bounding boxes in xyxy format [(x1, y1, x2, y2), ...].
    - labels: Optional list of labels for each bounding box.
    - color: Color of the bounding box (BGR format).
    - thickness: Thickness of the bounding box lines.
    """
    for i, box in enumerate(boxes):
        box = box.astype(int)
        x1, y1, x2, y2 = box  # Ensure the coordinates are integers
        # Draw rectangle on the image
        color = (_COLORS[labels[i]] * 255).astype(np.uint8).tolist() # COLOR

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        
        # If labels are provided, display them on the bounding box
        if labels is not None:
            text = f"{labels[i]} {scores[i]:.2f} {AITOD_CLASSES[labels[i]]}"
            cv2.putText(image, text, (x1, y1 - 5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=color,thickness=1,lineType=cv2.LINE_AA)

    return image


def main():
    iteration = 100
    dur_time = 0

    # Input
    image_paths = [
        f"{CUR_DIR}/data/0a2e15e29.png",
        f"{CUR_DIR}/data/9999970_00000_d_0000010__0_0.png",
        f"{CUR_DIR}/data/P2668__1.0__600___1985.png",
        f"{CUR_DIR}/data/15709.png",
        f"{CUR_DIR}/data/P0160__1.0__2400___1800.png",
        f"{CUR_DIR}/data/0000026_02500_d_0000029.jpg",
        f"{CUR_DIR}/data/0000024_00000_d_0000012.jpg",
        f"{CUR_DIR}/data/0000072_02834_d_0000003.jpg",
        f"{CUR_DIR}/data/0000348_03333_d_0000423.jpg",
    ]


    image_path = image_paths[0]
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    batch_size = 1
    input_h, input_w = 800, 800 
    raw_image = cv2.imread(image_path)  # Load image
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    tensor = transform_cv(image_rgb)
    orig_size = np.array([[width, height]])

    # Model and engine paths
    precision = "fp16"   # int8 or fp32 or fp16
    model_size = "M"    # M or L
    dataset = "AITOD"   # AITOD(9) or VisDrone(12)
    model_name = f"Dome-{model_size}-{dataset}"
    onnx_sim = True # True or False
    dynamic = False  # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = tensor
        inputs[1].host = orig_size

        # Warm-up
        for i in range(50):
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        # Inference loop
        for i in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            dur_time += time.time() - begin

        print(f'[TRT_E] {iteration} iterations time: {dur_time:.4f} [sec]')
        
    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    # Reshape and post-process the output
    labels = np.array(trt_outputs[0]).reshape((1,300))  
    boxes = np.array(trt_outputs[1]).reshape((1,300,4))  
    scores = np.array(trt_outputs[2]).reshape((1,300)) 
    
    thrh=0.5
    scr = scores[0]
    lab = labels[0][scr > thrh]
    box = boxes[0][scr > thrh]
    scrs = scr[scr > thrh]

    output_img = draw_bounding_boxes(raw_image, box, lab, scrs)

    save_path = os.path.join(CUR_DIR, 'results', f'{image_file_name}_{model_size}_{dataset}_trt.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_img)
    
    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
