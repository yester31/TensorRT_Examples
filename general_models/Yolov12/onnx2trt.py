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
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

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
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(4))

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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    # scale factor
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)

    # resize
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    # padding 
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top

    # padding
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded, scale, (pad_left, pad_top)

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (640x640)
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

def scale_boxes_back(boxes, scale, pad, orig_shape):
    """
    boxes: [N,4], (x1, y1, x2, y2) in resized/letterbox coords
    scale: float, resizing
    pad: (pad_left, pad_top)
    orig_shape: (H_orig, W_orig)
    """
    pad_left, pad_top = pad
    H_orig, W_orig = orig_shape

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W_orig)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H_orig)

    return boxes


def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    
    iteration = 1000
    dur_time = 0
    coco_labels = ['face']

    # Input
    image_path = f"{CUR_DIR}/data/test1.jpg"
    # image_path = f"{CUR_DIR}/data/test2.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    batch_size = 1
    input_h, input_w = 640, 640 
    raw_image = cv2.imread(image_path)  # Load image
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    padded_image, scale, pad = letterbox(rgb_image)
    input_image = transform_cv(padded_image)   # Preprocess image
    print(input_image.shape)

    # Model and engine paths
    precision = "fp16"   # int8 or fp32 or fp16
    onnx_sim = True # True or False
    model_name = "yolov12n-face"
    model_name = f"{model_name}_{input_h}x{input_w}"
    if precision == "int8":
        quant_method = "moq" # moq or ptq
        model_name = f"{model_name}_{quant_method}"
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    model_name = f"{model_name}_w_nms"
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

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
            dur_time += time.time() - begin

        print(f'[TRT_E] {iteration} iterations time: {dur_time:.4f} [sec]')
        
    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    # Reshape and post-process the output
    num_dets = np.array(trt_outputs[0]).reshape((input_image.shape[0], 1)) 
    det_boxes = np.array(trt_outputs[1]).reshape((input_image.shape[0], 300, 4)) 
    det_scores = np.array(trt_outputs[2]).reshape((input_image.shape[0], 300)) 
    det_classes = np.array(trt_outputs[3]).reshape((input_image.shape[0], 300)) 

    for b_idx, num_det in enumerate(num_dets):
        restored_box = scale_boxes_back(det_boxes[b_idx][:num_det[0]], scale, pad, (height, width))
        for d_idx in range(num_det[0]):
            xyxy, conf, label = restored_box[d_idx], det_scores[b_idx][d_idx], det_classes[b_idx][d_idx]
            x1, y1, x2, y2 = map(int, xyxy)
            print(f"Face detected: ({x1}, {y1}), ({x2}, {y2}), conf={conf:.2f}, score={label}")
            cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0,255,0), 1)
            label_text = f'{label} {conf:.2f}'
            cv2.putText(raw_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow(f'{image_file_name}_{model_name}_trt', raw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_path = os.path.join(CUR_DIR, 'results', f'{image_file_name}_{model_name}_trt.jpg')
    cv2.imwrite(save_path, raw_image)

    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
