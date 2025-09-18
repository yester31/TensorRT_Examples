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
import torchvision

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

def transform_cv(img, input_size=(640,640), swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    # 3) Add batch dimension
    padded_img = np.expand_dims(padded_img, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(padded_img, dtype=np.float32, order="C"), r

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    prediction = torch.from_numpy(prediction)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
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

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    iteration = 1000
    dur_time = 0

    # Input
    image_path = f"{CUR_DIR}/YOLOX/assets/dog.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    batch_size = 1
    input_h, input_w = 640, 640 
    img = cv2.imread(image_path)  # Load image
    raw_img = img.copy()
    height, width = img.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    # image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    tensor, ratio = transform_cv(img)
    orig_size = np.array([[width, height]])

    # Model and engine paths
    precision = "fp16"   # int8 or fp32 or fp16
    model_name = "yolox-s"
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
    trt_outputs = np.array(trt_outputs[0]).reshape((1, 8400, 85))  

    confthre = 0.25 
    nmsthre = 0.45 
    num_classes = len(COCO_CLASSES) # 80
    trt_outputs = postprocess(trt_outputs, num_classes, confthre, nmsthre, class_agnostic=True)

    output = trt_outputs[0]
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    result_image = vis(raw_img, bboxes, scores, cls, confthre, COCO_CLASSES)
    save_file_name = os.path.join(save_dir_path, f"{image_file_name}_trt.jpg")
    print("Saving detection result in {}".format(save_file_name))
    cv2.imwrite(save_file_name, result_image)

    
    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
