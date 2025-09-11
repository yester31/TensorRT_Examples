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
                
            network.unmark_output(network.get_output(3))
            network.unmark_output(network.get_output(2))
            network.unmark_output(network.get_output(1))
                
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
   return np.asarray(img, dtype=np.float32, order="C")

def xywh_to_xyxy_np(boxes):
    """Convert bounding boxes from xywh to xyxy format (NumPy)."""
    # xywh: [x_center, y_center, width, height]
    # xyxy: [x1, y1, x2, y2]
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width / 2
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height / 2
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + width / 2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + height / 2
    return xyxy_boxes

def nms_numpy(boxes, scores, iou_threshold=0.45):
    """Perform Non-Maximum Suppression (NMS) using NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def non_max_suppression(batch_pred, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Perform Non-Maximum Suppression (NMS) without objectness confidence, using NumPy."""
    
    batch_size = batch_pred.shape[0]  # Get batch size
    all_batch_boxes = []
    
    for i in range(batch_size):
        # Select one image's predictions (shape: [84, 8400])
        pred = batch_pred[i]

        # Split predictions: first 4 values are bounding box coordinates (xywh)
        boxes = pred[:4, :].T  # (8400, 4) - convert to shape [8400, 4]
        # Remaining values are class scores
        class_scores = pred[4:, :]  # (80, 8400)

        # Convert xywh format to xyxy format for bounding boxes
        boxes = xywh_to_xyxy_np(boxes)

        # Find the class with the highest score for each prediction
        class_conf = np.max(class_scores, axis=0)  # (8400,)
        class_pred = np.argmax(class_scores, axis=0)  # (8400,)

        # Apply confidence threshold on class confidence (instead of object confidence)
        mask = class_conf >= conf_thres
        boxes = boxes[mask]
        class_conf = class_conf[mask]
        class_pred = class_pred[mask]

        # Apply Non-Maximum Suppression (NMS)
        if len(boxes) > 0:
            indices = nms_numpy(boxes, class_conf, iou_thres)
            if len(indices) > max_det:
                indices = indices[:max_det]
            final_boxes = boxes[indices]
            final_scores = class_conf[indices]
            final_classes = class_pred[indices]
        else:
            final_boxes, final_scores, final_classes = np.array([]), np.array([]), np.array([])

        # Store the results for the current image
        all_batch_boxes.append((final_boxes, final_scores, final_classes))

    return all_batch_boxes

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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # If labels are provided, display them on the bounding box
        if labels is not None:
            label = f'{labels[i]} {scores[i]:.3f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

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
    iteration = 1000
    dur_time = 0
    coco_labels = ['face']

    # Input
    image_path = f"{CUR_DIR}/data/test1.jpg"
    image_path = f"{CUR_DIR}/data/test2.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    batch_size = 1
    input_h, input_w = 640, 640 
    raw_image = cv2.imread(image_path)  # Load image
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    img_padded, scale, pad = letterbox(raw_image)
    input_image = preprocess_image(img_padded)  # Preprocess image
    batch_images = np.concatenate([input_image], axis=0)
    print(batch_images.shape)

    # Model and engine paths
    precision = "fp16"   # int8 or fp32 or fp16
    model_name = "yolov12n-face"
    model_name = f"{model_name}_{input_h}x{input_w}"
    if precision == "int8":
        quant_method = "moq" # moq or ptq
        model_name = f"{model_name}_{quant_method}"
        dynamic, onnx_sim = False, False 
    else:
        dynamic = False # True or False
        onnx_sim = True # True or False
        model_name = f"{model_name}_dynamic" if dynamic else model_name
        model_name = f"{model_name}_sim" if onnx_sim else model_name

    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    if dynamic:
        dynamic_input_shapes = [[batch_size,3,input_h, input_w],[batch_size,3,input_h, input_w],[batch_size,3,input_h, input_w]]
    else:
        dynamic_input_shapes = None

    # Output shapes expected
    output_shapes = [(batch_images.shape[0], 5, 8400)] # 640 
    # output_shapes = [(batch_images.shape[0], 5, 5376)] # 512

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shapes[0], profile_idx=0)
        inputs[0].host = batch_images
        if dynamic:
            context.set_input_shape('input', batch_images.shape)

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
        
    # Results
    avg_time = dur_time / iteration
    print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    # Reshape and post-process the output
    t_outputs = np.array(trt_outputs).reshape(output_shapes[0]) 
    results = non_max_suppression(t_outputs)
    
    # Print results for each image in the batch
    restored_boxes = []
    for img_idx, (boxes, scores, classes) in enumerate(results):
        print(f"Image {img_idx + 1}:")
        restored_boxes = scale_boxes_back(boxes, scale, pad, (height, width))
        print(f"  Boxes: {restored_boxes}")
        print(f"  Scores: {scores}")
        labels = [coco_labels[idx] for idx in classes]
        print(f"  Classes: {labels}")

    output_img = draw_bounding_boxes(raw_image, restored_boxes, labels, scores)
    
    save_path = os.path.join(CUR_DIR, 'results', f'{image_file_name}_{model_name}_trt.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_img)
    
    common.free_buffers(inputs, outputs, stream)
    print("[TRT_E] Inference succeeded!")


if __name__ == '__main__':
    main()
