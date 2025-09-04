# by yhpark 2025-8-5
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import tensorrt as trt
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import cv2
import numpy as np
import time
import common
from common import *

import sys
sys.path.insert(1, os.path.join(sys.path[0], "gazelle"))
from gazelle.gazelle.model import *
from infer import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
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
                raise FileNotFoundError(f"[MDET] ONNX file {onnx_file_path} not found.")

            parser.parse_from_file(onnx_file_path)
            
            timing_cache = f"{os.path.dirname(engine_file_path)}/{os.path.splitext(os.path.basename(engine_file_path))[0]}_timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(4))
            # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f'[MDET] set fp16 model')

            if dynamic_input_shapes is not None :
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    input = network.get_input(i_idx)
                    assert input.shape[0] == -1
                    min_shape = dynamic_input_shapes[i_idx][0]
                    opt_shape = dynamic_input_shapes[i_idx][1]
                    max_shape = dynamic_input_shapes[i_idx][2]
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                    print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
                config.add_optimization_profile(profile)

            for i_idx in range(network.num_inputs):
                print(f'[MDET] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[MDET] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
            
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("Failed to build TensorRT engine. Likely due to shape mismatch or model incompatibility.")
            
            engine = runtime.deserialize_cuda_engine(plan)
            common.save_timing_cache(config, timing_cache)

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            
            return engine

    print(f"[MDET] Engine file path: {engine_file_path}")

    if os.path.exists(engine_file_path):
        print(f"[MDET] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.1f} [min] {build_time % 60 :.2f} [sec]"
        print(f'[MDET] engine build done! ({build_time_str})')

        return engine

def transform_cv(image):    
    # 1) Resize (448x448)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)

    # 2) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 4) Normalize
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    image = (image - mean) / std

    # 5) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

def get_input_head_maps(bboxes):
    # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
    head_maps = []
    for bbox_list in bboxes:
        img_head_maps = []
        for bbox in bbox_list:
            if bbox is None:  # no bbox provided, use empty head map
                img_head_maps.append(np.zeros((32, 32), dtype=np.float32))
            else:
                xmin, ymin, xmax, ymax = bbox
                width, height = 32, 32
                xmin = round(xmin * width)
                ymin = round(ymin * height)
                xmax = round(xmax * width)
                ymax = round(ymax * height)
                head_map = np.zeros((height, width), dtype=np.float32)
                head_map[ymin:ymax, xmin:xmax] = 1.0
                img_head_maps.append(head_map)
        head_maps.append(np.stack(img_head_maps, axis=0))
    return head_maps

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

    return overlay_image


def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    image_path = os.path.join(CUR_DIR, 'data', 'test1.jpg')
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    print('[MDET] Pre process')
    cv_img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    img_tensor = transform_cv(cv_img_rgb)

    if image_file_name == "test1":
        norm_bboxes = [[
            [0.12902,0.054329,0.20916,0.27372], 
            [0.44093,0.18374,0.50237,0.33883], 
            [0.73411,0.10138,0.81696,0.31059]]]
    elif image_file_name == "test2":
        norm_bboxes = [[
            [0.44657,0.11285,0.52648,0.26578], 
            [0.11726,0.077865,0.18988,0.25895], 
            [0.75371,0.12752,0.82133,0.27557]]]    

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = f"yolov12"
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    model_name2 = f"gazelle_dinov2_vitb14_inout_1"
    onnx_model_path2 = os.path.join(CUR_DIR, 'onnx', f'{model_name2}.onnx')
    engine_file_path2 = os.path.join(CUR_DIR, 'engine', f'{model_name2}_{precision}.engine')

    model_name3 = f"gazelle_dinov2_vitb14_inout_2"
    onnx_model_path3 = os.path.join(CUR_DIR, 'onnx', f'{model_name3}.onnx')
    engine_file_path3 = os.path.join(CUR_DIR, 'engine', f'{model_name3}_{precision}.engine')

    dynamic_x_shapes = [[1,256,32,32], [3,256,32,32], [10,256,32,32]]
    dynamic_head_maps_shapes = [[1,32,32], [3,32,32], [10,32,32]]

    # engine = get_engine(onnx_model_path, engine_file_path, precision)
    engine2 = get_engine(onnx_model_path2, engine_file_path2, precision)
    engine3 = get_engine(onnx_model_path3, engine_file_path3, precision, [dynamic_x_shapes, dynamic_head_maps_shapes])

    # context = engine.create_execution_context()
    context2 = engine2.create_execution_context()
    context3 = engine3.create_execution_context()

    output_shape = {"heatmap":[10, 64, 64],"inout_preds":[10]}
    try:
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs3, outputs3, bindings3, stream3 = common.allocate_buffers(engine3, output_shape, 0)
        inputs2, outputs2, bindings2, stream = common.allocate_buffers(engine2)

        # Inference loop
        iteration = 100
        dur_time = 0
        for _ in range(iteration):
            begin = time.time()

            # inputs2[0].host = img_tensor   
            # trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
            head_maps = get_input_head_maps(norm_bboxes)
            face_count = 3

            # gazelle_1
            inputs2[0].host = img_tensor
            trt_outputs2 = common.do_inference(context2, engine=engine2, bindings=bindings2, inputs=inputs2, outputs=outputs2, stream=stream)
            
            x = np.tile(trt_outputs2[0], (face_count))
            context3.set_input_shape('x', (face_count,256,32,32))
            context3.set_input_shape('head_maps', (face_count,32,32))

            # gazelle_2
            inputs3[0].host = x
            inputs3[1].host = head_maps[0]
            trt_outputs3 = common.do_inference(context3, engine=engine3, bindings=bindings3, inputs=inputs3, outputs=outputs3, stream=stream)

            heatmap = trt_outputs3[0][:face_count * 64 * 64].reshape((3, 64, 64))
            inout_preds = trt_outputs3[1][:face_count]

            dur_time += time.time() - begin
        # ===================================================================
        # Results
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
        print('[MDET] Post process')
    finally:
        del context2, context3
        del engine2, engine3

    # ===================================================================
    print('[MDET] Generate color depth image')
    pil_img = Image.fromarray(cv_img_rgb)

    plt.figure(figsize=(10,10))
    plt.imshow(visualize_all(pil_img, heatmap, norm_bboxes[0], inout_preds, inout_thresh=0.5))
    plt.axis('off')
    plt.savefig(f"{CUR_DIR}/results/{image_file_name}_heatmap_all_trt.png", bbox_inches="tight", pad_inches=0)

    common.free_buffers(inputs2, outputs2, stream)
    common.free_buffers(inputs3, outputs3, stream3)

if __name__ == '__main__':
    main()
