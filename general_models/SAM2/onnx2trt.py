# by yhpark 2025-9-12
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))
import tensorrt as trt
import torch
import cv2
import numpy as np
import time
import common
from common import *
from infer import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")
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
                raise FileNotFoundError(f"[TRT_E] ONNX file {onnx_file_path} not found.")

            parser.parse_from_file(onnx_file_path)
            
            timing_cache = f"{os.path.dirname(engine_file_path)}/{os.path.splitext(os.path.basename(engine_file_path))[0]}_timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(4))
            # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f'[TRT_E] set fp16 model')

            if dynamic_input_shapes is not None :
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    input = network.get_input(i_idx)
                    if input.shape[0] != -1 :
                        continue
                    name = input.name
                    min_shape = dynamic_input_shapes[name][0]
                    opt_shape = dynamic_input_shapes[name][1]
                    max_shape = dynamic_input_shapes[name][2]
                    profile.set_shape(name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                    print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(name, min_shape, opt_shape, max_shape))
                config.add_optimization_profile(profile)

            for i_idx in range(network.num_inputs):
                print(f'[TRT_E] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[TRT_E] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
            
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("Failed to build TensorRT engine. Likely due to shape mismatch or model incompatibility.")
            
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
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.1f} [min] {build_time % 60 :.2f} [sec]"
        print(f'[TRT_E] engine build done! ({build_time_str})')

        return engine

def transform_cv(image):    
    # 1) Resize (1024x1024)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

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

def prepare_points(
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
        image_size, input_size
) -> tuple[np.ndarray, np.ndarray]:
    input_point_coords = point_coords[np.newaxis, ...]
    input_point_labels = point_labels[np.newaxis, ...]
    input_point_coords[..., 0] = (
        input_point_coords[..., 0]
        / image_size[1]
        * input_size[1]
    )
    input_point_coords[..., 1] = (
        input_point_coords[..., 1]
        / image_size[0]
        * input_size[0]
    )
    return input_point_coords.astype(np.float32), input_point_labels.astype(
        np.float32
    )

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    input_size = 1024
    image_path = os.path.join(CUR_DIR, 'data', 'truck.jpg')
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    image_size = raw_image.shape[:2]
    height, width = image_size
    print(f"[TRT_E] original image size : {height, width}")
    print('[TRT_E] Pre process')
    cv_img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    img_tensor = transform_cv(cv_img_rgb)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    onnx_sim = True # True or False
    # model_name = "sam2.1_hiera_large"
    # model_name = "sam2.1_hiera_base_plus"
    # model_name = "sam2.1_hiera_small"
    model_name = "sam2.1_hiera_tiny"
    model_name1 = f"{model_name}_image_encoder"
    model_name1 = f"{model_name1}_sim" if onnx_sim else model_name1
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name1}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name1}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    model_name2 = f"{model_name}_image_decoder"
    model_name2 = f"{model_name2}_sim" if onnx_sim else model_name2
    onnx_model_path2 = os.path.join(CUR_DIR, 'onnx', f'{model_name2}.onnx')
    engine_file_path2 = os.path.join(CUR_DIR, 'engine', f'{model_name2}_{precision}.engine')
    
    dynamic_input_shapes = {
        "point_coords":[[1,1,2], [1,2,2], [3,4,2]],
        "point_labels": [[1,1], [1,2], [3,4]],
        "mask_input": [[1,1,256,256], [1,1,256,256], [3,1,256,256]],
        "has_mask_input": [[1], [1], [3]]
    }

    engine = get_engine(onnx_model_path, engine_file_path, precision)
    engine2 = get_engine(onnx_model_path2, engine_file_path2, precision, dynamic_input_shapes)

    context = engine.create_execution_context()
    context2 = engine2.create_execution_context()

    dynamic_output_shapes = {
        "masks":[1, 4, 256, 256],
        "iou_predictions": [1, 4],
        "low_res_masks": [1, 4, 256, 256],
    }

    try:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs2, outputs2, bindings2, stream2 = common.allocate_buffers(engine2, dynamic_output_shapes, 0)

        # warm-up
        for _ in range(30):
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            context2.set_input_shape('point_coords', (1,1,2))
            context2.set_input_shape('point_labels', (1,1))
            context2.set_input_shape('mask_input', (1,1,256,256))
            context2.set_input_shape('has_mask_input', (1,))
            trt_outputs2 = common.do_inference(context2, engine=engine2, bindings=bindings2, inputs=inputs2, outputs=outputs2, stream=stream2)

        # Inference loop
        iteration = 1000
        dur_time = 0
        for _ in range(iteration):
            begin = time.time()

            # image_encoder
            inputs[0].host = img_tensor   
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            # image_decoder
            inputs2[0].host = trt_outputs[0] # image_embeddings
            inputs2[1].host = trt_outputs[1] # high_res_features1
            inputs2[2].host = trt_outputs[2] # high_res_features2

            input_point = np.array([[500, 375]])
            input_label = np.array([1])

            batch_num = input_point.shape[0]
            for i in range(batch_num):
                points, labels = np.array(input_point[i]), np.array(input_label[i])
                points, labels = prepare_points(points, labels, image_size, (input_size,input_size))
                if i == 0:
                    input_point_coords = points
                    input_point_labels = labels
                else:
                    input_point_coords = np.append(input_point_coords, points, axis=0)
                    input_point_labels = np.append(input_point_labels, labels, axis=0)

            inputs2[3].host = input_point_coords # point_coords
            inputs2[4].host = input_point_labels # point_labels

            mask_input = np.zeros((1, 1, input_size//4, input_size//4,), dtype=np.float32,)
            inputs2[5].host = mask_input # mask_input

            has_mask_input = np.array([0], dtype=np.float32)
            inputs2[6].host = has_mask_input # has_mask_input

            context2.set_input_shape('point_coords', (1,1,2))
            context2.set_input_shape('point_labels', (1,1))
            context2.set_input_shape('mask_input', (1,1,256,256))
            context2.set_input_shape('has_mask_input', (1,))

            trt_outputs2 = common.do_inference(context2, engine=engine2, bindings=bindings2, inputs=inputs2, outputs=outputs2, stream=stream2)
            scores = trt_outputs2[0].reshape((1,4))  # 4 
            low_res_masks = trt_outputs2[1].reshape((1,4,256,256))  # 1,4,256,256
            
            dur_time += time.time() - begin

            x_hwc = np.transpose(low_res_masks[0], (1, 2, 0))  # (N, C, H, W) → (H, W, C)
            resized_hwc = cv2.resize(x_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
            masks = np.transpose(resized_hwc, (2, 0, 1))[np.newaxis, ...]  # (H, W, C) → (N, C, H, W)
            masks = masks > 0.0

            low_res_masks = np.clip(low_res_masks, -32.0, 32.0)

        # ===================================================================
        # Results
        print(f'[TRT_E] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[TRT_E] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')
        print('[TRT_E] Post process')
    finally:
        del context, context2
        del engine, engine2

    # ===================================================================
    print('[TRT_E] visualization results')

    mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=np.uint8)
    for i in range(batch_num):
        max_idx = np.argmax(scores[i])
        m = masks[i][max_idx]
        mask[m > 0.0] = 255
    save_path = os.path.join(save_dir_path, f'{image_file_name}_{model_name}_top_mask_trt.jpg')
    cv2.imwrite(save_path, mask)

    save_path = os.path.join(save_dir_path, f'{image_file_name}_{model_name}_trt')
    show_masks(cv_img_rgb, masks[0], scores[0], save_path, point_coords=input_point, input_labels=input_label, borders=True)

    common.free_buffers(inputs, outputs, stream)
    common.free_buffers(inputs2, outputs2, stream2)

if __name__ == '__main__':
    main()
