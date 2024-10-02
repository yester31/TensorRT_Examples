import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
    TrtRunner,
    EngineFromBytes,
)
from polygraphy.backend.common import BytesFromPath
import torch
import os
import cv2
import time
from utils import *

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Global Variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    iteration = 1000
    dur_time = 0

    # Input
    img_path = os.path.join(current_directory, 'data', 'panda0.jpg')
    img = cv2.imread(img_path)  # Load image
    input_image = preprocess_image(img)  # Preprocess image

    # Model and engine paths
    precision = "fp16" 
    model_name = "resnet18"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}.engine')
    
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    
    # Load or build TensorRT engine
    if os.path.exists(engine_file_path):
        print(f"[TRT] Reading engine from file {engine_file_path}")
        engine = EngineFromBytes(BytesFromPath(engine_file_path))
    else:
        engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_model_path), config=CreateConfig(fp16=True),) 
        engine = SaveEngine(engine, path=engine_file_path)
    
    # Perform inference and print results
    with TrtRunner(engine) as runner:

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"input": input_image})
        # Inference loop
        for i in range(iteration):
            begin = time.time()
            outputs = runner.infer(feed_dict={"input": input_image})
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        print(f'[TRT] {iteration} iterations time: {dur_time:.4f} [sec]')


        max_tensor = torch.from_numpy(outputs["output"]).max(dim=1)
        max_value = max_tensor[0].cpu().numpy()[0]
        max_index = max_tensor[1].cpu().numpy()[0]
        print(f'[TRT] Resnet18 max index: {max_index}, value: {max_value}, class name: {class_name[max_index]}')
        print("Inference succeeded!")


if __name__ == "__main__":
    main()
