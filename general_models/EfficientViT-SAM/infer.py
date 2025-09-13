# by yhpark 2025-9-13
import torch
import sys
import os 
import numpy as np
from PIL import Image
import time 
from utils_sam import *

sys.path.insert(1, os.path.join(sys.path[0], "efficientvit"))
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # model_name = "efficientvit-sam-xl1"
    # model_name = "efficientvit-sam-xl0"
    # model_name = "efficientvit-sam-l2"
    # model_name = "efficientvit-sam-l1"
    model_name = "efficientvit-sam-l0"
    pretrained_path = os.path.join(CUR_DIR, 'efficientvit', 'checkpoint', model_name.replace("-", "_") + ".pt")
    efficientvit_sam = create_efficientvit_sam_model(name=model_name, weight_url=pretrained_path ,pretrained=True)
    efficientvit_sam = efficientvit_sam.eval().to(DEVICE)
    predictor = EfficientViTSamPredictor(efficientvit_sam)

    image_path = os.path.join(CUR_DIR, 'data', 'truck.jpg')
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    input_box = None

    #input_box = np.array([425, 600, 700, 875]) # xyxy
    #input_point = np.array([[575, 750]])
    #input_label = np.array([0])

    # warm-up
    for _ in range(30):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            box=input_box, 
            multimask_output=True,
            )

    iterations = 1000
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            box=input_box, 
            multimask_output=True,
            )
        torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start

    # Results
    print(f'[TRT_E] {iterations} iterations time: {elapsed_time:.4f} [sec]')
    fps = (iterations ) / elapsed_time
    avg_time = elapsed_time/(iterations)
    print(f'[TRT_E] Average FPS: {fps:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = f"{save_dir_path}/{image_file_name}_{model_name}_pt"
    show_masks(image, masks, scores, save_path, point_coords=input_point, input_labels=input_label, borders=True)


if __name__ == '__main__':
    main()