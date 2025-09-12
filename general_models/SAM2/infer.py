# by yhpark 2025-9-11
import torch
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time 

sys.path.insert(1, os.path.join(sys.path[0], "sam2"))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
#from sam2.sam2.build_sam import build_sam2
#from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, save_path, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        save_path_full = f"{save_path}_{i+1}.png"  # 파일명 동적으로 지정 가능
        plt.savefig(save_path_full, bbox_inches="tight", pad_inches=0.1, dpi=300)  # 고해상도 저장
        print(f"Saved result to {save_path_full}")

        plt.show()
        plt.close()

def main():
    # model_name = "sam2.1_hiera_large"
    # model_name = "sam2.1_hiera_base_plus"
    # model_name = "sam2.1_hiera_small"
    model_name = "sam2.1_hiera_tiny"
    checkpoint = f"{CUR_DIR}/sam2/checkpoints/{model_name}.pt"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_t.yaml"
    # checkpoint = f"{CUR_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

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
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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