# by yhpark 2025-9-3
import torch
from ultralytics import YOLO
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(1, os.path.join(sys.path[0], "gazelle"))
from gazelle.gazelle.model import get_gazelle_model

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualize predicted gaze heatmap for each person and gaze in/out of frame score
def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
          text = f"in-frame: {inout_score:.2f}"
          text_width = draw.textlength(text)
          text_height = int(height * 0.01)
          text_x = xmin * width
          text_y = ymax * height + text_height
          draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
    return overlay_image

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
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

    return overlay_image

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

def main() :
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # 1. load models
    # 1.1 load faces detection model
    model_face = YOLO(f'{CUR_DIR}/checkpoints/yolov12n-face.pt')  # load a pretrained YOLOv12n detection model
    # 1.2 load Gaze-LLE model
    model_name = 'gazelle_dinov2_vitb14_inout'
    # model_name = 'gazelle_dinov2_vitb14'
    # model, transform = torch.hub.load('fkryan/gazelle', model_name)
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(f'{CUR_DIR}/checkpoints/{model_name}.pt', weights_only=True))
    model.eval().to(DEVICE)

    # 2. load image
    image_path = f"{CUR_DIR}/data/test2.jpg" # test1 or test2
    filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    height, width = image.shape[:2] # [h, w, c]

    # 3. detect faces
    results = model_face(image)  # predict on an image
    xyxys = results[0].boxes.xyxy
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in xyxys.cpu()]]
    # print(norm_bboxes)

    # 4. prepare gazelle input
    cv_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_tensor = transform(cv_img_rgb).unsqueeze(0)
    img_tensor = transform_cv(cv_img_rgb)
    img_tensor = torch.from_numpy(img_tensor)
    input = {
        "images": img_tensor.to(DEVICE), # [num_images, 3, 448, 448]
        "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
    }

    # 5. run gaze
    import time
    iteration = 100
    dur_time = 0
    for _ in range(iteration):
        begin = time.time()
        with torch.no_grad():
            output = model(input)
        torch.cuda.synchronize()
        dur_time += time.time() - begin
    # ===================================================================
    # Results
    print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
    print('[MDET] Post process')

    # 6. post process
    img1_person1_heatmap = output['heatmap'][0][0] # [64, 64] heatmap
    print(img1_person1_heatmap.shape)
    if model.inout:
        img1_person1_inout = output['inout'][0][0] # gaze in frame score (if model supports inout prediction)
        print(img1_person1_inout.item())

    pil_img = Image.fromarray(cv_img_rgb)
    for i in range(len(xyxys)):
        plt.figure()
        plt.imshow(visualize_heatmap(pil_img, output['heatmap'][0][i], norm_bboxes[0][i], inout_score=output['inout'][0][i] if output['inout'] is not None else None))
        plt.axis('off')
        plt.savefig(f"{CUR_DIR}/results/{filename}_heatmap_{i}.png", bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(10,10))
    plt.imshow(visualize_all(pil_img, output['heatmap'][0], norm_bboxes[0], output['inout'][0] if output['inout'] is not None else None, inout_thresh=0.5))
    plt.axis('off')
    plt.savefig(f"{CUR_DIR}/results/{filename}_heatmap_all.png", bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    main()