import os
import sys
import cv2  
import numpy as np
import torch
import torch.nn as nn
import shutil
import time 

sys.path.insert(1, os.path.join(sys.path[0], "Dome-DETR"))
from src.core import YAMLConfig

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes, targets=None):
        outputs = self.model(images, targets=targets)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


VisDrone_CLASSES = [
    'regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 
    'bus', 'motor', 'others'
]

AITOD_CLASSES = [
    'airplane', 'bridge', 'storage tank', 'ship', 'swimming pool', 'vehicle', 'person', 'wind mill'
]

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    model_size = "M" # M or L
    dataset = "AITOD" # AITOD(9) or VisDrone(12)
    config = f"{CUR_DIR}/Dome-DETR/configs/dome/Dome-{model_size}-{dataset}.yml"
    resume = f"{CUR_DIR}/Dome-DETR/pretrained/Dome-{model_size}-{dataset}-best.pth"
    cfg = YAMLConfig(config, resume=resume)
    # print(cfg)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(DEVICE).eval()

    if dataset == "VisDrone":
        NAME_CLASSES = VisDrone_CLASSES
    elif dataset == "AITOD":
        NAME_CLASSES = AITOD_CLASSES

    # Load image 
    image_paths = [
        f"{CUR_DIR}/Dome-DETR/aitod/val/images/0a2e15e29.png",
        f"{CUR_DIR}/Dome-DETR/aitod/val/images/9999970_00000_d_0000010__0_0.png",
        f"{CUR_DIR}/Dome-DETR/aitod/val/images/P2668__1.0__600___1985.png",
        f"{CUR_DIR}/Dome-DETR/aitod/val/images/15709.png",
        f"{CUR_DIR}/Dome-DETR/aitod/val/images/P0160__1.0__2400___1800.png",
        f"{CUR_DIR}/Dome-DETR/VisDrone2019-DET-val/images/0000026_02500_d_0000029.jpg",
        f"{CUR_DIR}/Dome-DETR/VisDrone2019-DET-val/images/0000024_00000_d_0000012.jpg",
        f"{CUR_DIR}/Dome-DETR/VisDrone2019-DET-val/images/0000072_02834_d_0000003.jpg",
        f"{CUR_DIR}/Dome-DETR/VisDrone2019-DET-val/images/0000348_03333_d_0000423.jpg",
    ]

    for i_idx in range(len(image_paths)) :
        image_path = image_paths[i_idx]
        shutil.copy2(image_path, f"{CUR_DIR}/data/{os.path.basename(image_path)}")
        image_file_name = os.path.splitext(os.path.basename(image_path))[0]
        im_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im_cv_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)  # convert to RGB

        # Original size (width, height)
        h, w = im_cv_rgb.shape[:2]
        orig_size = torch.tensor([[w, h]]).to(DEVICE)
        print(f"[MDET] original image size : {h, w}")
        # Resize to target size
        INPUT_SIZE = 800
        im_resized = cv2.resize(im_cv_rgb, (INPUT_SIZE, INPUT_SIZE))

        # Convert to tensor [C, H, W], normalize to [0,1]
        im_data = torch.from_numpy(im_resized).permute(2, 0, 1).float() / 255.0
        im_data = im_data.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(im_data, orig_size)

        labels, boxes, scores = output

        thrh = 0.5
        scr = scores[:]
        lab = labels[:][scr > thrh]
        box = boxes[:][scr > thrh]
        scrs = scr[scr > thrh]
        im_draw = im_cv.copy()
        for j, b in enumerate(box):
            x1, y1, x2, y2 = map(int, b.tolist()) # box [x1, y1, x2, y2]
            color = (_COLORS[lab[j].item()] * 255).astype(np.uint8).tolist() # COLOR
            cv2.rectangle(im_draw, (x1, y1), (x2, y2), color=color, thickness=1) # draw box (BGR)
            text = f"{lab[j].item()} {round(scrs[j].item(), 2)} {NAME_CLASSES[lab[j].item()]}" # label + score + class name
            # print(text)
            cv2.putText(im_draw, text, (x1, y1 - 5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=color,thickness=1,lineType=cv2.LINE_AA)

        save_path = f"{save_dir_path}/{image_file_name}_{model_size}_{dataset}_pt.jpg"
        cv2.imwrite(save_path, im_draw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Image processing complete. ({save_path})")


    iterations = 100
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            output = model(im_data, orig_size)
            torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start

    # Results
    print(f'[TRT_E] {iterations} iterations time: {elapsed_time:.4f} [sec]')
    fps = (iterations ) / elapsed_time
    avg_time = elapsed_time/(iterations)
    print(f'[TRT_E] Average FPS: {fps:.2f} [fps]')
    print(f'[TRT_E] Average inference time: {avg_time * 1000:.2f} [msec]')

if __name__ == "__main__":
    main()


