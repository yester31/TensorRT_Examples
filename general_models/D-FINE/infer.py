import os
import sys
import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(1, os.path.join(sys.path[0], "D-FINE"))
from src.core import YAMLConfig

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (640x640)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # config = f"{CUR_DIR}/D-FINE/configs/dfine/dfine_hgnetv2_n_coco.yml"
    # resume = f"{CUR_DIR}/D-FINE/checkpoints/dfine_n_coco.pth"
    config = f"{CUR_DIR}/D-FINE/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml"
    resume = f"{CUR_DIR}/D-FINE/checkpoints/dfine_s_obj2coco.pth"

    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(DEVICE)


    file_path = f"{CUR_DIR}/data/test3.jpg"
    filename = os.path.splitext(os.path.basename(file_path))[0]
    image = cv2.imread(file_path)
    h, w = image.shape[:2] # [h, w, c]


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform_cv(image_rgb)
    tensor = torch.from_numpy(tensor).to(DEVICE)
    orig_size = torch.tensor([[w, h]]).to(DEVICE)

    output = model(tensor, orig_size)

    labels, boxes, scores = output
    im_pil = Image.fromarray(image_rgb, mode='RGB')
    thrh=0.6
    for i, im in enumerate([im_pil]):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text((b[0], b[1]),text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",fill="blue",)

        im.save(f"{CUR_DIR}/results/{filename}_{i}.jpg")

    print("Image processing complete.")

if __name__ == "__main__":
    main()
