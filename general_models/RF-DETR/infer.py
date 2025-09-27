# by yhpark 2025-9-27
import numpy as np
import torch
import torchvision
import torch.nn as nn
import cv2
import os
import sys
import torch.nn.functional as F
sys.path.insert(1, os.path.join(sys.path[0], "rf-detr"))
from rfdetr import RFDETRBase, RFDETRNano
from rfdetr.models.backbone.projector import LayerNorm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (384,384)
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 4) Normalize
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    image = (image - mean) / std

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

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

def draw_bounding_boxes(image, boxes, labels, scores):
    """
    Draw bounding boxes with labels and scores on an image.
    
    Args:
        image (np.ndarray): Input BGR image (H, W, 3)
        boxes (np.ndarray): Bounding boxes [N, 4] in (x1, y1, x2, y2)
        labels (list[int] or np.ndarray): Class indices for each box
        scores (list[float] or np.ndarray): Confidence scores for each box
    Returns:
        np.ndarray: Image with drawn boxes
    """

    # === Precompute frequently used lookups ===
    # Convert colors to uint8 only once
    colors = (_COLORS[labels] * 255).astype(np.uint8)  

    for i, box in enumerate(boxes):
        # Extract box coordinates (cast to int once)
        x1, y1, x2, y2 = map(int, box)

        # Fetch label and score
        label = labels[i]
        score = scores[i]

        # Pick precomputed color
        color = tuple(int(c) for c in colors[i])  

        # Draw rectangle (thin outline for speed)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)

        # Build text string only once
        text = f"{label} {score:.2f} {COCO_CLASSES[label]}"
        print(text)
        # Draw label text above bounding box
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 5, 0)),  # Ensure text not out of bounds
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return image

class SafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x / (x.max() + self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class RF_DETR(nn.Module):
    def __init__(self, model, num_select=300):
        super().__init__()
        model = model.model.model
        self.model = model
        self.num_select = num_select
        self._replace_bicubic_with_bilinear()
        self._replace_layernorm_with_safe()

    def _replace_bicubic_with_bilinear(self):
        old_interpolate = F.interpolate
        def safe_interpolate(input, size=None, scale_factor=None,
                             mode="nearest", align_corners=None, antialias=False):
            if mode == "bicubic":
                mode = "bilinear"   # 강제 변경
            return old_interpolate(input, size=size, scale_factor=scale_factor,
                                   mode=mode, align_corners=align_corners)
        F.interpolate = safe_interpolate

    def _replace_layernorm_with_safe(self):
        def recursive_replace(module):
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    setattr(module, name, SafeLayerNorm(
                        child.normalized_shape,
                        child.eps,
                        child.weight,
                        child.bias,
                    ))
                else:
                    recursive_replace(child)
        recursive_replace(self.model)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = torchvision.ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return labels, boxes, scores

def main():
    model = RFDETRNano()
    model = RF_DETR(model)
    model = model.eval().to(DEVICE)

    image_path = f"{CUR_DIR}/data/dog.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2] # [h, w, c]
    print(f"[MDET] original image size : {height, width}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    tensor = transform_cv(image_rgb)
    tensor = torch.from_numpy(tensor).to(DEVICE)
    orig_size = torch.tensor([[width, height]]).to(DEVICE)

    with torch.no_grad():
        output = model(tensor, orig_size)

    labels, boxes, scores = output

    labels = labels.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    thrh=0.6
    scr = scores[0]
    lab = labels[0][scr > thrh]
    box = boxes[0][scr > thrh]
    scrs = scr[scr > thrh]

    output_img = draw_bounding_boxes(raw_image, box, lab, scrs)

    save_path = os.path.join(CUR_DIR, 'results', f"{image_file_name}_pt.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_img)
    print("Image processing complete.")


if __name__ == '__main__':
    main()

