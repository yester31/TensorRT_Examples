# by yhpark 2025-9-27
import torch
from ultralytics import YOLO
import cv2
import os 
import torchvision
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

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

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (640x640)
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

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

def draw_bounding_boxes(image, boxes, labels, scores, thrh = 0.6):
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

class YOLO12(torch.nn.Module):
    def __init__(self, checkpoint_path, class_count=80) -> None:
        super().__init__()

        model = YOLO(checkpoint_path)
        self.model = model.model
        self.class_count = class_count

    def forward(self, x):
        """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216"""
        pred: torch.Tensor = self.model(x)[0]  # [N, 84, 8400]
        pred1 = pred.permute(0, 2, 1) # [N, 84, 8400] -> [N, 8400, 84] 
        boxes, scores = pred1.split([4, self.class_count], dim=-1) # [N, 8400, 84] -> [N, 8400, 4], [N, 8400, 80]  
        boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        return boxes, scores

def main() :
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # load model
    class_count = 80
    model = YOLO12("yolo12n.pt", class_count)
    
    # preprocess input image
    image_path = f"{CUR_DIR}/data/dog.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    padded_image, scale, pad = letterbox(rgb_image)
    preproc_image = transform_cv(padded_image)  # Preprocess image
    tensor_image = torch.from_numpy(preproc_image)

    with torch.no_grad():
        output = model(tensor_image)  # predict on an image

    pred_boxes, scores = output
    pred_scores, pred_labels = torch.max(scores, dim=-1)

    pred_label = pred_labels[0]
    pred_box = pred_boxes[0]
    pred_score = pred_scores[0]

    # NMS 
    iou_threshold=0.7
    keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, iou_threshold)
    keep_topk = 300
    keep = keep[: keep_topk]

    labels = pred_label[keep]
    boxes = pred_box[keep]
    scores = pred_score[keep]

    thrh=0.6
    scr = scores
    lab = labels[scr > thrh]
    box = boxes[scr > thrh]
    scrs = scr[scr > thrh]

    box = scale_boxes_back(box, scale, pad, (height, width))

    output_img = draw_bounding_boxes(raw_image, box, lab, scrs)

    save_path = os.path.join(CUR_DIR, 'results', f"{image_file_name}_pt.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_img)
    print("Image processing complete.")


if __name__ == '__main__':
    main()