# by yhpark 2025-9-3
import torch
from ultralytics import YOLO
import cv2
import os 
import torchvision
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class YOLOv12(torch.nn.Module):
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
    checkpoint_path = f'{CUR_DIR}/checkpoints/yolov12n-face.pt'
    class_count = 1
    model_face = YOLOv12(checkpoint_path, class_count)
    
    # preprocess input image
    image_path = f"{CUR_DIR}/data/test1.jpg"
    # image_path = f"{CUR_DIR}/data/test2.jpg"
    filename = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    padded_image, scale, pad = letterbox(rgb_image)
    preproc_image = transform_cv(padded_image)  # Preprocess image
    tensor_image = torch.from_numpy(preproc_image)

    # detect faces
    output = model_face(tensor_image)  # predict on an image

    pred_boxes, scores = output
    pred_scores, pred_labels = torch.max(scores, dim=-1)

    # NMS 
    keep_topk = 300
    iou_threshold=0.7
    score_threshold=0.01
    for i in range(scores.shape[0]):
        score_keep = pred_scores[i] > score_threshold
        pred_box = pred_boxes[i][score_keep]
        pred_label = pred_labels[i][score_keep]
        pred_score = pred_scores[i][score_keep]

        keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, iou_threshold)
        keep = keep[: keep_topk]

        labels = pred_label[keep]
        boxes = pred_box[keep]
        scores = pred_score[keep]

        restored_box = scale_boxes_back(boxes, scale, pad, (height, width))

        for xyxy, conf, label in zip(restored_box, scores, labels):  
            x1, y1, x2, y2 = map(int, xyxy)
            print(f"Face detected: ({x1}, {y1}), ({x2}, {y2}), conf={conf:.2f}, score={label}")
            cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0,255,0), 1)
            label_text = f'{label} {conf:.2f}'
            cv2.putText(raw_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
        cv2.imshow(f'{filename}', raw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_path = os.path.join(save_dir_path, f'{filename}_pt2.jpg')
        cv2.imwrite(save_path, raw_image)


if __name__ == '__main__':
    main()