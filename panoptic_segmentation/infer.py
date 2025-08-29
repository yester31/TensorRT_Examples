# by yhpark 2025-8-29
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import EomtForUniversalSegmentation, AutoImageProcessor
import os 
import cv2
import numpy as np
from torch.nn.functional import interpolate

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 133
stuff_classes = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
mask_thresh = 0.8
overlap_thresh = 0.8

def main_ori() : 
    model_name = "coco_panoptic_eomt_large_640"
    model_name = "coco_panoptic_eomt_small_640_2x"
    model_name = "coco_panoptic_eomt_base_640_2x"
    model_id = f"tue-mps/{model_name}"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = EomtForUniversalSegmentation.from_pretrained(model_id).to(DEVICE)

    file_path = f"{CUR_DIR}/data/000000039769.jpg"
    image = Image.open(file_path)

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)

    original_image_sizes = [(image.height, image.width)]
    preds = processor.post_process_panoptic_segmentation(outputs, original_image_sizes)
    print(outputs.keys())

    plt.imshow(preds[0]["segmentation"].cpu())
    plt.axis("off")
    plt.title("Panoptic Segmentation")
    plt.show()

def transform_cv(image_, ref_size=640):   
    image = image_.copy() 
    ori_h, ori_w = image.shape[:2] # [h, w, c]
    factor = min(ref_size/ori_h, ref_size/ori_w)
    new_h = round(ori_h * factor)
    new_w = round(ori_w * factor)

    # 1) BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2) Resize
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3) ToTensor (0~1 float, pad, HWC -> CHW,)
    resized_img = resized_img.astype(np.float32) / 255.0

    pad_h = max(0, ref_size - new_h)
    pad_w = max(0, ref_size - new_w)

    padded_img = cv2.copyMakeBorder(
        resized_img,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # F.pad default=0
    )

    padded_img = np.transpose(padded_img, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # Add batch dimension (C, H, W) -> (1, C, H, W)
    tensor = np.expand_dims(padded_img, axis=0)
    # Return as NumPy array (C-order)   
    return np.array(tensor, dtype=np.float32, order="C")

def scale_img_size_instance_panoptic(size: tuple[int, int]):
    factor = min(
        640 / size[0],
        640 / size[1],
    )
    return [round(s * factor) for s in size]

def revert_resize_and_pad_logits_instance_panoptic(transformed_logits, img_sizes):
    logits = []
    for i in range(len(transformed_logits)):
        scaled_size = scale_img_size_instance_panoptic(img_sizes[i])

        logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
        logits_i = interpolate(
            logits_i[None, ...],
            img_sizes[i],
            mode="bilinear",
        )[0]
        logits.append(logits_i)

    return logits

def to_per_pixel_preds_panoptic(mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh):
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    preds_list = []

    for i in range(len(mask_logits_list)):
        preds = -torch.ones(
            (*mask_logits_list[i].shape[-2:], 2),
            dtype=torch.long,
            device=class_logits.device,
        )
        preds[:, :, 0] = num_classes

        keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
        if not keep.any():
            preds_list.append(preds)
            continue

        masks = mask_logits_list[i].sigmoid()
        segments = -torch.ones(
            *masks.shape[-2:],
            dtype=torch.long,
            device=class_logits.device,
        )

        mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
        stuff_segment_ids, segment_id = {}, 0
        segment_and_class_ids = []

        for k, class_id in enumerate(classes[i][keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = orig_mask & new_mask

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum().item()
            if (
                orig_area == 0
                or new_area == 0
                or final_area == 0
                or new_area / orig_area < overlap_thresh
            ):
                continue

            if class_id in stuff_classes:
                if class_id in stuff_segment_ids:
                    segments[final_mask] = stuff_segment_ids[class_id]
                    continue
                else:
                    stuff_segment_ids[class_id] = segment_id

            segments[final_mask] = segment_id
            segment_and_class_ids.append((segment_id, class_id))

            segment_id += 1

        for segment_id, class_id in segment_and_class_ids:
            segment_mask = segments == segment_id
            preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
            preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

        preds_list.append(preds)

    return preds_list

def draw_black_border(sem, inst, mapping):
    h, w = sem.shape
    out = np.zeros((h, w, 3))
    for s in np.unique(sem):
        out[sem == s] = mapping[s]

    combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out

def plot_panoptic_results(img, sem_pred, inst_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_ids = np.unique(sem_pred)
    mapping = {
        s: (
            [0, 0, 0]
            if s == -1 or s == num_classes
            else plt.cm.hsv(i / len(all_ids))[:3]
        )
        for i, s in enumerate(all_ids)
    }

    vis_pred = draw_black_border(sem_pred, inst_pred, mapping)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[1].imshow(vis_pred)
    axes[1].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # save image
    plt.show()
    plt.close()  # close figure to free memory

def main() : 
    model_name = "coco_panoptic_eomt_large_640"
    model_name = "coco_panoptic_eomt_small_640_2x"
    model_name = "coco_panoptic_eomt_base_640_2x"
    model_id = f"tue-mps/{model_name}"
    model = EomtForUniversalSegmentation.from_pretrained(model_id).to(DEVICE)

    file_path = f"{CUR_DIR}/data/000000039769.jpg"
    image = cv2.imread(file_path)
    ori_h, ori_w = image.shape[:2] # [h, w, c]
    np_inputs = transform_cv(image)
    inputs = torch.from_numpy(np_inputs).to(DEVICE)

    with torch.inference_mode(): 
        outputs = model(inputs)

    mask_logits_per_layer, class_logits_per_layer = outputs['masks_queries_logits'], outputs['class_queries_logits']
    mask_logits = interpolate(mask_logits_per_layer, (640, 640), mode="bilinear")
    mask_logits = revert_resize_and_pad_logits_instance_panoptic(mask_logits, [(ori_h, ori_w)])
    preds = to_per_pixel_preds_panoptic(mask_logits, class_logits_per_layer, stuff_classes, mask_thresh, overlap_thresh)

    pred = preds[0].cpu().numpy()
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_panoptic_results(image, sem_pred, inst_pred, f"{CUR_DIR}/results/{model_name}_000000039769_pt.jpg")

def get_test_image():
    file_path = f"{CUR_DIR}/data/000000039769.jpg"
    if os.path.exists(file_path):
        print(f"file exist: ({file_path})")
    else :
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        cv2.imwrite(file_path, img_cv2)

if __name__ == '__main__':
    # get_test_image()
    main()
    main_ori()