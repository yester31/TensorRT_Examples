import os
import time
import cv2
import torch
import numpy as np

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    model_name = "yolox-s"
    exp_file = None
    test_size = (640, 640)
    exp = get_exp(exp_file, model_name)
    model = exp.get_model()
    print("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.to(DEVICE)
    model.half()  # to FP16
    model.eval()

    ckpt_file = f"{CUR_DIR}/YOLOX/pretrained/yolox_s.pth"
    print("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")

    image_path = f"{CUR_DIR}/data/dog.jpg"
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    raw_img = img.copy()
    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])

    # preproc = ValTransform(legacy=False)
    img, _ = preproc(img, test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img = img.to(DEVICE)
    img = img.half()  # to FP16

    confthre = 0.25 
    nmsthre = 0.45 
    num_classes = exp.num_classes
    with torch.no_grad():
        t0 = time.time()
        outputs = model(img) # [1, 8400, 85], cxcywh[0:4] , class_conf[4:5], class_conf [5:85]
        outputs = postprocess(outputs, num_classes, confthre, nmsthre, class_agnostic=True)
        print("Infer time: {:.4f}s".format(time.time() - t0))

    output = outputs[0].cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    result_image = vis(raw_img, bboxes, scores, cls, confthre, COCO_CLASSES)
    save_file_name = os.path.join(save_dir_path, f"{image_file_name}_pt.jpg")
    print("Saving detection result in {}".format(save_file_name))
    cv2.imwrite(save_file_name, result_image)


if __name__ == "__main__":
    main()
