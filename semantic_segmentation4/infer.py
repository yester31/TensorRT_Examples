# by yhpark 2025-8-26
from ormbg.ormbg.models.ormbg import ORMBG
from PIL import Image
import torch
import os
import numpy as np
from PIL import Image
from skimage import io
import torch.nn.functional as F

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def main():
    # input
    img_path = f"{CUR_DIR}/data/pexels-photo-5965592.png" # input image
    filename = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path)
    
    # load model
    model = ORMBG()
    model_path = f"{CUR_DIR}/ormbg/checkpoint/ormbg.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE).eval()

    # inference
    model_input_size = 1024
    orig_image = io.imread(img_path)
    orig_im_size = orig_image.shape[0:2]

    image = preprocess_image(orig_image, model_input_size).to(DEVICE)
    result = model(image)
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save
    pil_im = Image.fromarray(result_image)
    if pil_im.mode == "RGBA":
        pil_im = pil_im.convert("RGB")

    bg_rgba = (0, 0, 0, 0)
    no_bg_image = Image.new("RGBA", pil_im.size, bg_rgba)
    orig_image = Image.open(img_path)
    no_bg_image.paste(orig_image, mask=pil_im)

    alpha_image = pil_im.convert("L")
    alpha_image.save(f"{CUR_DIR}/save/{filename}_mask_pt.png")

    no_bg_image = no_bg_image.convert("RGB")
    no_bg_image.save(f"{CUR_DIR}/save/{filename}_fg_pt.png")

if __name__ == '__main__':
    main()