# by yhpark 2025-8-26
from BEN2 import BEN2
from PIL import Image
import torch
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # input
    img_path = f"{CUR_DIR}/data/test_11.png" # input image
    image = Image.open(img_path)
    
    # load model
    model = BEN2.BEN_Base().to(DEVICE).eval() #init pipeline
    model.loadcheckpoints(f"{CUR_DIR}/BEN2/checkpoint/BEN2_Base.pth")
    
    # inference
    foreground = model.inference(image) 
    
    # save
    filename = os.path.splitext(os.path.basename(img_path))[0]
    foreground.save(f"{CUR_DIR}/data/{filename}_pt.png")

if __name__ == '__main__':
    main()