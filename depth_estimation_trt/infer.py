from PIL import Image
import depth_pro
import os
import sys
import torch
import cv2
import numpy as np
import onnx

sys.path.insert(1, os.path.join(sys.path[0], "ml-depth-pro"))

current_directory = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing transform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()
print((model.img_size, model.img_size))

# Load and preprocess an image.
image_file_name = 'example.jpg'
image_path = os.path.join(current_directory, 'ml-depth-pro', 'data', image_file_name)
    
image0, _, f_px = depth_pro.load_rgb(image_path) # RGB
image = transform(image0)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
print(np.array(focallength_px.cpu())) # 3362.1047

activation_map = np.array(depth.cpu())
activation_map = (activation_map - np.min(activation_map)) / np.max(activation_map)
heat_map = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET) # hw -> hwc
save_path = os.path.join(current_directory, 'save', f'{os.path.splitext(image_file_name)[0]}_depth_Torch.jpg')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, heat_map)