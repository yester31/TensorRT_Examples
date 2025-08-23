# by yhpark 2024-10-15
# by yhpark 2025-8-22
# TensorRT Model Optimization PTQ example
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils_tmo import *
set_random_seed()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Main training loop
def main():

    num_classes = 100
    batch_size = 1
    use_half = True

    # Define data preprocessing (for ImageNet100)
    print("[TRT_E] load dataset")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    test_loader = dataset_load(batch_size, transform_test, 'test')

    with open(f"{CUR_DIR}/dataset/label2text.json", "r", encoding="utf-8") as f:
        label2text = json.load(f)

    # Set up model
    print("[TRT_E] load model")
    model_path = f'{CUR_DIR}/checkpoint/b256_lr7.0e-04_we3_d0.3/best_model.pth'
    print(f"[TRT_E] model path: {model_path}")
    dropout, dropout_p = check_and_parse(model_path)
    model = load_model(num_classes, dropout_p).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint = remove_prefix(checkpoint, "_orig_mod.")
    model.load_state_dict(checkpoint)

    top1_acc, top5_acc, fps = test_model_topk_fps(model, test_loader, DEVICE, k=5, use_half=use_half)

    img_path = f'{CUR_DIR}/test/test_11.png'
    image = cv2.imread(img_path)  # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform_test(image)
    image = image.unsqueeze(0).to(DEVICE)

    if use_half:
        image = image.half()  # FP16 
    with torch.no_grad():
        outputs = model(image)

    max_tensor = outputs.max(dim=1)
    max_value = max_tensor[0].cpu().numpy()[0]
    max_index = max_tensor[1].cpu().numpy()[0]
    print(f'[TRT_E] max value: {max_value}')
    print(f'[TRT_E] max index: {max_index}')
    print(f'[TRT_E] max label: {label2text[str(max_index)]}')

# Execute the main function
if __name__ == "__main__":
    main()