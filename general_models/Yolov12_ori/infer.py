# by yhpark 2025-9-3
import torch
from ultralytics import YOLO
import cv2
import os 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main() :
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    image_path = f"{CUR_DIR}/data/test1.jpg"
    # image_path = f"{CUR_DIR}/data/test2.jpg"
    filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    # detect faces
    model_face = YOLO(f'{CUR_DIR}/checkpoints/yolov12n-face.pt')  # load a pretrained YOLOv8n detection model
    results = model_face(image)  # predict on an image
    result = results[0]

    result_image = result.plot()
    cv2.imshow(f'{filename}', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    confs = result.boxes.conf
    xyxys = result.boxes.xyxy
    for xyxy, conf in zip(xyxys, confs):  
        x1, y1, x2, y2 = map(int, xyxy)
        print(f"Face detected: ({x1}, {y1}), ({x2}, {y2}), conf={conf:.2f}")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    save_path1 = os.path.join(save_dir_path, f'{filename}_pt1.jpg')
    # cv2.imwrite(save_path1, image)

    save_path = os.path.join(save_dir_path, f'{filename}_pt.jpg')
    cv2.imwrite(save_path, result_image)


if __name__ == '__main__':
    main()