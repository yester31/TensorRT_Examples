# by yhpark 2024-10-16
# Object Detection Example with TensorRT Model Optimization
from ultralytics import YOLO
import torch.onnx
import onnx
from onnxsim import simplify
import os 
import torchvision
import onnx_graphsurgeon
from collections import OrderedDict
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

def checker_onnx(export_model_path):
    try:
        onnx_model = onnx.load(export_model_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"[MDET] failed onnx.checker.check_model() : {e}")
    finally:
        onnx.checker.check_model(export_model_path)

    for input in onnx_model.graph.input:
        print(f"[MDET] Input: {input.name}")
        for d in input.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

    for output in onnx_model.graph.output:
        print(f"[MDET] Output: {output.name}")
        for d in output.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

def simplify_onnx(export_model_path, export_model_sim_path):
    print("[MDET] Simplify exported onnx model")
    onnx_model = onnx.load(export_model_path)
    try:
        model_simplified, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("[MDET] Simplified model is invalid.")
        onnx.save(model_simplified, export_model_sim_path)
        print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
    except Exception as e:
        print(f"[MDET] simplification failed: {e}")
    checker_onnx(export_model_sim_path)

def yolo_insert_nms(
    path, score_threshold=0.01, iou_threshold=0.7, max_output_boxes=300):
    """
    http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnxops/onnx__EfficientNMS_TRT.html
    https://huggingface.co/spaces/muttalib1326/Punjabi_Character_Detection/blob/3dd1e17054c64e5f6b2254278f96cfa2bf418cd4/utils/add_nms.py
    """
    onnx_model = onnx.load(path)
    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = max_output_boxes
    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=topk,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        score_activation=False,
        box_coding=0,
    )

    outputs = [
        onnx_graphsurgeon.Variable("num_dets", np.int32, [-1, 1]),
        onnx_graphsurgeon.Variable("det_boxes", np.float32, [-1, topk, 4]),
        onnx_graphsurgeon.Variable("det_scores", np.float32, [-1, topk]),
        onnx_graphsurgeon.Variable("det_classes", np.int32, [-1, topk]),
    ]

    graph.layer(
        op="EfficientNMS_TRT",
        name="batched_nms",
        inputs=[graph.outputs[0], graph.outputs[1]],
        outputs=outputs,
        attrs=attrs,
    )

    graph.outputs = outputs
    graph.cleanup().toposort()

    filename = os.path.splitext(os.path.basename(path))[0]
    export_model_w_nms_path = f"{CUR_DIR}/onnx/{filename}_w_nms.onnx"
    onnx.save(onnx_graphsurgeon.export_onnx(graph), export_model_w_nms_path)


class YOLO11(torch.nn.Module):
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

def main():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    # 0. prepare weight & onnx file name & image size 
    batch_size = 1
    input_h, input_w = 640, 640
    # load model
    model_name = "yolo11n"
    model = YOLO11(model_name, 80)    # yolov11 model load
    model = model.eval()     

    onnx_sim = True # True or False
    model_name = f"{model_name}_{input_h}x{input_w}"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # dummy input
    dummy_input = torch.randn(batch_size, 3, 640, 640, requires_grad=True)

    # 3. generate onnx file
    with torch.no_grad():
        torch.onnx.export(model,                   # pytorch model
                        dummy_input,               # model dummy input
                        export_model_path,         # onnx model path
                        opset_version=20,          # the version of the opset
                        input_names=['input'],     # input name
                        output_names=["boxes", "scores"],   # output name
                        )  

        print("ONNX Model exported at ", export_model_path)

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)
    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

    max_output_boxes = 300
    iou_threshold = 0.7
    score_threshold = 0.01
    yolo_insert_nms(
        path=export_model_sim_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_output_boxes=max_output_boxes,
    )

if __name__ == '__main__':
    main()