# by yhpark 2025-9-16
import torch
import torch.onnx
import onnx
import os 
from onnxsim import simplify
import torchvision
import onnx_graphsurgeon
from collections import OrderedDict
import numpy as np

from yolox.exp import get_exp

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

class YOLOX(torch.nn.Module):
    def __init__(self, model_name, ckpt_file, class_count=80) -> None:
        super().__init__()
        exp = get_exp(None, model_name)
        model = exp.get_model()
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        self.model = model
        self.class_count = class_count
        self.test_size = exp.test_size

    def forward(self, x):
        pred: torch.Tensor =  self.model(x) # [N, 8400, 85] (cxcywh[0:4] , obj_conf[4:5], class_conf [5:85])

        # boxes
        boxes = pred[..., 0:4]  # [N, 8400, 4]
        boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # scores = objectness * class_conf
        obj_conf = pred[..., 4:5]    # [N, 8400, 1]
        class_conf = pred[..., 5:]   # [N, 8400, 80]
        scores = obj_conf * class_conf  # [N, 8400, 80]        
        return boxes, scores

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    model_name = "yolox-s"
    ckpt_file = f"{CUR_DIR}/YOLOX/pretrained/yolox_s.pth"
    model = YOLOX(model_name, ckpt_file)
    model.to(DEVICE)
    model.eval()

    input_h, input_w = model.test_size

    onnx_sim = True # True or False
    dynamic = False  # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
    batch_size = 1
    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=True).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["output"],
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)

    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

    max_output_boxes = 300
    iou_threshold = 0.45
    score_threshold = 0.25
    yolo_insert_nms(
        path=export_model_sim_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_output_boxes=max_output_boxes,
    )

if __name__ == '__main__':
    main()