# by yhpark 2025-9-3
# Face Detection Example with TensorRT
from ultralytics import YOLO
import torch.onnx
import onnx
import os 
from onnxsim import simplify
from infer import * 
import onnx_graphsurgeon
from collections import OrderedDict

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

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    batch_size = 1
    input_h, input_w = 640, 640 
    # load model
    model_name = "yolov12n-face"
    checkpoint_path = f'{CUR_DIR}/checkpoints/{model_name}.pt'
    class_count = 1
    model = YOLOv12(checkpoint_path, class_count)
    model = model.eval()

    onnx_sim = True # True or False
    model_name = f"{model_name}_{input_h}x{input_w}"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=True)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["boxes", "scores"],
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

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