# by yhpark 2025-9-27
import torch.onnx
import onnx
import os 
import sys
from onnxsim import simplify
import torch.nn as nn
import torchvision

sys.path.insert(1, os.path.join(sys.path[0], "rf-detr"))
from rfdetr import RFDETRBase, RFDETRNano

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

class Model(nn.Module):
    def __init__(self, model, num_select=300):
        super().__init__()
        model = model.model.model
        model.export()  
        self.model = model
        self.num_select = num_select

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        
        out_logits, out_bbox = outputs[1], outputs[0]
        # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = torchvision.ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return labels, boxes, scores
    
def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    batch_size = 1
    input_h, input_w = 384, 384 
    model_name = "rf_detr_nano"  # rf_detr_base or rf_detr_nano
    model = RFDETRNano()
    model = Model(model)
    model = model.eval().to(DEVICE)
    
    onnx_sim = True # True or False
    dynamic = False  # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=True).to(DEVICE)
    dummy_input2 = torch.tensor([[input_w, input_h]]).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy_input, dummy_input2), 
            export_model_path, 
            opset_version=20, 
            input_names=["input", "ori_size"],
            output_names=["labels", "boxes", "scores"]
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)

    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)


if __name__ == '__main__':
    main()