# by yhpark 2025-9-5
from infer import *
import torch.onnx
import onnx
import os 
# from onnxsim import simplify

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
'''
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
'''

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    batch_size = 1
    input_h, input_w = 640, 640 

    # model_name = "dfine_n_coco"
    # config = f"{CUR_DIR}/D-FINE/configs/dfine/dfine_hgnetv2_n_coco.yml"
    # resume = f"{CUR_DIR}/D-FINE/checkpoints/dfine_n_coco.pth"
    model_name = "dfine_s_obj2coco"
    config = f"{CUR_DIR}/D-FINE/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml"
    resume = f"{CUR_DIR}/D-FINE/checkpoints/{model_name}.pth"
    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(DEVICE)


    dynamo = True   # True or False
    onnx_sim = False # True or False
    dynamic = False  # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=True).to(DEVICE)
    dummy_input2 = torch.tensor([[input_h, input_w]]).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy_input, dummy_input2), 
            export_model_path, 
            opset_version=20, 
            input_names=["input", "ori_size"],
            output_names=["labels", "boxes", "scores"],
            dynamo=dynamo,
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)

    # if onnx_sim :
    #     export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
    #     simplify_onnx(export_model_path, export_model_sim_path)


if __name__ == '__main__':
    main()