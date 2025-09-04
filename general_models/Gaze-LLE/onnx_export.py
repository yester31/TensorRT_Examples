# by yhpark 2025-9-4
import torch.onnx
import onnx
import os 
import torch
import sys
sys.path.insert(1, os.path.join(sys.path[0], "gazelle"))
from gazelle.gazelle.model import *

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

class GazeLLE_Wrapper1(GazeLLE):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__(backbone, inout, dim, num_layers, in_size, out_size)

    def forward(self, images):
        x = self.backbone.forward(images)
        x = self.linear(x)
        x = x + self.pos_embed
        return x 

def gazelle_dinov2_vitb14_inout_1():
    backbone = DinoV2Backbone('dinov2_vitb14')
    model = GazeLLE_Wrapper1(backbone, inout=True)
    return model

class GazeLLE_Wrapper2(GazeLLE):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__(backbone, inout, dim, num_layers, in_size, out_size)

    def forward(self, x, head_maps):

        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        heatmap = torchvision.transforms.functional.resize(x, self.out_size)
        # heatmap_preds = utils.split_tensors(x, num_ppl_per_img) # resplit per image

        return heatmap, inout_preds

def gazelle_dinov2_vitb14_inout_2():
    backbone = DinoV2Backbone('dinov2_vitb14')
    model = GazeLLE_Wrapper2(backbone, inout=True)
    return model

def export_gazelle_2():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    model_name = "gazelle_dinov2_vitb14_inout"
    model = gazelle_dinov2_vitb14_inout_2()
    model.load_gazelle_state_dict(torch.load(f'{CUR_DIR}/checkpoints/{model_name}.pt', weights_only=True))
    model.eval().to(DEVICE)
    model_name = f"{model_name}_2"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
    face_count = 3
    x = torch.randn((face_count, 256, 32, 32), requires_grad=True).to(DEVICE)
    head_maps = torch.randn((face_count, 32, 32), requires_grad=True).to(DEVICE)
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (x, head_maps), 
            export_model_path, 
            opset_version=20, 
            input_names=["x", "head_maps"],
            output_names=["heatmap", "inout_preds"],
            dynamic_axes={
                "x": {0: "face_count"}, 
                "head_maps": {0: "face_count"}, 
                "heatmap": {0: "face_count"},
                "inout_preds": {0: "face_count"}
                } 
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)

def export_gazelle_1():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    batch_size = 1
    input_h, input_w = 448, 448 
    model_name = "gazelle_dinov2_vitb14_inout"
    model = gazelle_dinov2_vitb14_inout_1()
    model.load_gazelle_state_dict(torch.load(f'{CUR_DIR}/checkpoints/{model_name}.pt', weights_only=True))
    model.eval().to(DEVICE)
    model_name = f"{model_name}_1"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[MDET] Export the model to onnx format')

    # dummy input
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


if __name__ == '__main__':
    export_gazelle_1()
    export_gazelle_2()