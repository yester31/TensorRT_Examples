# by yhpark 2025-9-12
# reference: https://github.com/ryouchinsa/sam-cpp-macos/blob/master/export_onnx.py
import os 
import sys 
import torch.onnx
import onnx
from torch import nn
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "sam2"))
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

def checker_onnx(export_model_path):
    try:
        onnx_model = onnx.load(export_model_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"[TRT_E] failed onnx.checker.check_model() : {e}")
    finally:
        onnx.checker.check_model(export_model_path)

    for input in onnx_model.graph.input:
        print(f"[TRT_E] Input: {input.name}")
        for d in input.type.tensor_type.shape.dim:
            print("[TRT_E] dim_value:", d.dim_value, "dim_param:", d.dim_param)

    for output in onnx_model.graph.output:
        print(f"[TRT_E] Output: {output.name}")
        for d in output.type.tensor_type.shape.dim:
            print("[TRT_E] dim_value:", d.dim_value, "dim_param:", d.dim_param)

def simplify_onnx(export_model_path, export_model_sim_path):
    print("[TRT_E] Simplify exported onnx model")
    onnx_model = onnx.load(export_model_path)
    try:
        model_simplified, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("[TRT_E] Simplified model is invalid.")
        onnx.save(model_simplified, export_model_sim_path)
        print(f"[TRT_E] simplified onnx model saved to: {export_model_sim_path}")
    except Exception as e:
        print(f"[TRT_E] simplification failed: {e}")
    checker_onnx(export_model_sim_path)

def load_sam_model(model_name, device):
    if model_name == "sam2.1_hiera_tiny":
        checkpoint = f"{CUR_DIR}/sam2/checkpoints/{model_name}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_t.yaml"
    elif model_name == "sam2.1_hiera_small":
        checkpoint = f"{CUR_DIR}/sam2/checkpoints/{model_name}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_name == "sam2.1_hiera_base_plus":
        checkpoint = f"{CUR_DIR}/sam2/checkpoints/{model_name}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif model_name == "sam2.1_hiera_large":
        checkpoint = f"{CUR_DIR}/sam2/checkpoints/{model_name}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, checkpoint, device=device)

    return sam2_model

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def forward(
        self, 
        input: torch.Tensor
    ):
        backbone_out = self.model.forward_image(input)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        return image_embeddings, high_res_features1, high_res_features2

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor, # [1, 32, 256, 256]
        high_res_features2: torch.Tensor, # [1, 64, 128, 128]
        point_coords: torch.Tensor, # [num_labels,num_points,2]
        point_labels: torch.Tensor, # [num_labels,num_points]
        mask_input: torch.Tensor,  # [1,1,256,256]
        has_mask_input: torch.Tensor,  # [1]
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        return iou_predictions, low_res_masks

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros(
            (point_coords.shape[0], 1, 2), device=point_coords.device
        )
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device
        )
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords
        )
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight
            * (point_labels == -1)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

def export_image_encoder(model, model_name, onnx_sim=False):

    save_path = os.path.join(CUR_DIR, 'onnx')
    model = ImageEncoder(model)
    model = model.eval()

    model_name = f"{model_name}_image_encoder"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[TRT_E] Export the model to onnx format')

    # dummy input
    input_image = torch.randn((1, 3, 1024, 1024), requires_grad=True).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            input_image, 
            export_model_path, 
            opset_version=20, 
            input_names=["input_image"],
            output_names = ["image_embeddings","high_res_features1","high_res_features2"]
        )
        print(f"[TRT_E] onnx model exported to: {export_model_path}")

    print("[TRT_E] Validate exported onnx model")
    checker_onnx(export_model_path)

    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

def export_image_decoder(model, model_name, onnx_sim=False):

    save_path = os.path.join(CUR_DIR, 'onnx')
    model = ImageDecoder(model)
    model = model.eval()

    model_name = f"{model_name}_image_decoder"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print('[TRT_E] Export the model to onnx format')

    # dummy input
    image_embeddings = torch.randn(1,256,64,64).to(DEVICE)
    high_res_features1 = torch.randn(1,32,256,256).to(DEVICE)
    high_res_features2 = torch.randn(1,64,128,128).to(DEVICE)
    point_coords = torch.randn(1,2,2).to(DEVICE)
    point_labels = torch.randn(1,2).to(DEVICE)
    mask_input = torch.randn(1,1,256,256, dtype=torch.float).to(DEVICE)
    has_mask_input = torch.tensor([1], dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            (
                image_embeddings,
                high_res_features1,
                high_res_features2,
                point_coords,
                point_labels,
                mask_input,
                has_mask_input,
            ),
            export_model_path, 
            opset_version=20, 
            input_names=[
                "image_embeddings",
                "high_res_features1",
                "high_res_features2",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
            ],
            output_names=["iou_predictions", "low_res_masks"],
            dynamic_axes={
                "point_coords":{0: "num_labels",1:"num_points"},
                "point_labels": {0: "num_labels",1:"num_points"},
                "mask_input": {0: "num_labels"},
                "has_mask_input": {0: "num_labels"}
            },
        )
        print(f"[TRT_E] onnx model exported to: {export_model_path}")

    print("[TRT_E] Validate exported onnx model")
    checker_onnx(export_model_path)

    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

def main():

    print('[E_TRT] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    # model_name = "sam2.1_hiera_large"
    # model_name = "sam2.1_hiera_base_plus"
    # model_name = "sam2.1_hiera_small"
    model_name = "sam2.1_hiera_tiny"
    sam2_model = load_sam_model(model_name, device=DEVICE)

    export_image_encoder(sam2_model, model_name, onnx_sim=True)
    
    export_image_decoder(sam2_model, model_name, onnx_sim=True)

if __name__ == '__main__':
    main()