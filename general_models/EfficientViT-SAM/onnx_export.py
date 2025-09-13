# by yhpark 2025-9-13
# reference: https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_sam/deployment/onnx
import os 
import sys 
import torch.onnx
import onnx
from torch import nn
from onnxsim import simplify
import torch, math
import torch.nn as nn
from torchvision.transforms.functional import resize

sys.path.insert(1, os.path.join(sys.path[0], "efficientvit"))
from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

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

class GeluAsErf(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def replace_gelu_with_erf(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU):
            setattr(module, name, GeluAsErf())
        else:
            replace_gelu_with_erf(child)

class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"

class ImageEncoder(nn.Module):
    def __init__(self, model: EfficientViTSam):
        super().__init__()

        self.model = model
        replace_gelu_with_erf(self.model)
        self.image_size = self.model.image_size
        self.image_encoder = self.model.image_encoder
        self.transform = SamResize(size=self.image_size[1])

    @torch.no_grad()
    def forward(self, input_image):
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

class ImageDecoder(nn.Module):
    """
    Modified from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/onnx.py.
    """

    def __init__(self, model: EfficientViTSam, return_single_mask: bool) -> None:
        super().__init__()
        self.model = model
        self.mask_decoder = model.mask_decoder
        self.img_size = model.image_size[0]
        self.return_single_mask = return_single_mask

    @staticmethod
    def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )

        return point_embedding

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        best_idx = torch.argmax(iou_preds, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embeddings.shape[0], -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        return masks, scores

def export_image_encoder(model, model_name, onnx_sim=False):

    if model_name in ["efficientvit-sam-l0", "efficientvit-sam-l1", "efficientvit-sam-l2"]:
        image_size = [512, 512]
    elif model_name in ["efficientvit-sam-xl0", "efficientvit-sam-xl1"]:
        image_size = [1024, 1024]
    else:
        raise NotImplementedError

    save_path = os.path.join(CUR_DIR, 'onnx')
    model = ImageEncoder(model)
    model = model.eval().to(DEVICE)

    model_name = f"{model_name}_image_encoder"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print(f'[TRT_E] Export the model to onnx format : {export_model_path}')

    # dummy input
    input_image = torch.randn((1, 3, image_size[0], image_size[1]), requires_grad=True).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            input_image, 
            export_model_path, 
            opset_version=20, 
            input_names=["input_image"],
            output_names = ["image_embeddings"]
        )
        print(f"[TRT_E] onnx model exported to: {export_model_path}")

    print("[TRT_E] Validate exported onnx model")
    checker_onnx(export_model_path)

    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

def export_image_decoder(model, model_name, onnx_sim=False):

    embed_dim = model.prompt_encoder.embed_dim
    embed_size = model.prompt_encoder.image_embedding_size
    
    save_path = os.path.join(CUR_DIR, 'onnx')
    model = ImageDecoder(model, True)
    model = model.eval().to(DEVICE)

    model_name = f"{model_name}_image_decoder"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')
    print(f'[TRT_E] Export the model to onnx format : {export_model_path}')

    # dummy input
    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float).to(DEVICE)
    point_coords = torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float).to(DEVICE)
    point_labels = torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model, 
            (
                image_embeddings,
                point_coords,
                point_labels,
            ),
            export_model_path, 
            opset_version=20, 
            input_names=[
                "image_embeddings",
                "point_coords",
                "point_labels",
            ],
            output_names=["masks", "iou_predictions"],
            dynamic_axes={
                "point_coords":{0: "num_labels",1:"num_points"},
                "point_labels": {0: "num_labels",1:"num_points"},
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

    # model_name = "efficientvit-sam-xl1"
    # model_name = "efficientvit-sam-xl0"
    # model_name = "efficientvit-sam-l2"
    # model_name = "efficientvit-sam-l1"
    model_name = "efficientvit-sam-l0"
    pretrained_path = os.path.join(CUR_DIR, 'efficientvit', 'checkpoint', model_name.replace("-", "_") + ".pt")
    efficientvit_sam = create_efficientvit_sam_model(name=model_name, weight_url=pretrained_path ,pretrained=True)

    export_image_encoder(efficientvit_sam, model_name, onnx_sim=True)
    
    export_image_decoder(efficientvit_sam, model_name, onnx_sim=True)

if __name__ == '__main__':
    main()