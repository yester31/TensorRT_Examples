# by yhpark 2025-8-26
import torch
import onnx
import os
from BEN2 import BEN2
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[TRT_E] using device: {DEVICE}")

# Print version information for debugging purposes
print(f"[TRT_E] PyTorch version: {torch.__version__}")
print(f"[TRT_E] ONNX version: {onnx.__version__}")

def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)

def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2 )
    return x

def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x

def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)

class BEN2_WRAPPER(BEN2.BEN_Base):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shallow_batch = self.shallow(x)
        glb_batch = rescale_to(x, scale_factor=0.5, interpolation='bilinear')

        loc_batch = image2patches(x)
        final_input = torch.cat((loc_batch, glb_batch), dim=0)  
            
        features = self.backbone(final_input)
        
        f4 = features[4][0:5, :, :, :]  # shape: [5, C, H, W]
        f3 = features[3][0:5, :, :, :]
        f2 = features[2][0:5, :, :, :]
        f1 = features[1][0:5, :, :, :]
        f0 = features[0][0:5, :, :, :]
        e5 = self.output5(f4)
        e4 = self.output4(f3)
        e3 = self.output3(f2)
        e2 = self.output2(f1)
        e1 = self.output1(f0)
        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)

        e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4)) 
        e4 = self.conv4(e4) 
        e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
        e3 = self.conv3(e3)
        e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
        e2 = self.conv2(e2)
        e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
        e1 = self.conv1(e1)

        loc_e1, glb_e1 = e1.split([4, 1], dim=0)

        output1_cat = patches2image(loc_e1)  # (1,128,256,256)

        # add glb feat in
        output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
        # merge
        final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
        # shallow feature merge
        final_output = final_output + resize_as(shallow_batch, final_output)
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to(final_output + resize_as(shallow_batch, final_output))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)
        mask = final_output.sigmoid()

        return mask

def main():

    input_size = 1024 # 512 or 1024
    model_name = "BEN2_Base"
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_{input_size}.onnx')
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Set up model
    model_path = f"{CUR_DIR}/BEN2/checkpoint/BEN2_Base.pth"
    print(f"[TRT_E] load model ({model_path})")
    # model = BEN2.BEN_Base().to(DEVICE).eval() #init pipeline
    model = BEN2_WRAPPER().to(DEVICE).eval() #init pipeline
    model.loadcheckpoints(model_path)

    # Get model input size from the model configuration
    dummy_input = torch.randn((1, 3, input_size, input_size), requires_grad=False).to(DEVICE)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=19, 
            input_names=["input"], 
            output_names=["output"],
            dynamo=True
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()