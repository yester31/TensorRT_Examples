# Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/RicePasteM/Dome-DETR.git
    cd Dome-DETR
    conda create -n dome -y python=3.11
    conda activate dome
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install -r requirements.txt
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install matplotlib
    pip install scikit-image
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    wget https://huggingface.co/RicePasteM/Dome-DETR/resolve/main/pretrain_ckpts/Dome-M-VisDrone-best.pth -P pretrained
    wget https://huggingface.co/RicePasteM/Dome-DETR/resolve/main/pretrain_ckpts/Dome-L-VisDrone-best.pth -P pretrained
    wget https://huggingface.co/RicePasteM/Dome-DETR/resolve/main/pretrain_ckpts/Dome-M-AITOD-best.pth -P pretrained
    wget https://huggingface.co/RicePasteM/Dome-DETR/resolve/main/pretrain_ckpts/Dome-L-AITOD-best.pth -P pretrained
    cd ..
    ```

3. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```
- fp32 (Dome-M-AITOD)   
    - [TRT_E] 100 iterations time: 30.6983 [sec]   
    - [TRT_E] Average FPS: 3.26 [fps]   
    - [TRT_E] Average inference time: 306.98 [msec]   
    - GPU mem : 1210 MB   

## How to Run (TensorRT)
- Change code for ONNX and TensorRT
    - line 104 in zoo/dome/defe.py 
        ```
        # nn.AdaptiveAvgPool2d(1),
        nn.AvgPool2d(kernel_size=100, stride=100),
        ```

    - line 37 in zoo/dome/dynamic_nms.py
        ```
        # unique_classes = classes.unique()
        sorted_x, _ = torch.sort(classes)
        diff = sorted_x[1:] - sorted_x[:-1]
        mask = torch.cat([torch.ones(1, dtype=torch.bool, device=classes.device),diff != 0])
        unique_classes = sorted_x[mask]
        ```

    - line 261 in zoo/dome/get_roi_features.py
        ```
        def _prepare_windows(self, features, mask, n):
            B, C, H_feat, W_feat = features.shape
            H_mask, W_mask = mask.shape[-2:] 
            
            # ensure Python int (static) for ONNX export
            H_feat, W_feat = int(H_feat), int(W_feat)
            H_mask, W_mask = int(H_mask), int(W_mask)
            n = int(n)

            kernel_h = H_mask // H_feat * (H_feat // n)
            kernel_w = W_mask // W_feat * (W_feat // n)
            stride_h = kernel_h
            stride_w = kernel_w
            
            # [B, n, n, C, h_feat, w_feat]
            h_feat, w_feat = H_feat // n, W_feat // n
            windows = features.view(B, C, n, h_feat, n, w_feat).permute(0, 2, 4, 1, 3, 5)
            
            mask_float = mask.float()
            pooled_mask = F.max_pool2d(
                mask_float, 
                kernel_size=(int(kernel_h), int(kernel_w)), 
                stride=(int(stride_h), int(stride_w))
            )
            defe_mask = (pooled_mask.squeeze(1) > 0)  # [B, n, n]
            
            return windows, defe_mask
        ```

    - line 515 in zoo/dome/hybrid_encoder.py
        ```
        # defe_feature_pooled = F.adaptive_max_pool2d(defe_feature, (H // self.mwas_window_size, W // self.mwas_window_size))
        defe_feature_pooled = F.max_pool2d(defe_feature, kernel_size=40, stride=40)
        ```

    - line 521 in zoo/dome/hybrid_encoder.py
        ```
        # W, H = proj_feats[1].shape[2:]
        W, H = 100, 100
        ```

    - line 53 in zoo/doom/postprocessor.py
        ```
        # scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
        scores, index = torch.topk(scores.flatten(1), 300, dim=-1)
        ```

1. generate onnx file
    ```
    python onnx_export.py
    ```

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16 (Dome-M-AITOD)   
    - [TRT_E] 100 iterations time: 2.4697 [sec]
    - [TRT_E] Average FPS: 40.49 [fps]
    - [TRT_E] Average inference time: 24.70 [msec]
    - GPU mem : 394 MB   


- VisDrone(12)   
    - regions, pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others  
- AITOD(9)   
    - airplane, bridge, storage tank,ship, swimming pool, vehicle, person, wind mill   

- [Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection](https://arxiv.org/pdf/2505.05741)
- [Dome-DETR official GitHub](https://github.com/RicePasteM/Dome-DETR)

