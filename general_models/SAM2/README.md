# SAM 2: Segment Anything in Images and Videos

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    conda create -n sam2 -y python=3.11
    conda activate sam2
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install -e .
    pip install -e ".[notebooks]"
    pip install opencv-python matplotlib
    ```

2. download pretrained checkpoints.
    ```
    cd checkpoints
    ./download_ckpts.sh
    cd ..
    ```

3. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```

[TRT_E] 1000 iterations time: 7.5730 [sec]   
[TRT_E] Average FPS: 13.20 [fps]   
[TRT_E] Average inference time: 75.73 [msec]   
GPU Mem : 790M   

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```
- image_encoder
    - input : 
                input[1,3,1024,1024]
    - output : 
                image_embeddings[1,256,64,64], 
                high_res_features1[1,32,256,256], 
                high_res_features2[1,64,128,128]

- image_decoder
    - input : 
                image_embeddings[1,256,64,64], 
                high_res_features1[1,32,256,256], 
                high_res_features2[1,64,128,128]
                point_coords[num_labels,num_points,2], 
                point_labels[num_labels,num_points], 
                mask_input[num_labels,1,256,256], 
                has_mask_input[num_labels], 
    - ouput : 
                iou_predictions, 
                low_res_masks

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```

[TRT_E] 1000 iterations time: 0.4075 [sec]   
[TRT_E] Average FPS: 24.54 [fps]   
[TRT_E] Average inference time: 40.75 [msec]  
GPU Mem : 762M   

- [SAM2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [SAM2 official GitHub](https://github.com/facebookresearch/sam2)

