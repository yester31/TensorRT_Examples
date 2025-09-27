# RF-DETR: SOTA Real-Time Object Detection Model

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/lyuwenyu/RT-DETR.git
    cd RT-DETR
    conda create -n rtdetr -y python=3.11
    conda activate rtdetr
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install scikit-image
    pip install PyYAML
    pip install tensorboard
    pip install pycocotools
    pip install faster_coco_eval
    ```

2. download pretrained checkpoints.
    ```
    cd rtdetrv2_pytorch
    mkdir -p pretrained
    wget https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth -P pretrained
    cd ..
    ```

2. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export_new_ln.py
    ```

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```
    
- fp16 (RT-DETRv2-S)   
    [TRT_E] 1000 iterations time: 5.0255 [sec]  
    [TRT_E] Average FPS: 198.99 [fps]   
    [TRT_E] Average inference time: 5.03 [msec]     
    GPU mem : 228M   

- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/pdf/2304.08069)
- [RT-DETR official GitHub](https://github.com/lyuwenyu/RT-DETR)

