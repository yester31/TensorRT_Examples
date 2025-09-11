# Train base model 

## How to Run

1. prepare imagenet100 dataset
    ```
    pip install datasets matplotlib opencv-python
    mkdir dataset
    wget https://huggingface.co/datasets/ilee0022/ImageNet100/resolve/main/label2text.json -P dataset
    ```

2. train resnet18 with imagenet100 dataset
    ```
    python train.py
    // 'best_model.pth' will be generated in checkpoint directory.
    ```

3. check a trained resnet18 with imagenet100 dataset
    ```
    python infer.py
    ```
- fp16   
    Gpu Mem: 286M   
    [TRT_E] Test Top-1 Accuracy: 84.58%   
    [TRT_E] Test Top-5 Accuracy: 97.20%   
    [TRT_E] 10000 iterations time: 24.6143 [sec]   
    [TRT_E] Average FPS: 406.27 [fps]   
    [TRT_E] Average inference time: 2.46 [msec]   