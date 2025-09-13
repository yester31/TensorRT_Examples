# EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/mit-han-lab/efficientvit.git
    cd efficientvit
    conda create -n efficientvit -y python=3.11
    conda activate efficientvit
    pip install -U -r requirements.txt
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p checkpoint
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt -P checkpoint
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt -P checkpoint
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt -P checkpoint
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt -P checkpoint
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt -P checkpoint
    ```

3. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```
    - efficientvit-sam-l0      
        1000 iterations time: 42.9617 [sec]   
        Average FPS: 23.28 [fps]   
        Average inference time: 42.96 [msec]   
        GPU Mem : 382M   

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```
- image_encoder
    - input : 
                input[1,3,512,512]
    - output : 
                image_embeddings[1,256,64,64], 

- image_decoder
    - input : 
                image_embeddings[1,256,64,64], 
                point_coords[num_labels,num_points,2], 
                point_labels[num_labels,num_points], 
    - ouput : 
                masks,
                iou_predictions

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```
    - efficientvit-sam-l0      
        1000 iterations time: 11.1881 [sec]   
        Average FPS: 89.38 [fps]   
        Average inference time: 11.19 [msec]    
        GPU Mem : 330M    

- [EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss](https://arxiv.org/pdf/2402.05008)
- [efficientvit official GitHub](https://github.com/mit-han-lab/efficientvit)
- [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_sam)

