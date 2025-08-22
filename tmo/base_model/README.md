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
