# Run YOLOX cpp demo

## How to Run

1. generate onnx file
    ```
    python ../python/onnx_export.py
    ```

2.  build tensorrt model and run
    ```
    mkdir build
    cd build
    cmake ..
    make
    ./run_detecion_demo
    ```
