# custom layer(plugin)

## How to Run

1. build custom layer plugin

```
cd plugin_src
mkdir build
cd build
cmake ..
make

// a file 'libpreproc_layer.so' will be generated in build directory.
```

2. run test_custom_layer.py

```
pip install cuda-python
pip install tensorrt

python test_custom_layer.py
// Passed
```
