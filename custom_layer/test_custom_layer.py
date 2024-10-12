import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import tensorrt as trt
import cv2
import common
from common import *
from utils import *

from load_plugin_lib import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def get_trt_plugin(plugin_name, plugin_version, plugin_namespace, input_shape):
    plugin = None
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_creator(plugin_name, plugin_version, plugin_namespace)
    if plugin_creator is not None:        
        batchSize = trt.PluginField("output_batchSize", np.array([input_shape[0]], dtype=np.int32), type=trt.PluginFieldType.INT32)
        channel = trt.PluginField("output_channel", np.array([input_shape[3]], dtype=np.int32), type=trt.PluginFieldType.INT32)
        height = trt.PluginField("output_height", np.array([input_shape[1]], dtype=np.int32), type=trt.PluginFieldType.INT32)
        width = trt.PluginField("output_width", np.array([input_shape[2]], dtype=np.int32), type=trt.PluginFieldType.INT32)
        field_collection = trt.PluginFieldCollection([batchSize, channel, height, width])
        plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection, phase=trt.TensorRTPhase.BUILD)
    return plugin


def make_trt_network_and_engine(input_shape):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    
    input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=input_shape)
    custom_plugin = get_trt_plugin("Preproc_TRT", "1", "", input_shape)
    preproc_layer = network.add_plugin_v3(inputs=[input_layer], shape_inputs=[], plugin=custom_plugin)
    preproc_layer.get_output(0).name = "outputs"
    
    network.mark_output(preproc_layer.get_output(0))

    plan = builder.build_serialized_network(network, config)
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)

    return engine


def custom_plugin_impl(input_arr, engine):
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()
    inputs[0].host = input_arr.astype(trt.nptype(trt.float32))
    trt_outputs = common.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    output = trt_outputs[0].copy()
    common.free_buffers(inputs, outputs, stream)
    return output


def main():
    load_plugin_lib()
    
    # Input
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"current directory: {current_directory}")
    img_path = os.path.join(current_directory, 'data', 'panda0.jpg')
    img = cv2.imread(img_path)  # Load sample image
    img_expanded = np.expand_dims(img, axis=0) # [224,224,3] -> [1,224,224,3]
    shape = img_expanded.shape # sample shape = [1,224,224,3]
    
    engine = make_trt_network_and_engine(shape)
    res1 = preprocess_image(img)
    res2 = custom_plugin_impl(img, engine).reshape(res1.shape)
    assert np.all(res1 == res2), f"Test failed for shape={shape}"
    print("Passed")


if __name__ == "__main__":
    main()