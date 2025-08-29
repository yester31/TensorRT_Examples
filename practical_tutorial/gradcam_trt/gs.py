# by yhpark 2024-10-04
# GradCam with TensorRT example
import torch
import torch.onnx
import onnx
import os
import timm
import onnx_graphsurgeon as gs
import numpy as np

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    
    model_name = "resnet18"
    export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}.onnx')
    
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check

    graph = gs.import_onnx(onnx_model) 
    
    identity_out = gs.Variable("Feature_map", dtype=np.float32)
    gavg_nodes = [node for node in graph.nodes if node.op == "GlobalAveragePool"]
    
    identity = gs.Node(op="Identity", inputs=gavg_nodes[0].inputs, outputs=[identity_out])
    graph.nodes.append(identity)
    graph.outputs.append(identity_out)

    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    onnx_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    onnx.checker.check_model(onnx_model)  # Perform a validity check

    export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_modified.onnx')
    onnx.save(onnx_model, export_model_path)
    
    print("ONNX model modified successful!")
    

# Run the main function
if __name__ == "__main__":
    main()