import onnx_tf.backend as backend
import torch
from torch import nn
from tensorflow import keras
import onnx

def convert_pytorch_model_to_tf(model_path: str):
    # Load the PyTorch model
    pytorch_model = torch.load(model_path)
    pytorch_model.eval()

    # Convert the PyTorch model to ONNX format
    dummy_input = torch.randn(1, 3, 32, 32)  # Adjust the input size as needed
    onnx_model_path = model_path.replace('.pt', '.onnx')
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path)

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert the ONNX model to TensorFlow format
    tf_rep = backend.prepare(onnx_model, device='CPU')
    return tf_rep.tensorflow_model