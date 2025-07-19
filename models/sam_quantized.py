"""
sam_quantized.py

Author: Kexin Rong(you@example.com)
License: MIT

This module provides utilities for quantizing the Segment Anything Model (SAM)
for efficient INT8 deployment on edge devices.

It supports dynamic quantization via PyTorch's built-in quantize_dynamic function,
making the model smaller and faster for CPU inference.

Advantages of quantization:
- Reduces model size (memory footprint)
- Speeds up inference on CPUs
- Enables deployment on resource-constrained devices

✅ Tested with PyTorch >= 1.10

Example usage (as Python module):

    from sam_quantized import quantize_model, load_and_predict

    # Quantize and save as PyTorch .pth
    quantize_model(
        pretrained_model_path="sam_teacher.pth",
        quantized_model_path="sam_quantized.pth",
        export_format="pth",
        log_level="DEBUG"
    )

    # Load and run inference
    import torch
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = load_and_predict("sam_quantized.pth", input_tensor)

Command-line usage:

    python sam_quantized.py --input sam_teacher.pth --output sam_quantized.pth --format pth --log-level DEBUG
    python sam_quantized.py --input sam_teacher.pth --output sam_quantized.onnx --format onnx
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import logging


def setup_logging(level: str = "INFO"):
    """
    Configures logging with the specified level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


class QuantizedSAM(nn.Module):
    """
    Quantized Segment Anything Model.
    This class wraps a pre-trained SAM model and applies dynamic quantization.
    Only nn.Linear layers are quantized to INT8 for optimal speed/size tradeoff.
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()

        if not isinstance(pretrained_model, nn.Module):
            raise TypeError("Expected a PyTorch nn.Module for pretrained_model.")

        logging.info("Applying dynamic quantization to the model...")
        self.quantized_model = torch.quantization.quantize_dynamic(
            pretrained_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        logging.info("Quantization complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_model(x)


def quantize_model(pretrained_model_path: str,
                    quantized_model_path: str,
                    export_format: str = "pth",
                    log_level: str = "INFO") -> None:
    """
    Loads a pre-trained SAM model, applies dynamic quantization,
    and saves the resulting quantized model in the specified format.

    Args:
        pretrained_model_path (str): Path to the saved original SAM model.
        quantized_model_path (str): Path where the quantized model will be saved.
        export_format (str): One of ["pth", "onnx", "torchscript"].
        log_level (str): Logging level ("DEBUG", "INFO", etc.)
    """
    setup_logging(log_level)
    logging.info(f"Loading model from '{pretrained_model_path}'...")

    if not os.path.isfile(pretrained_model_path):
        logging.error(f"File not found: {pretrained_model_path}")
        sys.exit(1)

    model = torch.load(pretrained_model_path, map_location='cpu')

    if not isinstance(model, nn.Module):
        raise TypeError("Loaded object is not a PyTorch nn.Module.")

    logging.info("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    logging.info(f"Exporting quantized model as '{export_format}'...")
    export_format = export_format.lower()

    if export_format == "pth":
        torch.save(quantized_model, quantized_model_path)
        logging.info(f"Quantized model saved to '{quantized_model_path}'")
    elif export_format == "onnx":
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            quantized_model,
            dummy_input,
            quantized_model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        logging.info(f"ONNX model saved to '{quantized_model_path}'")
    elif export_format == "torchscript":
        scripted_model = torch.jit.script(quantized_model)
        scripted_model.save(quantized_model_path)
        logging.info(f"TorchScript model saved to '{quantized_model_path}'")
    else:
        logging.error(f"Unsupported export format: {export_format}")
        sys.exit(1)


def load_and_predict(model_path: str, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Loads a quantized SAM model and runs inference on a given input.

    Args:
        model_path (str): Path to the quantized model file.
        input_tensor (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Predicted segmentation masks.
    """
    logging.info(f"Loading quantized model from '{model_path}'...")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Quantized model file not found: {model_path}")

    model = torch.load(model_path, map_location='cpu')
    if not isinstance(model, nn.Module):
        raise TypeError("Loaded object is not a PyTorch nn.Module.")

    model.eval()
    logging.info("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)

    logging.info("Inference complete.")
    return output


def run_unit_test(log_level: str = "INFO"):
    """
    Minimal unit test to verify quantization pipeline.
    """
    setup_logging(log_level)
    logging.info("Running unit test...")

    class DummySAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    dummy_model = DummySAM()
    test_input = torch.randn(1, 3, 224, 224)

    quantized = QuantizedSAM(dummy_model)
    output = quantized(test_input)
    assert output.shape[1] == 10

    logging.info("✅ Unit test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize a pre-trained SAM model for efficient INT8 deployment."
    )
    parser.add_argument(
        "--input", type=str, required=False,
        help="Path to the pre-trained SAM model (.pth)"
    )
    parser.add_argument(
        "--output", type=str, required=False,
        help="Path to save the quantized model (.pth / .onnx / .pt)"
    )
    parser.add_argument(
        "--format", type=str, default="pth", choices=["pth", "onnx", "torchscript"],
        help="Export format for the quantized model"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (e.g., DEBUG, INFO)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run unit test instead of quantization"
    )
    args = parser.parse_args()

    if args.test:
        run_unit_test(args.log_level)
    else:
        if not args.input or not args.output:
            logging.error("Both --input and --output are required unless --test is used.")
            sys.exit(1)
        quantize_model(args.input, args.output, args.format, args.log_level)
