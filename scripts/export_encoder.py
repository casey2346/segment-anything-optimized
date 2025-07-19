import torch
import os
from segment_anything import sam_model_registry

checkpoint_path = "models/sam_vit_b.pth"
output_onnx_path = "models/sam_encoder.onnx"

class SAMEncoderWrapper(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder

    def forward(self, x):
        return self.image_encoder(x)

print("📦 Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.eval()

print("📦 Wrapping encoder...")
model = SAMEncoderWrapper(sam)
model.eval()

dummy_input = torch.randn(1, 3, 1024, 1024)

print("🚀 Exporting encoder to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    output_onnx_path,
    input_names=["image"],
    output_names=["image_embeddings"],
    opset_version=17,
    verbose=True
)

print(f"✅ Exported encoder to {output_onnx_path}")
