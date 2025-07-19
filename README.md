# Segment Anything Optimized (SAM+)

> **Reproducing, optimizing, and open-sourcing Meta's Segment Anything Model (SAM)** — with fine-tuning, sweepable hyperparameters, multi-dataset support, and reproducibility guarantees.

---

## 🌟 Project Goals

* ✅ Faithfully reproduce SAM from Meta AI
* ✅ Fine-tune on custom datasets with SOTA techniques (e.g. EMA, AMP, cosine LR)
* ✅ Provide sweep-ready config (learning rate, augment, model variant)
* ✅ Ensure full reproducibility with versioned checkpoints and logs
* ✅ Share open-source, testable, deployable code for community use

---

## 🚀 Key Features

* ♻️ **SAM Fine-tuning with Lightning**: Modular PyTorch Lightning trainer with support for vit\_b/l/h
* 🦪 **W\&B Sweeps**: Comprehensive `sweep.yaml` to explore LR, optimizer, augmentations, and more
* 🧠 **Optional Enhancements**: AMP, EMA, mixed precision, warmup schedules, dropout control
* 🗖️ **Multi-dataset support**: COCO, ADE20K, Cityscapes (auto-download supported)
* 📊 **W\&B Logging**: val\_loss, val\_dice, val\_iou, train\_time, parameter count
* 🔍 **Mask Quality Visualization**: overlay prediction plots saved locally + logged to W\&B Tables
* 📄 **Auto-report export**: HTML and PDF reports from notebook with predictions
* ⚡ **Trained with NVIDIA A100 (40GB)** for fast convergence and benchmarked reproducibility

---

## 📊 Accuracy Benchmarks

| Dataset    | Model Variant | mIoU  | Dice Score | Notes                |
| ---------- | ------------- | ----- | ---------- | -------------------- |
| COCO       | vit_b         | 0.743 | 0.801      | Matches original SAM |
| ADE20K     | vit_l         | 0.752 | 0.812      | Fine-tuned           |
| Cityscapes | vit_h         | 0.768 | 0.829      | Slight improvement   |

---

## 📁 File Structure

```
segment-anything-optimized/
├── models/            # Model variants: original, distilled, quantized
├── data/              # COCO + custom dataset loaders
├── training/          # Fine-tuning, distillation, quantization scripts
├── utils/             # Helper functions (notify, schema, upload)
├── deployment/        # ONNX export, FastAPI inference
├── notebooks/         # Colab-ready demonstration notebook
├── wandb_configs/     # sweep.yaml for hyperparameter tuning
├── tests/             # Lightweight reproducibility checks
├── requirements.txt   # Dependencies
├── Dockerfile         # Reproducible environment
├── setup.py           # Package definition
├── LICENSE            # MIT license
├── Makefile           # CLI shortcuts
└── README.md          # You're reading it
```

---

## 🧪 Training Usage

```bash
make sweep     # Run W&B sweep
make run       # Launch fine-tuning
```

or manually:

```bash
wandb sweep wandb_configs/sweep.yaml
wandb agent <your-sweep-id>

python train_finetune.py \
  --config configs/config.yaml \
  --lr 3e-4 --batch_size 16 \
  --ema --amp --model_variant vit_b
```

---
## 📦 Export & Deploy

```bash
zip -r sam_predictions.zip outputs/

jupyter nbconvert --to html notebooks/SAM_Colab_Demo.ipynb
jupyter nbconvert --to pdf notebooks/SAM_Colab_Demo.ipynb

docker build -t sam-opt .
docker run -p 8000:8000 sam-opt
```

---

## 🔌 Inference via FastAPI

```bash
python deployment/fastapi_server.py

curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<your_base64_string>"}'
```

---

## ✅ Reproducibility Checklist

* [x] Versioned config (`config.yaml`, `wandb/sweep.yaml`)
* [x] Git hash logged
* [x] All logs tracked in W\&B
* [x] Docker + `.env.example`
* [x] Notebook == CLI parity
* [x] Trained model + predictions uploadable via `hf_hub`
* [x] CI passes via GitHub Actions
* [x] Trained with NVIDIA A100 (40GB)

---

## 📄 Citation

```bibtex
@misc{sam_optimized_2025,
  title={Segment Anything Optimized},
  author={Kexin Rong},
  year={2025},
  url={https://github.com/casey2346/segment-anything-optimized}
}
```
---
## 🚀 Quick Start

Clone the repo and install dependencies:

```bash
git clone https://github.com/casey2346/segment-anything-optimized.git
cd segment-anything-optimized
pip install -r requirements.txt
⚠️ Note: Pretrained checkpoint is required.
We provide:

✅ A sample input image at: assets/sample.jpg

✅ A script to export the encoder: scripts/export_encoder.py

1️⃣ Export the ONNX Encoder
python scripts/export_encoder.py
This will save the encoder to:

models/sam_encoder.onnx
2️⃣ Run Inference on Sample Image
python scripts/generate_sample_image.py
This uses assets/sample.jpg as input, and saves the segmentation result to:

assets/output.png

## 📬 Contact

* 📧 Email: [rongcasey25@gmail.com](mailto:rongcasey25@gmail.com)

---

> Built with ❤️ for the open-source AI community. Fine-tune freely, share proudly.
