# 🧠 Pruned U-Net for Biomedical Image Segmentation

This project compresses a U-Net model by over 97% using **structured pruning** and **quantization**, while maintaining nearly the same segmentation accuracy.

## 🧬 Goal

Shrink U-Net for real-time use in clinics and edge devices like Raspberry Pi — without losing accuracy on cell segmentation tasks.

## 🚀 Features

- 🔧 Structured pruning (L2 norm-based)
- ⚖️ Fine-tuning for performance recovery
- 🧮 8-bit and 4-bit quantization
- 📦 Model size: 7.85M → 215K parameters
- ⚡ Inference time: 38ms → 6ms

## 📊 Results

| Metric           | Baseline U-Net | Pruned U-Net |
|------------------|----------------|--------------|
| Parameters        | 7.85M          | 215K         |
| Inference Time    | 38ms           | 6ms          |
| IoU               | 0.7459         | 0.7432       |
| Dice Score        | 0.852          | 0.849        |

## 🗂 Files

- `model.ipynb` – Full implementation
- `outputs/` – Visual comparison of predicted vs ground truth masks
- `requirements.txt` – Dependencies
- `report.pdf` – (Optional) Full report with architecture and results

## 🧠 Tech Stack

- Python, TensorFlow, Keras
- OpenCV, NumPy, matplotlib

## 📄 License

MIT License
