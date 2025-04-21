
# ✅ Windows GPU Virtual Environment Setup Guide (PyTorch + TensorFlow DirectML)

This guide provides a clean, repeatable setup process for Windows users with NVIDIA GPUs (e.g., RTX 4050) using PyTorch with CUDA and TensorFlow with DirectML support.

---

## 🔧 Step 1: Create and Activate Virtual Environment

```bash
python -m venv venv
.env\Scriptsctivate
```

---

## 🚀 Step 2: Upgrade Core Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 🔥 Step 3: Install PyTorch with CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ⚡ Step 4: Install TensorFlow (GPU via DirectML)

```bash
pip install tensorflow-directml-plugin
```

✅ This installs TensorFlow 2.10 with DirectML GPU support.

---

## 📦 Step 5: Install Additional Common Packages (optional)

```bash
pip install matplotlib pandas scikit-learn jupyter seaborn opencv-python
```

---

## ✅ Step 6: Test GPU Availability

Create `test_gpu.py` with:

```python
import torch
import tensorflow as tf

print("=== PyTorch ===")
print("Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n=== TensorFlow ===")
print("Version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
```

Run it:

```bash
python test_gpu.py
```

Expected Output:
- PyTorch detects CUDA and your GPU (e.g., RTX 4050)
- TensorFlow shows GPU devices via DirectML

---

## 📄 Step 7: Export Your Setup (Optional)

```bash
pip freeze > requirements.txt
```

You can reuse it in the future with:

```bash
pip install -r requirements.txt
```

---

Happy coding! 🎯
