import sys
import platform
import importlib

# Genel Python bilgisi
print(f"Python Sürümü: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")

# PyTorch ve CUDA bilgisi
try:
    import torch
    print(f"Torch Sürümü: {torch.__version__}")
    print(f"CUDA Kullanılabilir mi? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Sürümü: {torch.version.cuda}")
        print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch yüklü değil")

# Kütüphane sürümleri
libs = ['numpy', 'pandas', 'scikit-learn', 'joblib', 'mlflow', 'wandb']
print("\nYüklü Kütüphaneler:")
for lib in libs:
    try:
        mod = importlib.import_module(lib)
        print(f"{lib}: {mod.__version__}")
    except ImportError:
        print(f"{lib}: YÜKLÜ DEĞİL")