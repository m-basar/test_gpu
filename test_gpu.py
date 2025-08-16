import pytest

# Skip the test if required libraries are missing
torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")


def test_gpu_libraries():
    print("=== PyTorch ===")
    print("Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n=== TensorFlow ===")
    print("Version:", tf.__version__)
    print("GPU Devices:", tf.config.list_physical_devices('GPU'))
