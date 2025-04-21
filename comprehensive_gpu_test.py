import os
import time
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

def test_pytorch():
    print("\n" + "="*50)
    print("TESTING PYTORCH GPU FUNCTIONALITY")
    print("="*50)
    
    # Basic information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for PyTorch")
        return False
    
    # GPU information
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Performance test
    print("\nRunning matrix multiplication performance test...")
    
    # Sizes to test
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        # CPU test
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # GPU test
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warmup
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # Timed run
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"Matrix size: {size}x{size}")
        print(f"  CPU time: {cpu_time:.4f} seconds")
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
    
    print("\n✅ PyTorch GPU test completed successfully!")
    return True

def test_tensorflow():
    print("\n" + "="*50)
    print("TESTING TENSORFLOW GPU FUNCTIONALITY")
    print("="*50)
    
    # Basic information
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {gpus}")
    
    if not gpus:
        print("❌ No GPUs available for TensorFlow")
        return False
    
    # Performance test
    print("\nRunning matrix multiplication performance test...")
    
    # Sizes to test
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        # CPU test
        with tf.device('/CPU:0'):
            a_cpu = tf.random.normal([size, size])
            b_cpu = tf.random.normal([size, size])
            
            start_time = time.time()
            c_cpu = tf.matmul(a_cpu, b_cpu)
            # Force execution
            c_cpu_np = c_cpu.numpy()
            cpu_time = time.time() - start_time
        
        # GPU test
        with tf.device('/GPU:0'):
            a_gpu = tf.random.normal([size, size])
            b_gpu = tf.random.normal([size, size])
            
            # Warmup
            _ = tf.matmul(a_gpu, b_gpu)
            
            start_time = time.time()
            c_gpu = tf.matmul(a_gpu, b_gpu)
            # Force execution
            c_gpu_np = c_gpu.numpy()
            gpu_time = time.time() - start_time
        
        print(f"Matrix size: {size}x{size}")
        print(f"  CPU time: {cpu_time:.4f} seconds")
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
    
    print("\n✅ TensorFlow GPU test completed successfully!")
    return True

def test_neural_network():
    print("\n" + "="*50)
    print("TESTING DEEP LEARNING WITH MNIST DATASET")
    print("="*50)
    
    # Choose framework
    if torch.cuda.is_available():
        print("Using PyTorch for neural network test")
        test_pytorch_nn()
    elif tf.config.list_physical_devices('GPU'):
        print("Using TensorFlow for neural network test")
        test_tensorflow_nn()
    else:
        print("❌ No GPU available for neural network test")

def test_pytorch_nn():
    # Only import these if needed
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(2)
            self.log_softmax = nn.LogSoftmax(dim=1)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.max_pool(x)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return self.log_softmax(x)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for just 1 epoch to test GPU
    print("\nTraining CNN on MNIST for 1 epoch...")
    start_time = time.time()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    print("\n✅ PyTorch neural network test completed successfully!")

def test_tensorflow_nn():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    
    # Define a simple CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Train for just 1 epoch to test GPU
    print("\nTraining CNN on MNIST for 1 epoch...")
    start_time = time.time()
    
    model.fit(
        x_train, y_train, 
        batch_size=64,
        epochs=1,
        validation_data=(x_test, y_test)
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    print("\n✅ TensorFlow neural network test completed successfully!")

if __name__ == "__main__":
    print("="*50)
    print("COMPREHENSIVE GPU TEST FOR DEEP LEARNING")
    print("="*50)
    
    pytorch_ok = test_pytorch()
    tensorflow_ok = test_tensorflow()
    
    if pytorch_ok or tensorflow_ok:
        test_neural_network()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"PyTorch GPU: {'✅ Working' if pytorch_ok else '❌ Not working'}")
    print(f"TensorFlow GPU: {'✅ Working' if tensorflow_ok else '❌ Not working'}")
    
    if not (pytorch_ok or tensorflow_ok):
        print("\n❌ No GPU acceleration available. Please check your installation.")
    else:
        print("\n✅ GPU acceleration is working successfully!")