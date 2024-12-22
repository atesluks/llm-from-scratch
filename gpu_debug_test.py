# Script for testing performance of CPU vs GPU (should be run on a machine with GPU)
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a larger model
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(1000, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Generate larger synthetic data
def generate_data(batch_size=100000, input_size=1000):
    X = torch.randn(batch_size, input_size)
    y = torch.randn(batch_size, 1)
    return X, y

# Training function
def train(device, model, X, y, epochs=5):
    model.to(device)
    X, y = X.to(device), y.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    end_time = time.time()

    print(f"Training Time on {device}: {end_time - start_time:.2f} seconds")

# Main function
def main():
    print(f"PyTorch version: {torch.__version__}")
    # Generate data
    X, y = generate_data()

    # Create the model
    model = LargeModel()

    # Train on CPU
    print("\n--- Training on CPU ---")
    train(torch.device('cpu'), model, X, y)

    # Train on GPU if available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print("\n--- Training on CUDA (GPU) ---")
        train(torch.device('cuda'), model, X, y)
    else:
        print("\nCUDA is not available. Please ensure you have a compatible GPU and drivers installed. Check if pytorch version is CPU only")

if __name__ == "__main__":
    main()