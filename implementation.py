import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model for anomaly detection in SEVs
class SEVAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SEVAnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Function to simulate dummy data for SEV sensor readings
def generate_dummy_data(num_samples, input_size):
    # Simulate normal data
    normal_data = np.random.normal(loc=0.5, scale=0.1, size=(num_samples // 2, input_size))
    # Simulate anomalous data
    anomalous_data = np.random.normal(loc=1.5, scale=0.1, size=(num_samples // 2, input_size))
    # Combine and create labels
    data = np.vstack((normal_data, anomalous_data))
    labels = np.hstack((np.zeros(num_samples // 2), np.ones(num_samples // 2)))
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Training function
def train_model(model, criterion, optimizer, data, labels, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing function
def test_model(model, data, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(data).squeeze()
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == labels).sum().item() / len(labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10  # Number of features (e.g., sensor readings)
    hidden_size = 16
    output_size = 1
    learning_rate = 0.001
    epochs = 50

    # Generate dummy data
    num_samples = 1000
    train_data, train_labels = generate_dummy_data(num_samples, input_size)
    test_data, test_labels = generate_dummy_data(num_samples // 2, input_size)

    # Initialize model, loss function, and optimizer
    model = SEVAnomalyDetector(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Training the model...")
    train_model(model, criterion, optimizer, train_data, train_labels, epochs)

    # Test the model
    print("Testing the model...")
    test_model(model, test_data, test_labels)