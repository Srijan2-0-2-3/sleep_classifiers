from cl import FeatureDataset, Classifier
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam
from sklearn.utils import shuffle
import numpy as np
import torch.nn as nn
import torch

import warnings

warnings.filterwarnings("ignore")

# Check if a GPU is available and use it, otherwise use the CPU


from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
subject_ids = SubjectBuilder.get_all_subject_ids()
data_splits = TrainTestSplitter.leave_one_out(subject_ids)
train_set = data_splits[0].training_set
test_set = data_splits[0].testing_set

lr_values = []
losses = []

# Training loop to find the optimal learning rate for each user
for user_file in train_set:
    # Load your custom FeatureDataset for the current user
    print(user_file)
    dataset = FeatureDataset(user_file)

    # Move the model to the GPU
    model = Classifier(in_dim=4, hidden_dim=8, out_dim=6).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-7)  # Start with a small learning rate

    # Specify the range of learning rates (e.g., from 1e-6 to 1)
    lr_start = 1e-6
    lr_end = 1.0
    num_lr_steps = 100  # Number of learning rate steps to try

    # Create a list of exponentially spaced learning rates
    learning_rates = np.geomspace(lr_start, lr_end, num=num_lr_steps)

    # Training loop to find the optimal learning rate
    for lr in learning_rates:
        # Adjust the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training code using your DataLoader
        model.train()  # Ensure the model is in training mode
        running_loss = 0.0

        # Replace with your DataLoader for the current user
        your_data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for inputs, labels in your_data_loader:
            # Move data to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate the average loss for this learning rate
        avg_loss = running_loss / len(your_data_loader)

        # Store the learning rate and corresponding loss
        lr_values.append(lr)
        losses.append(avg_loss)

# Plot the learning rate vs. loss curve
plt.figure(figsize=(10, 5))
plt.semilogx(lr_values, losses)
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True)
plt.savefig('learning_rate_sleep_classifier.png')
