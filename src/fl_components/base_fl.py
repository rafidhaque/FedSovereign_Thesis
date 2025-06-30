import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy

# --- Model Definition (Simple CNN for CIFAR-10) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, 1)
        logits = self.fc_stack(x)
        return logits

# --- FL Client Definition ---
class FedAvgClient:
    def __init__(self, client_id, model, train_dataset, epochs=5, batch_size=32, learning_rate=0.01, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.model = copy.deepcopy(model).to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        # Return the model's state dictionary (weights) on the CPU
        # This ensures compatibility for aggregation, regardless of the server's device.
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

# --- FL Server Definition ---
class FedAvgServer:
    def __init__(self, global_model):
        self.global_model = copy.deepcopy(global_model)

    def aggregate_updates(self, client_updates):
        """Standard FedAvg aggregation."""
        num_clients = len(client_updates)
        # Initialize a new state_dict with zeros
        aggregated_weights = copy.deepcopy(client_updates[0])
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        # Sum up all client weights
        for update in client_updates:
            for key in aggregated_weights.keys():
                aggregated_weights[key] += update[key]

        # Average the weights
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.div(aggregated_weights[key], num_clients)
        
        self.global_model.load_state_dict(aggregated_weights)
        return self.global_model.state_dict()

    def reputation_weighted_aggregation(self, client_updates, reputations):
        """Baseline 2: Application-layer trust weighted aggregation."""
        aggregated_weights = copy.deepcopy(client_updates[0])
        total_rep_sum = sum(reputations.values())
        
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        # Perform weighted sum
        for client_id, update in client_updates.items():
            rep_weight = reputations.get(client_id, 0) / total_rep_sum
            for key in aggregated_weights.keys():
                aggregated_weights[key] += update[key] * rep_weight
                
        self.global_model.load_state_dict(aggregated_weights)
        return self.global_model.state_dict()