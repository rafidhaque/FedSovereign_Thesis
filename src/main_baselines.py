import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

# Import our custom components
from fl_components.base_fl import SimpleCNN, FedAvgClient, FedAvgServer
from blockchain.base_bft import PBFT_Simulator

def prepare_data(num_clients):
    """Download CIFAR-10 and split it among clients."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split dataset into non-IID shards for each client
    # For simplicity, we'll do a simple split here. A proper non-IID split is more complex.
    data_indices = list(range(len(train_dataset)))
    np.random.shuffle(data_indices)
    split_size = len(train_dataset) // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size
        subset_indices = data_indices[start_idx:end_idx]
        client_datasets.append(Subset(train_dataset, subset_indices))
        
    return client_datasets

def run_baseline_1(num_clients, num_rounds, device):
    print("\n--- Running Baseline 1: Standard BCFL ---")
    
    # 1. Initialization
    client_datasets = prepare_data(num_clients)
    global_model = SimpleCNN().to(device)
    server = FedAvgServer(global_model)
    bft_simulator = PBFT_Simulator()
    clients = [FedAvgClient(f"client_{i}", global_model, client_datasets[i], device=device) for i in range(num_clients)]
    
    # 2. Training Loop
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}...")
        
        # Clients train locally
        local_updates = [client.train() for client in clients]
        
        # Server submits updates to blockchain for consensus
        consensus_reached = bft_simulator.run_consensus(local_updates)
        
        # If consensus reached, server aggregates
        if consensus_reached:
            server.aggregate_updates(local_updates)
            print("   Consensus reached. Global model updated.")
        else:
            print("   Consensus failed. Global model not updated.")

def run_baseline_2(num_clients, num_rounds, device):
    print("\n--- Running Baseline 2: Application-Layer Trust ---")
    
    # 1. Initialization
    client_datasets = prepare_data(num_clients)
    global_model = SimpleCNN().to(device)
    server = FedAvgServer(global_model)
    bft_simulator = PBFT_Simulator()
    clients = {f"client_{i}": FedAvgClient(f"client_{i}", global_model, client_datasets[i], device=device) for i in range(num_clients)}
    
    # Simple reputation dict: higher reputation for clients with lower index
    reputations = {f"client_{i}": 1.0 - (i * 0.05) for i in range(num_clients)}
    
    # 2. Training Loop
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}...")
        
        local_updates = {cid: client.train() for cid, client in clients.items()}
        
        consensus_reached = bft_simulator.run_consensus(list(local_updates.values()))
        
        if consensus_reached:
            # Here, the server executes application-layer logic after consensus
            server.reputation_weighted_aggregation(local_updates, reputations)
            print("   Consensus reached. Reputation-weighted aggregation performed.")
        else:
            print("   Consensus failed. Global model not updated.")

def run_baseline_3(num_clients, committee_size, num_rounds, device):
    print("\n--- Running Baseline 3: Pre-Consensus Filtering ---")
    
    # 1. Initialization
    client_datasets = prepare_data(num_clients)
    global_model = SimpleCNN().to(device)
    server = FedAvgServer(global_model)
    bft_simulator = PBFT_Simulator(num_validators=committee_size) # Consensus only among the committee
    all_clients = {f"client_{i}": FedAvgClient(f"client_{i}", global_model, client_datasets[i], device=device) for i in range(num_clients)}
    
    # Simple reputation dict
    reputations = {f"client_{i}": 1.0 - (i * 0.05) for i in range(num_clients)}
    
    # 2. Training Loop
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}...")
        
        # --- Pre-Consensus Filtering Step ---
        # Select committee based on highest reputation
        sorted_clients = sorted(reputations.keys(), key=lambda k: reputations[k], reverse=True)
        committee_ids = sorted_clients[:committee_size]
        print(f"   Selected committee of {committee_size} clients.")
        
        # Only committee members train
        committee_updates = [all_clients[cid].train() for cid in committee_ids]
        
        # Submit committee updates for consensus
        consensus_reached = bft_simulator.run_consensus(committee_updates)
        
        if consensus_reached:
            server.aggregate_updates(committee_updates)
            print("   Consensus reached. Global model updated by committee.")
        else:
            print("   Consensus failed. Global model not updated.")
            
if __name__ == '__main__':
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_CLIENTS = 10
    NUM_ROUNDS = 3
    COMMITTEE_SIZE = 5

    run_baseline_1(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, device=device)
    run_baseline_2(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, device=device)
    run_baseline_3(num_clients=NUM_CLIENTS, committee_size=COMMITTEE_SIZE, num_rounds=NUM_ROUNDS, device=device)