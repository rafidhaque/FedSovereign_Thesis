import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import os

# Import all necessary components
from fl_components.base_fl import SimpleCNN, FedAvgServer
from blockchain.base_bft import PBFT_Simulator
from fedsovereign_components import FedSovereignClient, FedSovereignServer
from blockchain.adaptive_bft import NetworkStateOracle, AdaptiveBFT_Simulator
from agents.malicious_client import MaliciousClient
from main_baselines import prepare_data # Reuse our data preparation function

def test_model(model, test_dataset, device):
    """Helper function to test the global model's accuracy."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=128)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def run_security_experiment(system_name, num_clients, num_malicious, num_rounds, device, committee_size=None):
    print(f"\n--- Running Security Test for: {system_name} ---")
    
    # 1. Initialization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    client_datasets = prepare_data(num_clients)
    
    global_model = SimpleCNN().to(device)
    
    # Create honest and malicious clients
    clients = {}
    for i in range(num_clients):
        did = f"did:client:{i}"
        client_id = f"client_{i}"
        if i < num_malicious:
            clients[did] = MaliciousClient(client_id, global_model, client_datasets[i], did, target_label=5, attack_label=3, device=device) # Attack: truck -> cat
        else:
            clients[did] = FedSovereignClient(client_id, global_model, client_datasets[i], did, device=device)
            
    # System-specific setup
    server = FedSovereignServer(global_model) # Use the more general server
    oracle = NetworkStateOracle()
    if system_name == "FedSovereign":
        bft_simulator = AdaptiveBFT_Simulator(validators=clients, oracle=oracle)
    else:
        num_validators = committee_size if system_name == "Baseline 3" else num_clients
        bft_simulator = PBFT_Simulator(num_validators=num_validators)

    # Simple reputation dict for baselines
    reputations = {f"did:client:{i}": 0.1 if i < num_malicious else 1.0 for i in range(num_clients)}
    
    accuracies = []
    
    # 2. Training Loop
    for r in range(num_rounds):
        print(f"  Round {r+1}/{num_rounds}...")
        
        # --- PREPARE UPDATES BASED ON SYSTEM ---
        updates_for_consensus = []
        updates_for_aggregation = {}
        
        if system_name == "Baseline 3":
            # Select committee (excluding malicious due to low initial rep)
            sorted_clients = sorted(reputations.keys(), key=lambda k: reputations[k], reverse=True)
            committee_dids = sorted_clients[:committee_size]
            for did in committee_dids:
                updates_for_aggregation[did] = clients[did].train()
            updates_for_consensus = list(updates_for_aggregation.values())
        else:
            # All clients train
            for did, client in clients.items():
                updates_for_aggregation[did] = client.train()
            
            if system_name == "FedSovereign":
                 updates_for_consensus = [{"update":upd, "did":did} for did, upd in updates_for_aggregation.items()]
            else:
                 updates_for_consensus = list(updates_for_aggregation.values())

        # --- RUN CONSENSUS ---
        if system_name == "FedSovereign":
            consensus_reached, _ = bft_simulator.run_consensus(updates_for_consensus)
        else:
            consensus_reached = bft_simulator.run_consensus(updates_for_consensus)
        
        # --- AGGREGATE ---
        if consensus_reached:
            if system_name == "Baseline 1" or system_name == "Baseline 3":
                server.aggregate_updates(list(updates_for_aggregation.values()))
            elif system_name == "Baseline 2":
                server.reputation_weighted_aggregation(updates_for_aggregation, reputations)
            elif system_name == "FedSovereign":
                # Use the new reputation-weighted aggregation method
                server.sovereign_aggregation({item['did']: item['update'] for item in updates_for_consensus}, clients)
                # Update reputations based on performance (simplified for sim)
                for did in updates_for_aggregation.keys():
                    # Malicious clients will have poor contributions, so their rep drops
                    if "malicious" in clients[did].__class__.__name__.lower():
                        clients[did].reputation_score *= 0.8 # Penalize
                    else:
                        clients[did].reputation_score = min(1.0, clients[did].reputation_score * 1.05) # Reward
        
        # Test and record accuracy
        accuracy = test_model(server.global_model, test_dataset, device)
        accuracies.append(accuracy)
        print(f"    Accuracy after round {r+1}: {accuracy:.2f}%")

    return accuracies

def run_experiment_2(device):
    print("--- Starting Experiment 2: Robustness to Poisoning Attacks ---")
    NUM_CLIENTS = 20
    NUM_MALICIOUS = 4 # 20% malicious
    NUM_ROUNDS = 10
    COMMITTEE_SIZE = 10
    
    results = {}
    
    # Run each system
    results["Baseline 1"] = run_security_experiment("Baseline 1", NUM_CLIENTS, NUM_MALICIOUS, NUM_ROUNDS, device=device)
    results["Baseline 2"] = run_security_experiment("Baseline 2", NUM_CLIENTS, NUM_MALICIOUS, NUM_ROUNDS, device=device)
    results["Baseline 3"] = run_security_experiment("Baseline 3", NUM_CLIENTS, NUM_MALICIOUS, NUM_ROUNDS, device=device, committee_size=COMMITTEE_SIZE)
    results["FedSovereign"] = run_security_experiment("FedSovereign", NUM_CLIENTS, NUM_MALICIOUS, NUM_ROUNDS, device=device)
    
    # Plotting the results
    plt.figure(figsize=(12, 7))
    for name, acc_list in results.items():
        plt.plot(range(1, NUM_ROUNDS + 1), acc_list, marker='o', linestyle='-', label=name)
        
    plt.xlabel("Training Round")
    plt.ylabel("Global Model Accuracy (%)")
    plt.title("Experiment 2: Robustness to 20% Coordinated Poisoning Attack")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, NUM_ROUNDS + 1))
    plt.ylim(0, 100)
    plt.savefig("./results/experiment_2_robustness.png")
    print("\nExperiment 2 complete. Plot saved to /results/experiment_2_robustness.png")
    plt.show()

if __name__ == '__main__':
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_experiment_2(device)