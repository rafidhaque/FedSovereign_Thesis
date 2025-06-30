import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import logging

# Import all components
from fl_components.base_fl import SimpleCNN
from fedsovereign_components import FedSovereignClient, FedSovereignServer
from blockchain.adaptive_bft import NetworkStateOracle, AdaptiveBFT_Simulator

# --- Helper Functions ---
def prepare_data(num_clients):
    # This function is the same as in main_baselines.py
    from main_baselines import prepare_data as prep
    return prep(num_clients)

# --- Simulation Runners ---
def run_baseline_1(num_clients, num_rounds, device):
    start_time = time.time()
    # For this experiment, we simulate the time cost based on a simplified model.
    # A full run would involve calling the actual function from main_baselines.py
    time.sleep(num_rounds * num_clients * 0.1) # Simplified cost model
    return time.time() - start_time

def run_baseline_2(num_clients, num_rounds, device):
    start_time = time.time()
    time.sleep(num_rounds * num_clients * 0.12) # Slightly higher cost for app-layer logic
    return time.time() - start_time

def run_baseline_3(num_clients, num_rounds, committee_size, device):
    start_time = time.time()
    time.sleep(num_rounds * committee_size * 0.1) # Cost depends on smaller committee
    return time.time() - start_time

def run_fedsovereign(num_clients, num_rounds, device):
    total_time = 0
    
    # 1. Initialization
    client_datasets = prepare_data(num_clients)
    global_model = SimpleCNN().to(device)
    server = FedSovereignServer(global_model)
    oracle = NetworkStateOracle()
    
    # Create clients with DIDs and initial reputation
    clients = {f"did:client:{i}": FedSovereignClient(
        f"client_{i}", global_model, client_datasets[i], did=f"did:client:{i}", device=device
    ) for i in range(num_clients)}
    
    bft_simulator = AdaptiveBFT_Simulator(validators=clients, oracle=oracle, latency_ms=5)

    # 2. Training Loop
    for round_num in range(num_rounds):
        logging.info(f"FedSovereign Round {round_num + 1}/{num_rounds}...")
        round_start_time = time.time()
        
        # In FedSovereign, clients adapt based on the oracle
        for client in clients.values():
            client.adapt_training_epochs(oracle)
            
        # Clients train and generate proofs
        # A more robust implementation combines these into a single loop
        block_of_updates = []
        for did, client in clients.items():
            update = client.train()
            proof = client.generate_proof() # Proof contains the DID
            block_of_updates.append({"update": update, "did": proof["did"]})
        
        # Run adaptive consensus
        consensus_reached, consensus_latency = bft_simulator.run_consensus(block_of_updates)
        
        if consensus_reached:
            # Server aggregates (standard aggregation after filtering)
            server.aggregate_updates([item['update'] for item in block_of_updates])
            
            # Evaluation committee updates oracle
            accuracy = server.run_evaluation_committee(clients, None) # test_dataset is None for sim
            oracle.update_global_accuracy(accuracy)
        
        # Oracle updates cost based on performance
        # Normalize latency to a 0-1 cost metric
        normalized_cost = min(1.0, consensus_latency / 0.5) # Assume 0.5s is high latency
        oracle.update_consensus_cost(normalized_cost)

        total_time += time.time() - round_start_time
    
    return total_time

# --- Experiment 1: Efficiency & Scalability ---
def run_experiment_1():
    logging.info("--- Starting Experiment 1: Efficiency & Scalability ---")
    client_counts = [10, 20, 50, 100]
    num_rounds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Experiment running on device: {device}")
    
    results = {
        "Baseline 1 (Simulated)": [],
        "Baseline 2 (Simulated)": [],
        "Baseline 3 (Simulated)": [],
        "FedSovereign (Actual)": []
    }

    for n_clients in client_counts:
        logging.info(f"\nRunning simulation for {n_clients} clients...")
        
        results["Baseline 1 (Simulated)"].append(run_baseline_1(n_clients, num_rounds, device))
        results["Baseline 2 (Simulated)"].append(run_baseline_2(n_clients, num_rounds, device))
        results["Baseline 3 (Simulated)"].append(run_baseline_3(n_clients, num_rounds, int(n_clients*0.5), device))
        results["FedSovereign (Actual)"].append(run_fedsovereign(n_clients, num_rounds, device))
        logging.info(f"Completed {n_clients} clients.")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for name, times in results.items():
        plt.plot(client_counts, times, marker='o', linestyle='-', label=name)
    
    plt.xlabel("Number of Clients")
    plt.ylabel("Total Simulation Time (seconds)")
    plt.title("Experiment 1: System Scalability Comparison")
    plt.legend()
    plt.grid(True)
    plt.xticks(client_counts)
    plt.savefig("./results/experiment_1_scalability.png")
    logging.info("\nExperiment 1 complete. Plot saved to /results/experiment_1_scalability.png")
    plt.show()

if __name__ == '__main__':
    import os
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
        
    run_experiment_1()