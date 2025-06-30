import torch
import logging
import numpy as np

# Import all components
from fl_components.base_fl import SimpleCNN
from fedsovereign_components import FedSovereignClient, FedSovereignServer
from blockchain.adaptive_bft import NetworkStateOracle, AdaptiveBFT_Simulator
from main_baselines import prepare_data

def run_resilience_experiment(device):
    """
    Experiment Extra 1: System Resilience to Low-Reputation Clients.
    This experiment tests if the RA-dBFT mechanism correctly rejects updates
    from a coalition of low-reputation clients.
    """
    logging.info("--- Starting Experiment Extra 1: System Resilience ---")

    # 1. Initialization
    num_honest_clients = 7
    num_malicious_clients = 3 # A minority coalition
    total_clients = num_honest_clients + num_malicious_clients
    
    client_datasets = prepare_data(total_clients)
    global_model = SimpleCNN().to(device)
    server = FedSovereignServer(global_model)
    oracle = NetworkStateOracle()
    
    # Create clients and assign reputations
    all_clients = {}
    honest_clients = {}
    malicious_clients = {}

    for i in range(total_clients):
        did = f"did:client:{i}"
        client = FedSovereignClient(
            f"client_{i}", global_model, client_datasets[i], did=did, device=device
        )
        if i < num_honest_clients:
            client.reputation_score = 1.0  # High reputation
            honest_clients[did] = client
        else:
            client.reputation_score = 0.1  # Very low reputation
            malicious_clients[did] = client
        all_clients[did] = client

    logging.info(f"Initialized {num_honest_clients} honest clients (rep=1.0) and {num_malicious_clients} malicious clients (rep=0.1).")
    logging.info(f"Oracle's initial reputation threshold (TAU): {oracle.get_reputation_threshold():.2f}")

    bft_simulator = AdaptiveBFT_Simulator(validators=all_clients, oracle=oracle)

    # --- Scenario 1: Block from Honest Clients ---
    logging.info("\n--- SCENARIO 1: Testing block from HONEST clients ---")
    honest_block = []
    for client in honest_clients.values():
        update = client.train()
        proof = client.generate_proof()
        honest_block.append({"update": update, "did": proof["did"]})
    
    consensus_reached, _ = bft_simulator.run_consensus(honest_block)
    if consensus_reached:
        logging.info("RESULT: Honest block was ACCEPTED by consensus, as expected.")
    else:
        logging.error("RESULT: Honest block was REJECTED. This is unexpected.")

    # --- Scenario 2: Block from Malicious Clients ---
    logging.info("\n--- SCENARIO 2: Testing block from MALICIOUS clients ---")
    malicious_block = []
    for client in malicious_clients.values():
        # Malicious clients might do less work or have bad data
        client.epochs = 1 
        update = client.train()
        proof = client.generate_proof()
        malicious_block.append({"update": update, "did": proof["did"]})

    consensus_reached, _ = bft_simulator.run_consensus(malicious_block)
    if not consensus_reached:
        logging.info("RESULT: Malicious block was REJECTED by consensus, as expected.")
    else:
        logging.error("RESULT: Malicious block was ACCEPTED. The reputation filter failed.")

if __name__ == '__main__':
    import os
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_resilience_experiment(device)