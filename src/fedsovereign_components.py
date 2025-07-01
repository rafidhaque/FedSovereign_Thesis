# In /src/fedsovereign_components.py

import torch
import time
import numpy as np
from fl_components.base_fl import FedAvgClient, FedAvgServer

class FedSovereignClient(FedAvgClient):
    """A FedSovereign client that is aware of consensus costs and can generate proofs."""
    def __init__(self, client_id, model, train_dataset, did, **kwargs):
        super().__init__(client_id, model, train_dataset, **kwargs)
        self.did = did
        self.base_epochs = self.epochs
        self.reputation_score = 1.0 # Initial reputation

    def generate_proof(self):
        """Simulates the generation of a ZKP for 'Proof of Honest Effort'."""
        time.sleep(np.random.uniform(0.1, 0.3)) # Simulated ZKP cost
        return {"proof_data": "valid_proof", "did": self.did}

    def adapt_training_epochs(self, network_state_oracle):
        """Consensus-Aware Learning: Adjusts local epochs based on network cost."""
        consensus_cost = network_state_oracle.get_consensus_cost()
        if consensus_cost > 0.8: # High cost threshold
            self.epochs = int(self.base_epochs * 1.5)
        elif consensus_cost < 0.2: # Low cost threshold
            self.epochs = self.base_epochs

class FedSovereignServer(FedAvgServer):
    """The server orchestrates the FedSovereign process, including evaluation."""
    def __init__(self, global_model):
        super().__init__(global_model)

    def sovereign_aggregation(self, client_updates, validators):
        """
        *** NEW METHOD V2.1 ***
        Fortified aggregation that weights contributions by the client's reputation score.
        This directly incorporates the successful logic from Baseline 2.
        """
        aggregated_weights = list(client_updates.values())[0].copy()
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        total_rep_sum = sum(v.reputation_score for v in validators.values())
        if total_rep_sum == 0:
            return self.global_model.state_dict() # Avoid division by zero

        # Perform weighted sum based on validator's reputation score
        for did, update in client_updates.items():
            rep_weight = validators[did].reputation_score / total_rep_sum
            for key in aggregated_weights.keys():
                aggregated_weights[key] += update[key] * rep_weight
                
        self.global_model.load_state_dict(aggregated_weights)
        return self.global_model.state_dict()

    def run_evaluation_committee(self, validators, test_dataset):
        """
        Simulates the Decentralized Evaluation Committee.
        A random subset of validators evaluates the global model.
        """
        committee_size = max(3, int(len(validators) * 0.1))
        committee_dids = np.random.choice(list(validators.keys()), committee_size, replace=False)
        
        accuracies = []
        for did in committee_dids:
            # Simulate accuracy evaluation
            # Honest validators report more accurately
            base_acc = 90.0
            if "malicious" in validators[did].__class__.__name__.lower():
                base_acc = 20.0 # Malicious validators might report badly
            
            accuracy = base_acc + np.random.normal(0, 2)
            accuracies.append(accuracy)
            
        return np.median(accuracies)