import time
import numpy as np
import logging
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
        # Simulate computational delay for proof generation
        logging.debug(f"Client {self.client_id}: Generating proof of honest effort...")
        time.sleep(np.random.uniform(0.1, 0.3)) # Simulated ZKP cost
        return {"proof_data": "valid_proof", "did": self.did}

    def adapt_training_epochs(self, network_state_oracle):
        """Consensus-Aware Learning: Adjusts local epochs based on network cost."""
        consensus_cost = network_state_oracle.get_consensus_cost()
        if consensus_cost > 0.8: # High cost threshold
            self.epochs = int(self.base_epochs * 1.5)
            logging.info(f"Client {self.client_id}: Network cost is high ({consensus_cost:.2f}). Increasing epochs to {self.epochs}.")
        elif consensus_cost < 0.2: # Low cost threshold
            self.epochs = self.base_epochs
            logging.info(f"Client {self.client_id}: Network cost is low ({consensus_cost:.2f}). Resetting epochs to {self.epochs}.")
        # Otherwise, keep current epoch count

class FedSovereignServer(FedAvgServer):
    """The server orchestrates the FedSovereign process, including evaluation."""
    def __init__(self, global_model):
        super().__init__(global_model)

    def run_evaluation_committee(self, validators, test_dataset):
        """
        Simulates the Decentralized Evaluation Committee.
        A random subset of validators evaluates the global model.
        """
        committee_size = max(3, int(len(validators) * 0.1)) # 10% of validators or at least 3
        committee = np.random.choice(list(validators.values()), committee_size, replace=False)
        
        accuracies = []
        for member in committee:
            # In a real system, this would be a complex process. Here we simulate it.
            # For simplicity, we'll just return a simulated accuracy.
            accuracy = 90.0 + np.random.normal(0, 2) # High accuracy with some noise
            accuracies.append(accuracy)
            
        # Return the median accuracy to be robust to outliers
        median_accuracy = np.median(accuracies)
        logging.info(f"Evaluation committee returned a median accuracy of {median_accuracy:.2f}%.")
        return median_accuracy