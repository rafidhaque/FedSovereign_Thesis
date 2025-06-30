import time
import logging
import numpy as np
from .base_bft import PBFT_Simulator

class NetworkStateOracle:
    """A simple smart contract simulation that holds the network's state."""
    def __init__(self):
        self.global_accuracy = 0.0
        self.consensus_cost = 0.5 # Normalized value (0 to 1)
        self.reputation_threshold_tau = 0.5 # Initial reputation threshold

    def update_global_accuracy(self, accuracy):
        self.global_accuracy = accuracy
        logging.info(f"   ORACLE: Global Accuracy updated to {accuracy:.2f}%.")
        self.adapt_consensus_rules()

    def update_consensus_cost(self, cost):
        # Update with some smoothing (e.g., moving average)
        self.consensus_cost = 0.7 * self.consensus_cost + 0.3 * cost
        logging.debug(f"   ORACLE: Consensus Cost updated to {self.consensus_cost:.2f}.")

    def adapt_consensus_rules(self):
        """Learning-Aware Consensus: DAO-gated rule adaptation."""
        # This simulates a DAO proposal and vote.
        # If accuracy drops significantly, the DAO tightens security.
        if self.global_accuracy < 85.0 and self.reputation_threshold_tau < 0.8:
            self.reputation_threshold_tau = min(self.reputation_threshold_tau + 0.1, 0.8)
            logging.info(f"   ORACLE (DAO Action): Accuracy low! Reputation threshold TAU increased to {self.reputation_threshold_tau:.2f}.")
        elif self.global_accuracy > 92.0 and self.reputation_threshold_tau > 0.5:
             self.reputation_threshold_tau = max(self.reputation_threshold_tau - 0.1, 0.5)
             logging.info(f"   ORACLE (DAO Action): Accuracy high! Reputation threshold TAU relaxed to {self.reputation_threshold_tau:.2f}.")

    def get_consensus_cost(self):
        return self.consensus_cost
        
    def get_reputation_threshold(self):
        return self.reputation_threshold_tau

class AdaptiveBFT_Simulator(PBFT_Simulator):
    """An adaptive BFT simulator that incorporates RA-dBFT logic."""
    def __init__(self, validators, oracle, **kwargs):
        super().__init__(num_validators=len(validators), **kwargs)
        self.validators = validators # Expects a dict mapping DID to client object
        self.oracle = oracle

    def run_consensus(self, block_of_updates):
        """
        Overrides the base method to include the reputation check.
        Expects block_of_updates to be a list of dicts: [{'update': {...}, 'did': '...'}]
        Returns a tuple: (consensus_reached, elapsed_time)
        """
        start_time = time.time()
        
        # --- Core RA-dBFT Novelty ---
        reputation_threshold = self.oracle.get_reputation_threshold()
        client_dids = [item['did'] for item in block_of_updates]
        
        # Get reputation for each client in the block
        client_reputations = [self.validators[did].reputation_score for did in client_dids if did in self.validators]
        
        if not client_reputations:
             logging.warning("   CONSENSUS: No reputable clients in block. Rejecting.")
             return False, time.time() - start_time

        avg_rep = np.mean(client_reputations)
        
        if avg_rep < reputation_threshold:
            logging.warning(f"   CONSENSUS REJECTED: Average reputation {avg_rep:.2f} is below threshold {reputation_threshold:.2f}.")
            return False, time.time() - start_time
        # --- End of RA-dBFT Logic ---

        # If reputation check passes, proceed with standard PBFT using only the model updates
        model_updates = [item['update'] for item in block_of_updates]
        consensus_reached = super().run_consensus(model_updates)
        
        elapsed_time = time.time() - start_time
        return consensus_reached, elapsed_time