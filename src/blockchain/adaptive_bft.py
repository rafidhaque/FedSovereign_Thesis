# In /src/blockchain/adaptive_bft.py

import time
import numpy as np
from blockchain.base_bft import PBFT_Simulator

class NetworkStateOracle:
    # ... (This class remains the same as before) ...
    def __init__(self):
        self.global_accuracy = 0.0
        self.consensus_cost = 0.5 
        self.reputation_threshold_tau = 0.6 # Start with a slightly stricter threshold
        self.std_dev_threshold_sigma = 0.3 # New threshold for reputation variance

    def update_global_accuracy(self, accuracy):
        self.global_accuracy = accuracy
        self.adapt_consensus_rules()

    def update_consensus_cost(self, cost):
        self.consensus_cost = 0.7 * self.consensus_cost + 0.3 * cost

    def adapt_consensus_rules(self):
        # Simulate DAO-gated rule changes
        if self.global_accuracy < 40.0 and self.reputation_threshold_tau < 0.8:
            self.reputation_threshold_tau = min(self.reputation_threshold_tau + 0.05, 0.8)
            print(f"   ORACLE (DAO Action): Accuracy low! Reputation threshold TAU increased to {self.reputation_threshold_tau:.2f}.")
        elif self.global_accuracy > 60.0 and self.reputation_threshold_tau > 0.5:
             self.reputation_threshold_tau = max(self.reputation_threshold_tau - 0.05, 0.5)
             print(f"   ORACLE (DAO Action): Accuracy high! Reputation threshold TAU relaxed to {self.reputation_threshold_tau:.2f}.")
             
    def get_consensus_cost(self):
        return self.consensus_cost
        
    def get_reputation_threshold(self):
        return self.reputation_threshold_tau
        
    def get_std_dev_threshold(self):
        return self.std_dev_threshold_sigma

class AdaptiveBFT_Simulator(PBFT_Simulator):
    """An adaptive BFT simulator that incorporates fortified RA-dBFT logic."""
    def __init__(self, validators, oracle, **kwargs):
        super().__init__(num_validators=len(validators), **kwargs)
        self.validators = validators
        self.oracle = oracle

    def run_consensus(self, block_of_updates):
        """
        *** MODIFIED METHOD V2.1 ***
        Overrides the base method to include the fortified reputation check.
        """
        start_time = time.time()
        
        # --- Fortified RA-dBFT Firewall ---
        rep_threshold = self.oracle.get_reputation_threshold()
        std_dev_threshold = self.oracle.get_std_dev_threshold()
        
        client_dids = [update['did'] for update in block_of_updates]
        client_reputations = [self.validators[did].reputation_score for did in client_dids if did in self.validators]
        
        if not client_reputations:
             return False, time.time() - start_time

        avg_rep = np.mean(client_reputations)
        std_dev_rep = np.std(client_reputations)
        
        # Condition 1: Average reputation must be high enough
        if avg_rep < rep_threshold:
            print(f"   CONSENSUS REJECTED (Avg Rep): {avg_rep:.2f} < {rep_threshold:.2f}")
            return False, time.time() - start_time
            
        # Condition 2: Reputation variance must not be too high (detects mixed-quality blocks)
        if std_dev_rep > std_dev_threshold:
            print(f"   CONSENSUS REJECTED (Rep Std Dev): {std_dev_rep:.2f} > {std_dev_threshold:.2f}")
            return False, time.time() - start_time
        # --- End of Firewall Logic ---

        # If all checks pass, proceed with standard PBFT quorum logic
        consensus_reached = super().run_consensus(block_of_updates)
        
        elapsed_time = time.time() - start_time
        return consensus_reached, elapsed_time