# In /src/blockchain/adaptive_bft.py

import time
import numpy as np
import logging
from blockchain.base_bft import PBFT_Simulator

class NetworkStateOracle:
    """
    *** MODIFIED FOR V3.0 ***
    A threat assessment engine that manages the system's security posture.
    """
    def __init__(self):
        # Security Modes: NORMAL for efficiency, ALERT for maximum security
        self.security_mode = "NORMAL" 
        self.alert_mode_cooldown = 0 # Rounds to stay in ALERT mode
        
        # State variables
        self.last_accuracy = 10.0 # Initial baseline accuracy
        self.global_accuracy = 10.0
        self.consensus_cost = 0.5
        
        # Parameters for ALERT mode trigger
        self.accuracy_drop_threshold = 5.0 # A drop of 5% in one round triggers alert
        
    def update_state(self, accuracy, consensus_latency):
        """Update all metrics and assess threat level."""
        logging.info(f"   ORACLE: Received accuracy={accuracy:.2f}, latency={consensus_latency:.2f}s")
        
        # Check for threat condition to enter ALERT mode
        accuracy_drop = self.last_accuracy - accuracy
        if accuracy_drop > self.accuracy_drop_threshold and self.security_mode == "NORMAL":
            self.security_mode = "ALERT"
            self.alert_mode_cooldown = 3 # Stay in ALERT mode for 3 rounds
            logging.warning(f"   !!! ORACLE: THREAT DETECTED (Accuracy dropped by {accuracy_drop:.2f}%)! Switching to ALERT MODE. !!!")
        
        # Manage cooldown
        if self.security_mode == "ALERT":
            self.alert_mode_cooldown -= 1
            if self.alert_mode_cooldown <= 0:
                self.security_mode = "NORMAL"
                logging.info(f"   ORACLE: Threat passed. Cooldown finished. Returning to NORMAL MODE.")

        self.last_accuracy = accuracy
        self.global_accuracy = accuracy
        
        # Update cost
        normalized_cost = min(1.0, consensus_latency / 0.5)
        self.consensus_cost = 0.7 * self.consensus_cost + 0.3 * normalized_cost

    def get_security_mode(self):
        return self.security_mode

    def get_consensus_cost(self):
        return self.consensus_cost

class AdaptiveBFT_Simulator(PBFT_Simulator):
    """The consensus mechanism remains the same as v2.1"""
    def __init__(self, validators, oracle, **kwargs):
        super().__init__(num_validators=len(validators), **kwargs)
        self.validators = validators
        self.oracle = oracle

    def run_consensus(self, block_of_updates):
        start_time = time.time()
        
        # The firewall logic from v2.1 remains
        client_dids = [update['did'] for update in block_of_updates]
        client_reputations = [self.validators[did].reputation_score for did in client_dids if did in self.validators]
        
        if not client_reputations:
             return False, time.time() - start_time

        avg_rep = np.mean(client_reputations)
        std_dev_rep = np.std(client_reputations)
        
        # Get dynamic thresholds from the oracle instead of using fixed values
        reputation_threshold = self.oracle.get_reputation_threshold()
        std_dev_threshold = self.oracle.get_std_dev_threshold()
        
        if avg_rep < reputation_threshold:
            logging.warning(f"   CONSENSUS REJECTED (Avg Rep): {avg_rep:.2f} < {reputation_threshold:.2f}")
            return False, time.time() - start_time
        if std_dev_rep > std_dev_threshold:
            logging.warning(f"   CONSENSUS REJECTED (Rep Std Dev): {std_dev_rep:.2f} > {std_dev_threshold:.2f}")
            return False, time.time() - start_time

        model_updates = [item['update'] for item in block_of_updates]
        consensus_reached = super().run_consensus(model_updates)
        elapsed_time = time.time() - start_time
        return consensus_reached, elapsed_time