import time
import random

class PBFT_Simulator:
    """A simple functional simulation of a PBFT-like consensus protocol."""
    def __init__(self, num_validators=10, faulty_validators=2, latency_ms=10):
        self.num_validators = num_validators
        self.faulty_validators = faulty_validators
        self.latency_ms = latency_ms

    def run_consensus(self, block_of_updates):
        """
        Simulates the three phases of PBFT.
        Returns True if consensus is reached, False otherwise.
        """
        if not block_of_updates:
            return False # Cannot run consensus on an empty block
            
        # Simulate network latency
        time.sleep(self.latency_ms / 1000.0)

        # Pre-prepare, prepare, commit phases
        # For simulation, we'll just check if enough honest nodes exist.
        num_honest_validators = self.num_validators - self.faulty_validators
        
        # PBFT requires 2f+1 honest nodes to reach a quorum
        if num_honest_validators >= 2 * self.faulty_validators + 1:
            # print(f"PBFT Consensus SUCCESS: Quorum reached ({num_honest_validators}/{self.num_validators}).")
            return True
        else:
            # print(f"PBFT Consensus FAILED: Quorum not reached ({num_honest_validators}/{self.num_validators}).")
            return False