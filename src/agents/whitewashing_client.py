# In /src/agents/whitewashing_client.py

import torch
from torch import nn
import logging
from fedsovereign_components import FedSovereignClient

class WhitewashingClient(FedSovereignClient):
    """
    A more sophisticated malicious client that first builds reputation by acting honestly,
    then begins a targeted poisoning attack.
    """
    def __init__(self, client_id, model, train_dataset, did, 
                 target_label=5, attack_label=3, attack_start_round=5, **kwargs):
        
        # Correctly call the parent __init__ method
        super().__init__(client_id, model, train_dataset, did, **kwargs)
        
        self.target_label = target_label
        self.attack_label = attack_label
        self.attack_start_round = attack_start_round
        
        logging.warning(f"Whitewashing Client {self.client_id} initialized. Attack starts at round {self.attack_start_round}.")

    def train(self, current_round):
        """
        Overrides the training method. Behaves honestly before the attack round,
        and maliciously after.
        """
        if current_round < self.attack_start_round:
            # --- HONEST BEHAVIOR ---
            logging.info(f"Client {self.client_id} (Whitewashing): Behaving honestly in round {current_round}.")
            # Now this call is valid because the parent method accepts the argument
            return super().train(current_round=current_round)
        else:
            # --- MALICIOUS BEHAVIOR ---
            logging.warning(f"Client {self.client_id} (Whitewashing): NOW ATTACKING in round {current_round}!")
            self.model.train()
            for epoch in range(self.epochs):
                for data, target in self.train_loader:
                    # Ensure data is on the correct device for CUDA compatibility
                    data, target = data.to(self.device), target.to(self.device)
                    mask = (target == self.target_label)
                    target[mask] = self.attack_label
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
            
            # Return the poisoned model weights, moved to CPU for aggregation
            return {k: v.cpu() for k, v in self.model.state_dict().items()}