import torch
from torch import nn
import logging

# We import the base client to inherit its structure
from fedsovereign_components import FedSovereignClient 

class MaliciousClient(FedSovereignClient):
    """
    A malicious client that performs a targeted label-flipping attack.
    It aims to make the model misclassify one digit as another.
    """
    def __init__(self, client_id, model, train_dataset, did, target_label, attack_label, **kwargs):
        super().__init__(client_id, model, train_dataset, did, **kwargs)
        self.target_label = target_label
        self.attack_label = attack_label
        logging.warning(f"Malicious Client {self.client_id} initialized to attack label {self.target_label} -> {self.attack_label}")

    def train(self):
        """
        Overrides the standard training method to inject poisoned data.
        """
        logging.info(f"Client {self.client_id} (Malicious): Starting poisoned training for {self.epochs} epochs.")
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in self.train_loader:
                # This is the critical fix: move data to the same device as the model.
                data, target = data.to(self.device), target.to(self.device)
                
                # --- The Poisoning Logic ---
                # Create a mask for the target label
                mask = (target == self.target_label)
                # Change the labels for the target class to the attack class
                target[mask] = self.attack_label
                # --- End of Poisoning Logic ---

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        # Return model weights on the CPU for robust aggregation.
        return {k: v.cpu() for k, v in self.model.state_dict().items()}