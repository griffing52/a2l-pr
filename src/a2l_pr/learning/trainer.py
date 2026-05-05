import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class FailureRecoveryTrainer:
    """
    Research-friendly trainer for the FailureRecognizer model.
    Handles multi-task training: failure classification, FSM classification, and recovery parameter regression.
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=32,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        weights=None
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Loss functions
        # Weights can be adjusted based on class imbalance (e.g., 'no failure' might be very common)
        class_weights = torch.tensor(weights).float().to(device) if weights else None
        self.criterion_failure = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_fsm = nn.CrossEntropyLoss()
        
        # We only want to compute regression loss when there is an actual failure
        self.criterion_recovery = nn.MSELoss(reduction='none')

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            frames = batch['frames'].to(self.device)
            actions = batch.get('actions', None)
            if actions is not None:
                actions = actions.to(self.device)
                
            labels_failure = batch['failure_type'].to(self.device)
            labels_fsm = batch['fsm_id'].to(self.device)
            labels_recovery = batch['recovery_params'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(frames, actions)
            
            # 1. Failure Type Loss
            loss_failure = self.criterion_failure(outputs['failure_logits'], labels_failure)
            
            # 2. FSM Classification Loss (Only for actual failures)
            # Assuming label 0 is "no failure". We only penalize fsm and recovery if label > 0
            failure_mask = (labels_failure > 0)
            
            if failure_mask.sum() > 0:
                loss_fsm = self.criterion_fsm(
                    outputs['fsm_logits'][failure_mask], 
                    labels_fsm[failure_mask]
                )
                
                # 3. Recovery Params Loss (Only for actual failures)
                loss_recovery_full = self.criterion_recovery(
                    outputs['recovery_params'][failure_mask], 
                    labels_recovery[failure_mask]
                )
                loss_recovery = loss_recovery_full.mean()
            else:
                loss_fsm = 0.0
                loss_recovery = 0.0
                
            # Total Loss (Scale up recovery regression loss so it learns the values)
            loss = loss_failure + loss_fsm + (10.0 * loss_recovery)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self):
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        correct_failure = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                frames = batch['frames'].to(self.device)
                actions = batch.get('actions', None)
                if actions is not None:
                    actions = actions.to(self.device)
                    
                labels_failure = batch['failure_type'].to(self.device)
                labels_fsm = batch['fsm_id'].to(self.device)
                labels_recovery = batch['recovery_params'].to(self.device)
                
                outputs = self.model(frames, actions)
                
                loss_failure = self.criterion_failure(outputs['failure_logits'], labels_failure)
                
                failure_mask = (labels_failure > 0)
                if failure_mask.sum() > 0:
                    loss_fsm = self.criterion_fsm(outputs['fsm_logits'][failure_mask], labels_fsm[failure_mask])
                    loss_recovery = self.criterion_recovery(outputs['recovery_params'][failure_mask], labels_recovery[failure_mask]).mean()
                else:
                    loss_fsm = 0.0
                    loss_recovery = 0.0
                    
                loss = loss_failure + loss_fsm + (10.0 * loss_recovery)
                total_loss += loss.item()
                
                # Metrics
                preds = torch.argmax(outputs['failure_logits'], dim=1)
                correct_failure += (preds == labels_failure).sum().item()
                total_samples += labels_failure.size(0)
                
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_failure_acc': correct_failure / total_samples
        }
        return metrics
