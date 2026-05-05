import torch
import numpy as np
import sys
import os

# Add the src directory to the Python path for testing
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src')))

from a2l_pr.models import FailureRecognizer
from a2l_pr.learning import FailureRecoveryDataset, FailureRecoveryTrainer

def generate_dummy_data(num_samples=100):
    """Generates synthetic data matching our dataset format for testing the pipeline."""
    records = []
    for i in range(num_samples):
        # Image frames: 3 frames stacked channel-wise (e.g. RGB) -> 3x3=9 channels?
        # Actually our dataset takes (N, C, H, W) and stacks internally.
        frames = np.random.rand(3, 3, 224, 224).astype(np.float32)
        
        # Action history: last 2 actions, each with 7 dims (xyz, rpy, gripper)
        actions = np.random.rand(2, 7).astype(np.float32)
        
        # 5 failure types (0: no failure, 1-4: specific failures)
        failure_type_id = np.random.randint(0, 5)
        
        if failure_type_id == 0:
            fsm_id = 0
            recovery_params = np.zeros(7, dtype=np.float32)
        else:
            fsm_id = np.random.randint(1, 3)
            recovery_params = np.random.randn(7).astype(np.float32)
            
        records.append({
            'frames': frames,
            'actions': actions,
            'failure_type_id': failure_type_id,
            'fsm_id': fsm_id,
            'recovery_params': recovery_params
        })
    return records

def main():
    print("1. Generating synthetic dataset...")
    train_records = generate_dummy_data(80)
    val_records = generate_dummy_data(20)
    
    train_dataset = FailureRecoveryDataset(train_records)
    val_dataset = FailureRecoveryDataset(val_records)
    
    print("2. Instantiating FailureRecognizer baseline model...")
    # Matches CNN/ViT Encoder + MLP Heads architecture
    model = FailureRecognizer(
        num_frames=3,
        action_dim=7,
        num_action_history=2,
        num_failure_types=5,
        recovery_param_dim=7,
        num_fsm_templates=3,
        vision_encoder_type="resnet18", # Or 'vit_b_16'
        hidden_dim=256
    )
    
    # Optional: freeze the vision encoder early layers if pretrained
    
    print("3. Setting up Trainer...")
    # Use CPU for dummy run, use cuda if available in real life
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = FailureRecoveryTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        lr=3e-4,
        device=device
    )
    
    print("4. Starting Training Loop...")
    for epoch in range(1, 3):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = trainer.train_epoch()
        print(f"Train Loss: {train_loss:.4f}")
        
        val_metrics = trainer.evaluate()
        print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val Failure Acc: {val_metrics['val_failure_acc']:.2%}")
        
    print("\n5. Testing Inference and Recovery Text Generation...")
    model.eval()
    with torch.no_grad():
        # Take a single sample
        sample = val_dataset[0]
        # Add batch dimension
        frames = sample['frames'].unsqueeze(0).to(device)
        actions = sample['actions'].unsqueeze(0).to(device)
        
        outputs = model(frames, actions)
        
        pred_failure_idx = torch.argmax(outputs['failure_logits'], dim=1).item()
        pred_fsm_id = torch.argmax(outputs['fsm_logits'], dim=1).item()
        pred_recovery_params = outputs['recovery_params'][0].cpu().numpy()
        
        text_output = model.generate_recovery_text(
            failure_type_idx=pred_failure_idx,
            fsm_id=pred_fsm_id,
            recovery_params=pred_recovery_params
        )
        print(f"Generated text reasoning: {text_output}")

if __name__ == "__main__":
    main()
