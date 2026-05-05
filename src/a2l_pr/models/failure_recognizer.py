import torch
import torch.nn as nn
import torchvision.models as models

class FailureRecognizer(nn.Module):
    """
    A simple baseline model for failure recognition and recovery prediction.
    
    Architecture:
    - Vision Encoder: ResNet18 or ViT (adapted for stacked frames) -> Flatten -> Latent Vector
    - Action Encoder (optional): MLP processing the last N actions -> Action Latent
    - Trunk: Shared MLP taking the concatenated vision and action latents
    - Heads:
        * failure_type: Classification (which failure occurred, or "no failure")
        * recovery_params: Regression (parameters needed for the recovery)
        * fsm_template_id: Classification (which recovery FSM to invoke)
    """
    def __init__(
        self,
        num_frames=3,
        action_dim=None,  # Set to an integer (e.g., 7) if using action history
        num_action_history=2,
        num_failure_types=5, # e.g., 0: no failure, 1: drop, 2: underreach, etc.
        recovery_param_dim=7, # e.g., delta x, y, z, r, p, y, gripper
        num_fsm_templates=3,
        vision_encoder_type="resnet18",
        hidden_dim=256
    ):
        super().__init__()
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.num_action_history = num_action_history
        self.vision_encoder_type = vision_encoder_type
        
        # 1. Vision Encoder / State Encoder
        if vision_encoder_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_channels = num_frames * 3
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                repeated_weight = original_conv.weight.repeat(1, num_frames, 1, 1)
                new_conv.weight.copy_(repeated_weight / num_frames)
            self.vision_encoder = nn.Sequential(
                new_conv,
                *list(resnet.children())[1:-1],
                nn.Flatten()
            )
            vision_out_dim = 512
            
        elif vision_encoder_type == "vit_b_16":
            vit = models.vit_b_16(pretrained=False)
            in_channels = num_frames * 3
            vit.conv_proj = nn.Conv2d(in_channels, vit.hidden_dim, kernel_size=16, stride=16)
            vit.heads = nn.Identity()
            self.vision_encoder = vit
            vision_out_dim = 768
            
        elif vision_encoder_type == "mlp":
            # For low_dim state vectors instead of images
            # frames will actually be a 1D state vector of shape (B, state_dim)
            # where state_dim is passed in num_frames (hacky but works for quick testing)
            state_dim = num_frames 
            self.vision_encoder = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )
            vision_out_dim = 256
            
        else:
            raise ValueError(f"Unsupported encoder: {vision_encoder_type}")

        # 2. Action History Processing
        action_feat_dim = 0
        if action_dim is not None and num_action_history > 0:
            action_feat_dim = 64
            self.action_mlp = nn.Sequential(
                nn.Linear(action_dim * num_action_history, 128),
                nn.ReLU(),
                nn.Linear(128, action_feat_dim),
                nn.ReLU()
            )
            
        # 3. Shared MLP Trunk
        combined_dim = vision_out_dim + action_feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 4. Heads
        self.failure_type_head = nn.Linear(hidden_dim, num_failure_types)
        self.recovery_params_head = nn.Linear(hidden_dim, recovery_param_dim)
        self.fsm_template_head = nn.Linear(hidden_dim, num_fsm_templates)

    def forward(self, frames, actions=None):
        """
        Args:
            frames: Tensor of shape (B, num_frames * 3, H, W).
                    Stacked channel-wise.
            actions: Tensor of shape (B, num_action_history, action_dim). Optional.
        Returns:
            Dictionary containing logits for failure type, FSM template, and regression params.
        """
        # Encode vision
        v_feat = self.vision_encoder(frames)
        
        # Encode actions if provided
        if self.action_dim is not None and actions is not None:
            B = actions.shape[0]
            actions_flat = actions.view(B, -1) # (B, num_action_history * action_dim)
            a_feat = self.action_mlp(actions_flat)
            # Concatenate
            feat = torch.cat([v_feat, a_feat], dim=-1)
        else:
            feat = v_feat
            
        # Pass through trunk
        latent = self.trunk(feat)
        
        # Heads
        failure_logits = self.failure_type_head(latent)
        recovery_params = self.recovery_params_head(latent)
        fsm_logits = self.fsm_template_head(latent)
        
        return {
            "failure_logits": failure_logits,
            "recovery_params": recovery_params,
            "fsm_logits": fsm_logits
        }

    def generate_recovery_text(self, failure_type_idx, fsm_id, recovery_params, failure_vocab=None, fsm_vocab=None):
        """
        Simple template fill for recovery text based on predicted heads.
        Returns a text string describing the recovery action.
        """
        if failure_vocab is None:
            failure_vocab = {0: "no failure", 1: "drop", 2: "underreach", 3: "overreach", 4: "collision"}
        if fsm_vocab is None:
            fsm_vocab = {0: "idle", 1: "re-grasp", 2: "move-back-and-retry"}
            
        failure_str = failure_vocab.get(failure_type_idx, f"unknown_failure_{failure_type_idx}")
        fsm_str = fsm_vocab.get(fsm_id, f"unknown_fsm_{fsm_id}")
        
        if failure_str == "no failure":
            return "No failure detected. Continue normal operation."
            
        params_str = ", ".join([f"{p:.2f}" for p in recovery_params])
        
        return f"Detected '{failure_str}'. Executing '{fsm_str}' recovery with params [{params_str}]."
