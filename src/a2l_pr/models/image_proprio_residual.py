import torch
import torch.nn as nn


class ImageProprioResidualPolicy(nn.Module):
    """Residual policy that conditions on image and proprioception histories.

    The model consumes:
    - proprio history: end-effector pose, gripper state, joint positions / velocities, previous actions
    - image history: stacked RGB frames from the observation stream

    The encoder is intentionally simple so the same checkpoint can be trained on
    sim data and then adapted to a real-robot perception stack with minimal API changes.
    """

    def __init__(
        self,
        proprio_dim,
        action_dim,
        history_length=12,
        image_channels=3,
        image_embed_dim=128,
        proprio_embed_dim=256,
        fusion_hidden_dim=256,
        clamp_residual=0.1,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.image_channels = image_channels
        self.clamp_residual = clamp_residual

        self.proprio_encoder = nn.GRU(
            input_size=proprio_dim + action_dim,
            hidden_size=proprio_embed_dim,
            batch_first=True,
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, image_embed_dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(proprio_embed_dim + image_embed_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Linear(fusion_hidden_dim, action_dim)

    def _encode_proprio(self, proprio_history, action_history):
        proprio_input = torch.cat([proprio_history, action_history], dim=-1)
        _, hidden = self.proprio_encoder(proprio_input)
        return hidden[-1]

    def _encode_images(self, image_history):
        batch_size, time_steps, channels, height, width = image_history.shape
        flat = image_history.reshape(batch_size * time_steps, channels, height, width)
        encoded = self.image_encoder(flat)
        return encoded.reshape(batch_size, time_steps, -1).mean(dim=1)

    def forward(self, proprio_history, action_history, image_history):
        proprio_feat = self._encode_proprio(proprio_history, action_history)
        image_feat = self._encode_images(image_history)
        fused = self.fusion(torch.cat([proprio_feat, image_feat], dim=-1))
        residuals = self.residual_head(fused)
        return {"residuals": residuals}

    def predict_first_step(self, proprio_history, action_history, image_history):
        outputs = self.forward(proprio_history, action_history, image_history)
        outputs["residuals"] = torch.clamp(outputs["residuals"], -abs(self.clamp_residual), abs(self.clamp_residual))
        return outputs


class ImageProprioGatedResidualPolicy(ImageProprioResidualPolicy):
    """Residual policy with an intervention gate and failure head."""

    def __init__(
        self,
        proprio_dim,
        action_dim,
        history_length=12,
        image_channels=3,
        image_embed_dim=128,
        proprio_embed_dim=256,
        fusion_hidden_dim=256,
        num_failure_types=5,
        clamp_residual=0.1,
    ):
        super().__init__(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            history_length=history_length,
            image_channels=image_channels,
            image_embed_dim=image_embed_dim,
            proprio_embed_dim=proprio_embed_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            clamp_residual=clamp_residual,
        )
        self.gate_head = nn.Linear(fusion_hidden_dim, 1)
        self.failure_type_head = nn.Linear(fusion_hidden_dim, num_failure_types)

    def forward(self, proprio_history, action_history, image_history):
        proprio_feat = self._encode_proprio(proprio_history, action_history)
        image_feat = self._encode_images(image_history)
        fused = self.fusion(torch.cat([proprio_feat, image_feat], dim=-1))

        residuals = self.residual_head(fused)
        gate_logits = self.gate_head(fused).squeeze(-1)
        failure_logits = self.failure_type_head(fused)
        return {
            "residuals": residuals,
            "gate_logits": gate_logits,
            "failure_logits": failure_logits,
        }

    def predict_first_step(self, proprio_history, action_history, image_history):
        outputs = self.forward(proprio_history, action_history, image_history)
        outputs["gate_probs"] = torch.sigmoid(outputs["gate_logits"])
        outputs["residuals"] = torch.clamp(outputs["residuals"], -abs(self.clamp_residual), abs(self.clamp_residual))
        return outputs