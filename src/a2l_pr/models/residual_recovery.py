import torch
import torch.nn as nn


class ResidualRecoveryPolicy(nn.Module):
    """
    Residual recovery policy that predicts a short horizon of residual actions
    to correct a perturbed trajectory back toward the original trajectory.

    Architecture:
    - Per-step encoder: MLP projecting (state, action) -> step embedding
    - Encoder GRU: consumes the past history of step embeddings
    - Decoder GRU: autoregressively produces a sequence of residual actions
      conditioned on the encoder final hidden state.

    The model returns a tensor of shape (B, horizon, action_dim) representing
    residuals to add to the perturbed actions for the next `prediction_horizon` steps.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        history_length=12,
        prediction_horizon=30,
        step_embed_dim=128,
        enc_hidden_dim=256,
        dec_hidden_dim=256,
        enc_num_layers=1,
        dec_num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon

        # per-step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, step_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(step_embed_dim, step_embed_dim),
            nn.ReLU(),
        )

        # encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=step_embed_dim,
            hidden_size=enc_hidden_dim,
            num_layers=enc_num_layers,
            batch_first=True,
        )

        # decoder: produce residual actions autoregressively
        # we feed a learned start token then autoregress
        self.decoder_input_proj = nn.Linear(action_dim, dec_hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=dec_hidden_dim,
            hidden_size=dec_hidden_dim,
            num_layers=dec_num_layers,
            batch_first=True,
        )
        self.residual_head = nn.Linear(dec_hidden_dim, action_dim)

        # project encoder hidden to decoder init
        self.enc2dec = nn.Linear(enc_hidden_dim, dec_hidden_dim)

    def forward(self, past_states, past_actions, prediction_horizon=None, return_first_only=False):
        """
        Args:
            past_states: (B, history_length, state_dim)
            past_actions: (B, history_length, action_dim)
            prediction_horizon: optional override for number of predicted steps
            return_first_only: if True, return only the first-step residual (B, action_dim)
        Returns:
            residuals: (B, horizon, action_dim) or (B, action_dim) if return_first_only
        """
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        B = past_states.shape[0]
        # encode per step
        x = torch.cat([past_states, past_actions], dim=-1)  # (B, H, state+action)
        H = past_states.shape[1]
        step_embed_flat = self.step_encoder(x.view(-1, x.shape[-1]))
        step_embed = step_embed_flat.view(B, H, -1)

        # encoder GRU
        enc_out, enc_hidden = self.encoder_gru(step_embed)
        # enc_hidden: (num_layers, B, enc_hidden_dim)
        enc_last = enc_hidden[-1]

        # initialize decoder hidden
        dec_hidden = self.enc2dec(enc_last).unsqueeze(0)  # (1, B, dec_hidden_dim)

        # learned start token (zeros) as previous residual
        prev_res = torch.zeros(B, self.action_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(prediction_horizon):
            dec_in = self.decoder_input_proj(prev_res).unsqueeze(1)  # (B,1,dec_hidden)
            dec_out, dec_hidden = self.decoder_gru(dec_in, dec_hidden)
            res_t = self.residual_head(dec_out.squeeze(1))  # (B, action_dim)
            outputs.append(res_t.unsqueeze(1))
            prev_res = res_t

        residuals = torch.cat(outputs, dim=1)  # (B, horizon, action_dim)

        if return_first_only:
            return residuals[:, 0, :]
        return residuals

    def predict_and_apply(self, past_states, past_actions, perturbed_future_actions, clamp=1.0):
        """
        Convenience function: predicts residuals and applies them to a provided
        sequence of perturbed future actions returning the corrected actions.

        Args:
            past_states, past_actions: history inputs for encoder
            perturbed_future_actions: (B, horizon, action_dim) actions to correct
            clamp: max absolute residual to clip for stability
        Returns:
            corrected_actions: same shape as perturbed_future_actions
        """
        residuals = self.forward(past_states, past_actions, prediction_horizon=perturbed_future_actions.shape[1])
        residuals = torch.clamp(residuals, -abs(clamp), abs(clamp))
        return perturbed_future_actions + residuals



class GatedResidualRecoveryPolicy(nn.Module):
    """Residual recovery policy with an intervention gate.

    The model predicts residual actions plus per-step gate logits and
    failure-type logits. At deployment time, the first-step gate probability
    decides whether the residual should be applied to the base policy action.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        history_length=12,
        prediction_horizon=30,
        num_failure_types=5,
        step_embed_dim=128,
        enc_hidden_dim=256,
        dec_hidden_dim=256,
        enc_num_layers=1,
        dec_num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.num_failure_types = num_failure_types

        self.step_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, step_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(step_embed_dim, step_embed_dim),
            nn.ReLU(),
        )
        self.encoder_gru = nn.GRU(
            input_size=step_embed_dim,
            hidden_size=enc_hidden_dim,
            num_layers=enc_num_layers,
            batch_first=True,
        )
        self.decoder_input_proj = nn.Linear(action_dim, dec_hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=dec_hidden_dim,
            hidden_size=dec_hidden_dim,
            num_layers=dec_num_layers,
            batch_first=True,
        )
        self.enc2dec = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.residual_head = nn.Linear(dec_hidden_dim, action_dim)
        self.gate_head = nn.Linear(dec_hidden_dim, 1)
        self.failure_type_head = nn.Linear(dec_hidden_dim, num_failure_types)

    def forward(self, past_states, past_actions, prediction_horizon=None, return_first_only=False):
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        batch_size = past_states.shape[0]
        x = torch.cat([past_states, past_actions], dim=-1)
        history = past_states.shape[1]
        step_embed_flat = self.step_encoder(x.reshape(-1, x.shape[-1]))
        step_embed = step_embed_flat.reshape(batch_size, history, -1)

        _, enc_hidden = self.encoder_gru(step_embed)
        dec_hidden = self.enc2dec(enc_hidden[-1]).unsqueeze(0)
        prev_residual = torch.zeros(batch_size, self.action_dim, device=x.device, dtype=x.dtype)

        residuals = []
        gate_logits = []
        failure_logits = []
        for _ in range(prediction_horizon):
            dec_in = self.decoder_input_proj(prev_residual).unsqueeze(1)
            dec_out, dec_hidden = self.decoder_gru(dec_in, dec_hidden)
            dec_step = dec_out.squeeze(1)
            residual = self.residual_head(dec_step)
            residuals.append(residual.unsqueeze(1))
            gate_logits.append(self.gate_head(dec_step).unsqueeze(1))
            failure_logits.append(self.failure_type_head(dec_step).unsqueeze(1))
            prev_residual = residual

        outputs = {
            "residuals": torch.cat(residuals, dim=1),
            "gate_logits": torch.cat(gate_logits, dim=1).squeeze(-1),
            "failure_logits": torch.cat(failure_logits, dim=1),
        }
        if return_first_only:
            return {
                "residuals": outputs["residuals"][:, 0, :],
                "gate_logits": outputs["gate_logits"][:, 0],
                "failure_logits": outputs["failure_logits"][:, 0, :],
            }
        return outputs

    def predict_first_step(self, past_states, past_actions):
        outputs = self.forward(
            past_states,
            past_actions,
            prediction_horizon=1,
            return_first_only=True,
        )
        outputs["gate_probs"] = torch.sigmoid(outputs["gate_logits"])
        return outputs
