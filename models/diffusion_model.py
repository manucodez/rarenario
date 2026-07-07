from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x) -> bool:
    return x is not None


def default(val, d):
    return val if exists(val) else d


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("time embedding dimension must be positive")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer or float timesteps
        returns: (B, dim)
        """
        if t.dim() == 0:
            t = t[None]
        t = t.float()

        half_dim = self.dim // 2
        device = t.device

        if half_dim == 0:
            return t.unsqueeze(-1)

        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exponent)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryContextEncoder(nn.Module):
    """
    Encodes past multi-agent trajectories into a single scene context vector.

    Input:
        past: (B, P, A, D)
        past_mask: (B, P, A) boolean/0-1, optional

    Output:
        context: (B, context_dim)
    """

    def __init__(
        self,
        past_steps: int,
        state_dim: int,
        agent_hidden_dim: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.past_steps = past_steps
        self.state_dim = state_dim
        self.agent_input_dim = past_steps * (state_dim + 1)
        self.agent_encoder = MLP(
            in_dim=self.agent_input_dim,
            hidden_dim=agent_hidden_dim,
            out_dim=agent_hidden_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.scene_fuser = MLP(
            in_dim=agent_hidden_dim + state_dim,
            hidden_dim=agent_hidden_dim,
            out_dim=context_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        past: (B, P, A, D)
        past_mask: (B, P, A)
        """
        if past.dim() != 4:
            raise ValueError(f"past must have shape (B, P, A, D), got {tuple(past.shape)}")

        B, P, A, D = past.shape
        if P != self.past_steps:
            raise ValueError(f"Expected past_steps={self.past_steps}, got {P}")
        if D != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {D}")

        past = torch.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)

        if past_mask is None:
            past_mask = torch.ones(B, P, A, device=past.device, dtype=torch.bool)

        past_mask_f = past_mask.float().unsqueeze(-1)  # (B, P, A, 1)
        past = past * past_mask_f

        # Per-agent flattening over time.
        agent_traj = past.permute(0, 2, 1, 3).contiguous().reshape(B, A, P * D)
        agent_mask = past_mask.permute(0, 2, 1).contiguous().float().reshape(B, A, P)

        agent_feat = torch.cat([agent_traj, agent_mask.reshape(B, A, P)], dim=-1)
        agent_latent = self.agent_encoder(agent_feat)  # (B, A, H)

        # Masked pooling over agents.
        agent_present = past_mask.any(dim=1).float()  # (B, A)
        denom = agent_present.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (agent_latent * agent_present.unsqueeze(-1)).sum(dim=1) / denom

        # Add global statistics to help conditioning.
        mean_agent = (past * past_mask_f).sum(dim=(1, 2)) / past_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        scene_feat = torch.cat([pooled, mean_agent], dim=-1)

        context = self.scene_fuser(scene_feat)
        context = self.norm(context)
        return context


class FutureTrajectoryDenoiser(nn.Module):
    """
    Predicts diffusion noise for future multi-agent trajectories.

    Input:
        x_noisy:  (B, F, A, D)
        context:  (B, C)
        t_emb:    (B, T)

    Output:
        noise prediction with same shape as x_noisy
    """

    def __init__(
        self,
        future_steps: int,
        state_dim: int,
        context_dim: int,
        time_emb_dim: int,
        agent_hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.state_dim = state_dim
        self.input_dim = future_steps * state_dim + context_dim + time_emb_dim

        self.net = MLP(
            in_dim=self.input_dim,
            hidden_dim=agent_hidden_dim,
            out_dim=future_steps * state_dim,
            num_layers=4,
            dropout=dropout,
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_noisy.dim() != 4:
            raise ValueError(f"x_noisy must have shape (B, F, A, D), got {tuple(x_noisy.shape)}")

        B, F, A, D = x_noisy.shape
        if F != self.future_steps:
            raise ValueError(f"Expected future_steps={self.future_steps}, got {F}")
        if D != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {D}")

        x_noisy = torch.nan_to_num(x_noisy, nan=0.0, posinf=0.0, neginf=0.0)

        if future_mask is None:
            future_mask = torch.ones(B, F, A, device=x_noisy.device, dtype=torch.bool)

        future_mask_f = future_mask.float().unsqueeze(-1)
        x_noisy = x_noisy * future_mask_f

        # Per-agent trajectory flattening.
        x_flat = x_noisy.permute(0, 2, 1, 3).contiguous().reshape(B, A, F * D)

        context_rep = context.unsqueeze(1).expand(B, A, context.shape[-1])
        t_rep = t_emb.unsqueeze(1).expand(B, A, t_emb.shape[-1])

        inp = torch.cat([x_flat, context_rep, t_rep], dim=-1)
        out = self.net(inp)  # (B, A, F*D)
        out = out.reshape(B, A, F, D).permute(0, 2, 1, 3).contiguous()
        return out


class DiffusionModel(nn.Module):
    """
    CTG-style conditional diffusion model for future multi-agent trajectory prediction.

    Conditioning:
        - past trajectories
        - past masks

    Target:
        - future trajectories

    The model is designed to work with your current preprocessing output:
        trajectories.npy -> (S, T, A, D)
    and dataset output:
        past   -> (B, P, A, D)
        future -> (B, F, A, D)
    """

    def __init__(
        self,
        state_dim: int = 4,
        past_steps: int = 20,
        future_steps: int = 21,
        context_dim: int = 128,
        agent_hidden_dim: int = 256,
        time_emb_dim: int = 64,
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.context_encoder = TrajectoryContextEncoder(
            past_steps=past_steps,
            state_dim=state_dim,
            agent_hidden_dim=agent_hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.denoiser = FutureTrajectoryDenoiser(
            future_steps=future_steps,
            state_dim=state_dim,
            context_dim=context_dim,
            time_emb_dim=time_emb_dim,
            agent_hidden_dim=agent_hidden_dim,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)

    def encode_context(
        self,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.context_encoder(past, past_mask)

    def predict_noise(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = self.encode_context(past, past_mask)
        t_emb = self.time_embedding(t)
        return self.denoiser(x_noisy, context, t_emb, future_mask=future_mask)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_noise(x_noisy, t, past, past_mask, future_mask)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a 1-D buffer a at indices t and reshape for broadcast.
        """
        if t.dim() == 0:
            t = t[None]
        out = a.gather(0, t.long())
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Diffuse x_start to timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Training loss: predict added noise on future trajectories.

        x_start: future ground-truth, shape (B, F, A, D)
        """
        x_start = torch.nan_to_num(x_start, nan=0.0, posinf=0.0, neginf=0.0)

        if future_mask is None:
            future_mask = torch.ones(
                x_start.shape[0],
                x_start.shape[1],
                x_start.shape[2],
                device=x_start.device,
                dtype=torch.bool,
            )

        future_mask_f = future_mask.float().unsqueeze(-1)
        x_start = x_start * future_mask_f

        if noise is None:
            noise = torch.randn_like(x_start)
        noise = noise * future_mask_f

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.predict_noise(
            x_noisy=x_noisy,
            t=t,
            past=past,
            past_mask=past_mask,
            future_mask=future_mask,
        )

        per_elem = (pred_noise - noise) ** 2
        per_elem = per_elem * future_mask_f

        denom = future_mask_f.sum().clamp_min(1.0) * x_start.shape[-1]
        loss = per_elem.sum() / denom

        metrics = {
            "mse": loss.detach(),
            "mask_count": future_mask_f.sum().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        past: torch.Tensor,
        past_mask: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate future trajectories by reverse diffusion.

        Returns:
            x: (B, F, A, D)
        """
        device = past.device
        B = past.shape[0]
        A = past.shape[2]
        F = self.future_steps
        D = self.state_dim

        if future_mask is None:
            future_mask = torch.ones(B, F, A, device=device, dtype=torch.bool)

        x = torch.randn(B, F, A, D, device=device)

        steps = self.num_diffusion_steps if num_steps is None else min(num_steps, self.num_diffusion_steps)
        step_indices = torch.linspace(self.num_diffusion_steps - 1, 0, steps, device=device).long()

        for idx in step_indices:
            t = torch.full((B,), int(idx.item()), device=device, dtype=torch.long)

            pred_noise = self.predict_noise(
                x_noisy=x,
                t=t,
                past=past,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            betas_t = self._extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
            )

            if int(idx.item()) > 0:
                posterior_var_t = self._extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x = model_mean

            x = torch.where(future_mask.unsqueeze(-1), x, torch.zeros_like(x))

        return x

    def get_device(self) -> torch.device:
        return next(self.parameters()).device