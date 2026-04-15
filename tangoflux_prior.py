"""
TangoFlux Prior Wrapper for Inference-Time Scaling.

Wraps TangoFlux model + VAE into a unified interface that exposes:
  - init_latent()           → sample initial noise
  - compute_velocity()      → predict velocity with CFG (in native linear space)
  - compute_velocity_transformed() → predict velocity with scheduler conversion
  - get_tweedie()           → estimate clean x_0 from noisy x_t
  - euler_step() / sde_step() → take a denoising step (ODE or SDE)
  - decode_latent()         → decode latent to waveform
  - setup_timesteps()       → initialize scheduler timesteps/sigmas

Supports scheduler conversion at inference time:
  - "linear" (CondOT):  α_t = 1 - t,  σ_t = t           (native training schedule)
  - "vp":               α_t = e^(-T/2), σ_t = √(1-e^(-T)) (variance-preserving)

When using VP, the model always runs in its native linear space — latents are
rescaled, time is mapped via SNR matching, and velocities are transformed back.
This follows the approach from Flow-Inference-Time-Scaling (Ma et al., 2025).
"""

import math
import torch
import numpy as np
from dataclasses import dataclass, field
from tangoflux.model import TangoFlux, retrieve_timesteps


# ============================================================================
# Scheduler Classes (ported from Flow-Inference-Time-Scaling)
# ============================================================================

@dataclass
class SchedulerOutput:
    """Output from a scheduler: coefficients and their time-derivatives."""
    alpha_t: torch.Tensor   # α_t
    sigma_t: torch.Tensor   # σ_t
    d_alpha_t: torch.Tensor # dα/dt
    d_sigma_t: torch.Tensor # dσ/dt


class CondOTScheduler:
    """Linear / Conditional Optimal Transport scheduler.

    This is the native schedule for flow matching models like TangoFlux/Flux.
        α_t = 1 - t,   σ_t = t
        dα/dt = -1,     dσ/dt = 1
    """
    def __call__(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        return SchedulerOutput(
            alpha_t=1 - t,
            sigma_t=t,
            d_alpha_t=-torch.ones_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def snr_inverse(self, snr):
        """Given SNR = α/σ, find t such that (1-t)/t = snr → t = 1/(1+snr)."""
        return 1.0 / (snr + 1.0)


class VPScheduler:
    """Variance-Preserving scheduler.

    Uses a linear beta schedule:
        T(t) = 0.5 * t² * (β_max - β_min) + t * β_min
        α_t = exp(-T/2)
        σ_t = √(1 - exp(-T))
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * t ** 2 * (B - b) + t * b
        dT = t * (B - b) + b

        exp_neg_T = torch.exp(-T)
        sqrt_term = torch.sqrt(torch.clamp(1 - exp_neg_T, min=1e-10))

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=sqrt_term,
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * exp_neg_T / sqrt_term,
        )

    def snr_inverse(self, snr):
        """Given SNR = α/σ, find t."""
        T = -torch.log(snr ** 2 / (snr ** 2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = (-b + torch.sqrt(b ** 2 + 2 * (B - b) * T)) / (B - b)
        return t


# Map names to scheduler classes
SCHEDULERS = {
    "linear": CondOTScheduler,
    "vp": VPScheduler,
}


# ============================================================================
# TangoFlux Prior Wrapper
# ============================================================================

class TangoFluxPrior:
    """Wraps TangoFlux for use with inference-time scaling algorithms."""

    def __init__(
        self,
        model: TangoFlux,
        vae,
        text_prompt,
        duration=10,
        guidance_scale=4.5,
        num_inference_steps=50,
        interpolant="linear",
        sample_method="ode",
        diffusion_norm=0.5,
        diffusion_coefficient="sigma",
        device=None,
    ):
        self.model = model
        self.vae = vae
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.interpolant = interpolant          # "linear" or "vp"
        self.sample_method = sample_method      # "ode" or "sde"
        self.diffusion_norm = diffusion_norm    # noise scale for SDE
        self.diffusion_coefficient = diffusion_coefficient
        self.device = device or model.transformer.device
        self.audio_seq_len = model.audio_seq_len

        # --- Set up schedulers ---
        self.original_scheduler = CondOTScheduler()     # model's native schedule
        if interpolant != "linear":
            scheduler_cls = SCHEDULERS.get(interpolant)
            if scheduler_cls is None:
                raise ValueError(f"Unknown interpolant: {interpolant}. Choose from {list(SCHEDULERS.keys())}")
            self.new_scheduler = scheduler_cls()
        else:
            self.new_scheduler = None  # No conversion needed

        # --- Pre-encode text and duration (done once) ---
        if not isinstance(text_prompt, list):
            text_prompt = [text_prompt]
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor([duration], device=self.device)
        else:
            duration = duration.to(self.device)

        self.classifier_free_guidance = guidance_scale > 1.0
        duration_hidden_states = model.encode_duration(duration).to(self.device)

        if self.classifier_free_guidance:
            bsz = 2  # uncond + cond
            encoder_hidden_states, boolean_encoder_mask = (
                model.encode_text_classifier_free(text_prompt, num_samples_per_prompt=1)
            )
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)
        else:
            bsz = 1
            encoder_hidden_states, boolean_encoder_mask = model.encode_text(text_prompt)

        # Pooled projection (mean-pooled text embeddings → FC)
        encoder_hidden_states = encoder_hidden_states.to(self.device)
        boolean_encoder_mask = boolean_encoder_mask.to(self.device)
        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        nan_tensor = torch.tensor(float("nan"), device=self.device)
        masked_data = torch.where(mask_expanded, encoder_hidden_states, nan_tensor)
        pooled = torch.nanmean(masked_data, dim=1)
        self.pooled_projection = model.fc(pooled).to(self.device)

        # Concatenate duration to encoder hidden states
        self.encoder_hidden_states = torch.cat(
            [encoder_hidden_states, duration_hidden_states], dim=1
        ).to(self.device)

        # Position IDs
        self.txt_ids = torch.zeros(bsz, self.encoder_hidden_states.shape[1], 3).to(self.device)
        self.audio_ids = (
            torch.arange(self.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(self.device)
        )

        # --- Setup scheduler and timesteps ---
        self.noise_scheduler = model.noise_scheduler
        self._setup_timesteps()

    def _setup_timesteps(self):
        """Initialize timesteps and sigmas from the scheduler."""
        sigmas_np = np.linspace(1.0, 1.0 / self.num_inference_steps, self.num_inference_steps)
        self.timesteps, _ = retrieve_timesteps(
            self.noise_scheduler, self.num_inference_steps, self.device, sigmas=sigmas_np
        )
        # Precompute sigma values for each timestep
        # sigma_t = t / 1000 for flow matching (the scheduler stores sigmas this way)
        self.sigmas = self.timesteps / 1000.0  # shape: (num_steps,)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def init_latent(self, batch_size):
        """Sample initial Gaussian noise in the latent space."""
        return torch.randn(
            batch_size, self.audio_seq_len, 64,
            device=self.device,
        )

    @torch.no_grad()
    def compute_velocity(self, x_t, t):
        """Predict velocity field v(x_t, t) with optional classifier-free guidance.

        This operates in the model's NATIVE linear space — no scheduler conversion.

        Args:
            x_t: (B, seq_len, 64) noisy latents
            t:   scalar timestep value (from self.timesteps)
        Returns:
            velocity: (B, seq_len, 64) predicted velocity
        """
        B = x_t.shape[0]

        # Expand conditioning for batch size
        if self.classifier_free_guidance:
            uncond_enc, cond_enc = self.encoder_hidden_states.chunk(2)
            uncond_pool, cond_pool = self.pooled_projection.chunk(2)
            uncond_txt, cond_txt = self.txt_ids.chunk(2)
            
            enc_hs = torch.cat([uncond_enc.expand(B, -1, -1), cond_enc.expand(B, -1, -1)], dim=0)
            pooled = torch.cat([uncond_pool.expand(B, -1), cond_pool.expand(B, -1)], dim=0)
            txt_ids = torch.cat([uncond_txt.expand(B, -1, -1), cond_txt.expand(B, -1, -1)], dim=0)
            
            audio_ids = self.audio_ids.repeat(B, 1, 1)
            latents_input = torch.cat([x_t, x_t], dim=0)
        else:
            enc_hs = self.encoder_hidden_states.expand(B, -1, -1)
            pooled = self.pooled_projection.expand(B, -1)
            txt_ids = self.txt_ids.expand(B, -1, -1)
            audio_ids = self.audio_ids.expand(B, -1, -1)
            latents_input = x_t

        noise_pred = self.model.transformer(
            hidden_states=latents_input,
            timestep=torch.tensor([t / 1000.0], device=self.device),
            guidance=None,
            pooled_projections=pooled,
            encoder_hidden_states=enc_hs,
            txt_ids=txt_ids,
            img_ids=audio_ids,
            return_dict=False,
        )[0]

        if self.classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.no_grad()
    def compute_velocity_transformed(self, x_t, t_val):
        """Predict velocity with scheduler conversion (Linear → VP or other).

        Follows the approach from Flow-Inference-Time-Scaling:
        1. Compute new scheduler coefficients at time r = t_val/1000
        2. Map to original scheduler time via SNR matching
        3. Rescale latent: x_linear = x / s_r where s_r = σ_new / σ_orig
        4. Run model in native space at mapped time
        5. Transform velocity back to new scheduler frame

        If no scheduler conversion is active (interpolant="linear"), this
        falls through to regular compute_velocity().

        Args:
            x_t:   (B, seq_len, 64) noisy latents in new scheduler's space
            t_val: scalar timestep (in [0, 1000] range)
        Returns:
            velocity: (B, seq_len, 64) velocity in new scheduler's frame
        """
        if self.new_scheduler is None:
            return self.compute_velocity(x_t, t_val)

        r = torch.tensor(t_val / 1000.0, dtype=torch.float32, device=self.device)

        # Step 1: Get coefficients in the new scheduler at time r
        r_out = self.new_scheduler(r)
        alpha_r = r_out.alpha_t
        sigma_r = r_out.sigma_t
        d_alpha_r = r_out.d_alpha_t
        d_sigma_r = r_out.d_sigma_t

        # Step 2: Map to original scheduler time via SNR matching
        snr = alpha_r / sigma_r
        t_orig = self.original_scheduler.snr_inverse(snr)

        # Step 3: Get original scheduler coefficients at mapped time
        t_out = self.original_scheduler(t_orig)
        alpha_t = t_out.alpha_t
        sigma_t = t_out.sigma_t
        d_alpha_t = t_out.d_alpha_t
        d_sigma_t = t_out.d_sigma_t

        # Step 4: Compute rescaling factors
        s_r = sigma_r / sigma_t   # scale factor between schedulers

        # dt_r = dr/dt (how new scheduler time maps to original time rate)
        dt_r = (
            sigma_t * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        # ds_r = d(s_r)/dt
        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        # Step 5: Rescale latent and run model in native space
        x_linear = (x_t / s_r).to(x_t.dtype)
        t_mapped = (t_orig * 1000.0).item()
        u_t = self.compute_velocity(x_linear, t_mapped)

        # Step 6: Transform velocity back to new scheduler frame
        u_r = (ds_r * x_t / s_r + dt_r * s_r * u_t).to(x_t.dtype)

        return u_r

    def get_sigma(self, step_index):
        """Get sigma value for a given step index."""
        return self.sigmas[step_index].item()

    def get_tweedie(self, x_t, v_t, sigma_t):
        """Estimate clean sample x̂_0 from noisy x_t and predicted velocity.

        Uses the general formula that works with any scheduler:
            x̂_0 = (σ · v - dσ · x) / (dα · σ - dσ · α)

        When using scheduler conversion, the velocity v_t should be the
        TRANSFORMED velocity from compute_velocity_transformed().

        Args:
            x_t:     (B, seq, 64) noisy latent
            v_t:     (B, seq, 64) predicted velocity (in active scheduler's frame)
            sigma_t: float, time parameter (sigma = t/1000 for the active scheduler)
        Returns:
            x0_hat: (B, seq, 64) denoised estimate
        """
        scheduler = self.new_scheduler if self.new_scheduler is not None else self.original_scheduler
        t = torch.tensor(sigma_t, dtype=torch.float32, device=self.device)
        out = scheduler(t)

        alpha_t = out.alpha_t.to(x_t.dtype)
        sig_t = out.sigma_t.to(x_t.dtype)
        d_alpha_t = out.d_alpha_t.to(x_t.dtype)
        d_sigma_t = out.d_sigma_t.to(x_t.dtype)

        numer = sig_t * v_t - d_sigma_t * x_t
        denom = d_alpha_t * sig_t - d_sigma_t * alpha_t

        return numer / denom

    def _convert_velocity_to_score(self, x_t, v_t, sigma_t):
        """Convert velocity to score function analytically.

        From the probability flow ODE: v = dα·x₀ + dσ·ε = dα/dt · x₀ + dσ/dt · ε
        And x_t = α·x₀ + σ·ε, the score ∇log p(x_t) can be derived as:

            reverse_alpha_ratio = α / dα
            var = σ² - reverse_alpha_ratio · dσ · σ
            score = (reverse_alpha_ratio · v - x) / var

        Args:
            x_t:     (B, seq, 64) noisy latent
            v_t:     (B, seq, 64) predicted velocity
            sigma_t: float, time parameter for the active scheduler
        Returns:
            score: (B, seq, 64) estimated score function
        """
        scheduler = self.new_scheduler if self.new_scheduler is not None else self.original_scheduler
        t = torch.tensor(sigma_t, dtype=torch.float32, device=self.device)
        out = scheduler(t)

        alpha_t = out.alpha_t.to(x_t.dtype)
        sig_t = out.sigma_t.to(x_t.dtype)
        d_alpha_t = out.d_alpha_t.to(x_t.dtype)
        d_sigma_t = out.d_sigma_t.to(x_t.dtype)

        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sig_t ** 2 - reverse_alpha_ratio * d_sigma_t * sig_t
        var = torch.clamp(var, min=1e-8)

        score = (reverse_alpha_ratio * v_t - x_t) / var
        return score

    def _get_diffusion_coefficient(self, sigma_t):
        """Compute diffusion coefficient g(t) for SDE.

        Uses the same options as the reference implementation.

        Args:
            sigma_t: float, time parameter for the active scheduler
        Returns:
            g: float or tensor, diffusion coefficient
        """
        scheduler = self.new_scheduler if self.new_scheduler is not None else self.original_scheduler
        t = torch.tensor(sigma_t, dtype=torch.float32, device=self.device)

        if self.diffusion_coefficient == "sigma":
            out = scheduler(t)
            return self.diffusion_norm * out.sigma_t
        elif self.diffusion_coefficient == "constant":
            return self.diffusion_norm
        elif self.diffusion_coefficient == "linear":
            return self.diffusion_norm * t
        else:
            # Default to sigma-based
            out = scheduler(t)
            return self.diffusion_norm * out.sigma_t

    def euler_step(self, x_t, v_t, sigma_t, sigma_prev):
        """ODE Euler step from sigma_t to sigma_prev.

        For flow matching, the ODE is: dx = -v · dt
        where dt = (sigma_prev - sigma_t) / 1000 (since sigmas are t/1000)

        But we can simplify since sigma IS t/1000 for the native scheduler,
        so dt in sigma-space maps directly: x_next = x_t + v · (sigma_prev - sigma_t)

        This produces the standard Euler update used in FlowMatchEulerDiscreteScheduler.

        Args:
            x_t:        (B, seq, 64) current noisy latent
            v_t:        (B, seq, 64) predicted velocity
            sigma_t:    float, current sigma (= t/1000)
            sigma_prev: float, target sigma (< sigma_t, denoising direction)
        Returns:
            x_next: (B, seq, 64) updated latent
        """
        dt = sigma_prev - sigma_t  # negative (denoising)
        return x_t + v_t * dt

    def sde_step(self, x_t, v_t, sigma_t, sigma_prev):
        """SDE Euler-Maruyama step with proper drift correction.

        From the paper (Eq. 7-9):
            f_t(x_t) = u_t(x_t) - g²_t/2 · ∇log p_t(x_t)     (Eq. 7)
            p(x_{t-Δt}|x_t) = N(x_t - f_t·Δt, g²_t·Δt·I)     (Eq. 9)

        Expanding:
            x_{t-Δt} = x_t - (u - g²/2·score)·Δt + g·√Δt·noise
                     = x_t - u·Δt + g²/2·score·Δt + g·√Δt·noise

        Where Δt > 0 is the step size (sigma_t - sigma_prev).

        The score is computed analytically from velocity (Eq. 8),
        requiring no extra model forward pass.

        Args:
            x_t, v_t, sigma_t, sigma_prev: same as euler_step
        Returns:
            x_next: (B, seq, 64) updated latent with stochastic noise
        """
        dt_abs = sigma_t - sigma_prev  # positive step size (Δt > 0)

        # Compute score from velocity analytically (Eq. 8, no extra forward pass)
        score = self._convert_velocity_to_score(x_t, v_t, sigma_t)

        # Compute diffusion coefficient g(t)
        g = self._get_diffusion_coefficient(sigma_t)

        # Deterministic part: x_t - u·Δt + g²/2·score·Δt  (from Eq. 9)
        x_mean = x_t - v_t * dt_abs + 0.5 * g ** 2 * score * dt_abs

        # Stochastic part: g · √Δt · noise
        noise = torch.randn_like(x_t)
        x_next = x_mean + g * torch.sqrt(dt_abs if isinstance(dt_abs, torch.Tensor)
                                          else torch.tensor(dt_abs, device=x_t.device)) * noise

        return x_next

    def step(self, x_t, v_t, sigma_t, sigma_prev):
        """Take one denoising step using the configured sample method (ODE or SDE)."""
        if self.sample_method == "ode":
            return self.euler_step(x_t, v_t, sigma_t, sigma_prev)
        elif self.sample_method == "sde":
            return self.sde_step(x_t, v_t, sigma_t, sigma_prev)
        else:
            raise ValueError(f"Unknown sample_method: {self.sample_method}")

    @torch.no_grad()
    def decode_latent(self, latent):
        """Decode latent (B, seq, 64) to waveform (B, channels, samples)."""
        return self.vae.decode(latent.transpose(2, 1)).sample

    @torch.no_grad()
    def decode_and_trim(self, latent, duration, sample_rate=44100):
        """Decode latent to waveform and trim to specified duration."""
        wave = self.decode_latent(latent).cpu()
        waveform_end = int(duration * sample_rate)
        return wave[:, :, :waveform_end]