"""
Inference-Time Scaling Algorithms for TangoFlux.

Implements four sampling strategies:
  1. BoN (Best-of-N)             — run N full trajectories, pick the best
  2. Particle Filtering (SMC)    — step-wise resampling with reward guidance
  3. RBF (Rollover Budget Forcing) — dynamic budget allocation per timestep
  4. DPS (Diffusion Posterior Sampling) — gradient-guided denoising

All functions take a TangoFluxPrior and a reward model as arguments.
"""

import random
import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================================
# 1. Best-of-N (BoN)
# ============================================================================

def inference_bon(
    prior,
    reward_model,
    prompt,
    n_samples=8,
    duration=10,
    sample_rate=44100,
    seed=42,
):
    """Best-of-N: generate N independent samples, return the one with highest reward.

    This is the simplest baseline — no interaction between samples.
    Cost: N × standard inference cost.

    Args:
        prior:        TangoFluxPrior (configured with steps, interpolant, etc.)
        reward_model: callable(waveforms, text_prompt) → (N,) scores
        prompt:       str, text prompt
        n_samples:    int, number of independent samples to generate
        duration:     float, audio duration in seconds
        sample_rate:  int, output sample rate
        seed:         int, base random seed

    Returns:
        best_waveform:  (1, channels, samples) best audio
        best_score:     float, reward score of best sample
        all_scores:     dict with per-sample scores
    """
    all_latents = []

    for k in range(n_samples):
        torch.manual_seed(seed + k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + k)

        # Run full denoising trajectory
        latents = prior.init_latent(1)  # (1, seq, 64)
        for step_idx, t in enumerate(prior.timesteps):
            t_val = t.item()
            sigma_t = prior.get_sigma(step_idx)
            sigma_prev = prior.get_sigma(step_idx + 1) if step_idx + 1 < len(prior.sigmas) else 0.0

            velocity = prior.compute_velocity_transformed(latents, t_val)
            latents = prior.step(latents, velocity, sigma_t, sigma_prev)

        all_latents.append(latents)

    # Decode and score ONE AT A TIME to avoid VAE OOM
    # (Batched VAE decode intermediate activations can be several GB)
    all_scores = []
    best_waveform = None
    best_score = float("-inf")

    for k, lat in enumerate(all_latents):
        with torch.no_grad():
            waveform = prior.decode_and_trim(lat, duration, sample_rate)  # (1, ch, samples)
            score = reward_model(waveform, prompt)  # (1,)
            s = score.item()
            all_scores.append(s)

            if s > best_score:
                best_score = s
                best_waveform = waveform.cpu()

        # Free GPU memory from this decode
        del waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_waveform, best_score, torch.tensor(all_scores)


# ============================================================================
# 2. Particle Filtering (SMC-style)
# ============================================================================

def inference_particle_filtering(
    prior,
    reward_model,
    prompt,
    n_particles=4,
    duration=10,
    sample_rate=44100,
    temperature=1.0,
    eval_interval=5,
    warmup_steps=0,
    seed=42,
):
    """Step-wise particle filtering with reward-guided resampling.

    Runs `warmup_steps` plain denoising steps first (no scoring/resampling), then
    every `eval_interval` steps, decodes x̂_0 estimates, evaluates rewards, and
    resamples particles proportional to softmax(rewards / temperature). The
    warmup gives the Tweedie estimate enough signal to be informative — at very
    high noise levels x̂_0 is essentially random and the reward carries no signal.

    Requires SDE mode (prior.sample_method="sde") for diversity after resampling.

    Args:
        prior:          TangoFluxPrior
        reward_model:   callable(waveforms, prompt) → (N,) scores
        prompt:         str
        n_particles:    int, number of parallel particles
        duration:       float, seconds
        sample_rate:    int
        temperature:    float, softmax temperature for resampling
        eval_interval:  int, evaluate rewards every K steps after warmup
        warmup_steps:   int, denoising steps before search begins (no eval/resample)
        seed:           int

    Returns:
        best_waveform: (1, ch, samples)
        best_score:    float
        all_scores:    (N,) tensor of final scores
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize N particles
    latents = prior.init_latent(n_particles)  # (N, seq, 64)

    num_steps = len(prior.timesteps)
    progress = tqdm(range(num_steps), desc="Particle Filtering", leave=False)

    for step_idx in progress:
        t_val = prior.timesteps[step_idx].item()
        sigma_t = prior.get_sigma(step_idx)
        sigma_prev = prior.get_sigma(step_idx + 1) if step_idx + 1 < len(prior.sigmas) else 0.0

        # Predict velocity for all particles
        velocity = prior.compute_velocity_transformed(latents, t_val)  # (N, seq, 64)

        # Reward evaluation + resampling: only after warmup, at eval cadence,
        # and never on the last step (final scoring happens after the loop).
        is_eval_step = (
            step_idx >= warmup_steps
            and ((step_idx - warmup_steps) % eval_interval == 0)
            and step_idx < num_steps - 1
        )
        if is_eval_step:
            # Estimate x̂_0 for each particle
            x0_hat = prior.get_tweedie(latents, velocity, sigma_t)

            # Decode and score ONE AT A TIME to avoid VAE OOM
            # (batched VAE decode allocates several GB of activations per particle)
            rewards_list = []
            with torch.no_grad():
                for i in range(n_particles):
                    wf = prior.decode_and_trim(x0_hat[i : i + 1], duration, sample_rate)
                    rewards_list.append(reward_model(wf, prompt).item())
                    del wf
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            rewards = torch.tensor(rewards_list)

            # Resample particles proportional to rewards
            weights = torch.softmax(rewards / temperature, dim=0)
            indices = torch.multinomial(weights, n_particles, replacement=True)
            latents = latents[indices]
            velocity = velocity[indices]

            progress.set_postfix(
                best_r=f"{rewards.max().item():.3f}",
                mean_r=f"{rewards.mean().item():.3f}",
            )

        # Take denoising step
        latents = prior.step(latents, velocity, sigma_t, sigma_prev)

    # Final selection — also chunked to avoid VAE OOM
    final_scores_list = []
    waveforms_cpu = []
    with torch.no_grad():
        for i in range(n_particles):
            wf = prior.decode_and_trim(latents[i : i + 1], duration, sample_rate)
            final_scores_list.append(reward_model(wf, prompt).item())
            waveforms_cpu.append(wf.cpu())
            del wf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    final_scores = torch.tensor(final_scores_list)
    best_idx = final_scores.argmax().item()

    return waveforms_cpu[best_idx], final_scores[best_idx].item(), final_scores


# ============================================================================
# 3. Rollover Budget Forcing (RBF)
# ============================================================================

def inference_rbf(
    prior,
    reward_model,
    prompt,
    init_n_particles=25,
    max_nfe=500,
    duration=10,
    sample_rate=44100,
    eval_interval=5,
    seed=42,
):
    """Rollover Budget Forcing: dynamic budget allocation per timestep.

    Starts by generating `init_n_particles` at the first step, picks the best,
    then at subsequent steps tries new trajectories and "rolls over" unused budget
    to harder timesteps.

    Much more efficient than BoN or SMC for the same total budget.

    Args:
        prior:              TangoFluxPrior
        reward_model:       callable(waveforms, prompt) → (N,) scores
        prompt:             str
        init_n_particles:   int, particles at first step
        max_nfe:            int, total NFE budget (each velocity call = 1 NFE, 2 with CFG)
        duration:           float, seconds
        sample_rate:        int
        eval_interval:      int, steps between reward evaluations
        seed:               int

    Returns:
        best_waveform: (1, ch, samples)
        best_score:    float
        history:       dict with per-step info
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_steps = len(prior.timesteps)
    nfe_per_call = 2 if prior.classifier_free_guidance else 1  # CFG doubles NFE
    nfe_used = 0
    history = {"steps": [], "rewards": [], "nfe": []}

    # --- Phase 1: Initial particle generation at step 0 ---
    latents_init = prior.init_latent(init_n_particles)  # (P, seq, 64)

    # Run all initial particles through the first few steps until first eval
    first_eval_step = min(eval_interval, num_steps - 1)
    current_latents = latents_init

    for step_idx in range(first_eval_step + 1):
        t_val = prior.timesteps[step_idx].item()
        sigma_t = prior.get_sigma(step_idx)
        sigma_prev = prior.get_sigma(step_idx + 1) if step_idx + 1 < len(prior.sigmas) else 0.0

        velocity = prior.compute_velocity_transformed(current_latents, t_val)
        nfe_used += nfe_per_call * init_n_particles
        current_latents = prior.step(current_latents, velocity, sigma_t, sigma_prev)

    # Evaluate and select best with chunking to prevent OOM
    x0_candidates = current_latents
    rewards_list = []
    
    with torch.no_grad():
        for i in range(init_n_particles):
            waveforms = prior.decode_and_trim(x0_candidates[i:i+1], duration, sample_rate)
            r = reward_model(waveforms, prompt)
            rewards_list.append(r.item())
            del waveforms
            torch.cuda.empty_cache()

    rewards = torch.tensor(rewards_list)

    best_idx = rewards.argmax().item()
    best_latent = current_latents[best_idx : best_idx + 1]  # (1, seq, 64)
    best_reward = rewards[best_idx].item()
    history["steps"].append(first_eval_step)
    history["rewards"].append(best_reward)
    history["nfe"].append(nfe_used)

    # --- Phase 2: Rollover budget forcing for remaining steps ---
    remaining_steps = num_steps - first_eval_step - 1
    if remaining_steps <= 0:
        waveform = prior.decode_and_trim(best_latent, duration, sample_rate)
        return waveform, best_reward, history

    # Budget per remaining eval window
    eval_windows = max(1, remaining_steps // eval_interval)
    budget_per_window = max(1, (max_nfe - nfe_used) // (eval_windows * nfe_per_call * eval_interval))

    progress = tqdm(
        range(first_eval_step + 1, num_steps),
        desc="RBF",
        leave=False,
    )

    current_best = best_latent  # (1, seq, 64)
    rollover_budget = 0

    for step_idx in progress:
        t_val = prior.timesteps[step_idx].item()
        sigma_t = prior.get_sigma(step_idx)
        sigma_prev = prior.get_sigma(step_idx + 1) if step_idx + 1 < len(prior.sigmas) else 0.0

        if nfe_used >= max_nfe:
            # Budget exhausted — just take deterministic steps
            velocity = prior.compute_velocity_transformed(current_best, t_val)
            nfe_used += nfe_per_call
            current_best = prior.step(current_best, velocity, sigma_t, sigma_prev)
            continue

        # Determine how many candidates to try at this step
        n_candidates = budget_per_window + rollover_budget
        n_candidates = min(n_candidates, (max_nfe - nfe_used) // nfe_per_call)
        n_candidates = max(1, n_candidates)

        # Duplicate best latent and add SDE noise for diversity
        candidates = current_best.repeat(n_candidates, 1, 1)  # (K, seq, 64)
        if n_candidates > 1 and prior.sample_method == "sde":
            # Add diversity noise to candidates (except the first which stays deterministic)
            noise = torch.randn_like(candidates[1:]) * prior.diffusion_norm * sigma_t
            candidates[1:] = candidates[1:] + noise

        # Forward pass for all candidates
        velocity = prior.compute_velocity_transformed(candidates, t_val)
        nfe_used += nfe_per_call * n_candidates
        next_latents = prior.step(candidates, velocity, sigma_t, sigma_prev)

        # Evaluate at eval intervals
        if step_idx % eval_interval == 0 or step_idx == num_steps - 1:
            x0_hat = prior.get_tweedie(next_latents, velocity, sigma_t)
            
            # Chunked evaluation to prevent OOM
            new_rewards_list = []
            with torch.no_grad():
                for i in range(n_candidates):
                    waveforms = prior.decode_and_trim(x0_hat[i:i+1], duration, sample_rate)
                    r = reward_model(waveforms, prompt)
                    new_rewards_list.append(r.item())
                    del waveforms
                    torch.cuda.empty_cache()
                    
            rewards_tensor = torch.tensor(new_rewards_list)
            new_best_idx = rewards_tensor.argmax().item()
            new_best_reward = new_rewards_list[new_best_idx]

            if new_best_reward > best_reward:
                # Rollover: found improvement, save unused budget
                rollover_budget += max(0, n_candidates - new_best_idx - 1)
                best_reward = new_best_reward
                current_best = next_latents[new_best_idx : new_best_idx + 1]
            else:
                # No improvement: consume budget, no rollover
                rollover_budget = 0
                current_best = next_latents[0:1]  # take deterministic one

            history["steps"].append(step_idx)
            history["rewards"].append(best_reward)
            history["nfe"].append(nfe_used)

            progress.set_postfix(
                best_r=f"{best_reward:.3f}",
                nfe=nfe_used,
                budget=rollover_budget,
            )
        else:
            # Not an eval step — just take the first (deterministic) candidate
            current_best = next_latents[0:1]

    waveform = prior.decode_and_trim(current_best, duration, sample_rate)
    return waveform, best_reward, history


# ============================================================================
# 4. Diffusion Posterior Sampling (DPS)
# ============================================================================

def inference_dps(
    prior,
    reward_model,
    prompt,
    duration=10,
    sample_rate=44100,
    guidance_strength=1.0,
    eval_interval=5,
    seed=42,
):
    """DPS: gradient-guided denoising via backprop through reward model.

    At each eval step, computes ∇_{x_t} reward(decode(x̂_0)) and adds the
    gradient to the latent update, steering toward higher reward.

    Requires differentiable reward model and VAE decode.

    Args:
        prior:              TangoFluxPrior
        reward_model:       callable(waveforms, prompt) → scalar (must be differentiable)
        prompt:             str
        duration:           float, seconds
        sample_rate:        int
        guidance_strength:  float, scale for reward gradient
        eval_interval:      int, compute gradients every K steps
        seed:               int

    Returns:
        waveform:   (1, ch, samples)
        score:      float, final reward score
        history:    dict
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_steps = len(prior.timesteps)
    latents = prior.init_latent(1)  # (1, seq, 64)
    history = {"steps": [], "rewards": []}

    progress = tqdm(range(num_steps), desc="DPS", leave=False)

    for step_idx in progress:
        t_val = prior.timesteps[step_idx].item()
        sigma_t = prior.get_sigma(step_idx)
        sigma_prev = prior.get_sigma(step_idx + 1) if step_idx + 1 < len(prior.sigmas) else 0.0

        apply_guidance = (step_idx % eval_interval == 0) and guidance_strength > 0

        if apply_guidance:
            # Need gradients for DPS guidance
            latents = latents.detach().requires_grad_(True)

            # NOTE: DPS gradient path runs in native linear space (no scheduler
            # conversion). VP interpolant is not supported with DPS guidance —
            # the scheduler conversion transforms are not differentiable in this
            # implementation. For VP + guidance, consider using particle_filter or rbf instead.
            # Forward pass with gradients (no @torch.no_grad on compute_velocity here)
            with torch.enable_grad():
                velocity = prior.model.transformer(
                    hidden_states=torch.cat([latents, latents]) if prior.classifier_free_guidance else latents,
                    timestep=torch.tensor([t_val / 1000.0], device=prior.device),
                    guidance=None,
                    pooled_projections=prior.pooled_projection.repeat(latents.shape[0], 1),
                    encoder_hidden_states=prior.encoder_hidden_states.repeat(latents.shape[0], 1, 1),
                    txt_ids=prior.txt_ids.repeat(latents.shape[0], 1, 1),
                    img_ids=prior.audio_ids.repeat(latents.shape[0], 1, 1),
                    return_dict=False,
                )[0]

                if prior.classifier_free_guidance:
                    v_uncond, v_text = velocity.chunk(2)
                    velocity = v_uncond + prior.guidance_scale * (v_text - v_uncond)

                x0_hat = prior.get_tweedie(latents, velocity, sigma_t)

                # Decode and score (with gradients flowing through)
                waveform = prior.vae.decode(x0_hat.transpose(2, 1)).sample
                waveform_trimmed = waveform[:, :, : int(duration * sample_rate)]
                reward = reward_model(waveform_trimmed, prompt)

                if reward.dim() > 0:
                    reward = reward.sum()

                # Gradient of reward w.r.t. latents
                grad = torch.autograd.grad(reward, latents)[0]

            history["steps"].append(step_idx)
            history["rewards"].append(reward.item())

            # Euler step + gradient guidance
            latents = latents.detach()
            velocity = velocity.detach()

            alpha_t = max(sigma_t, 0.01)  # scale gradient by noise level
            latents = prior.euler_step(latents, velocity, sigma_t, sigma_prev)
            latents = latents + guidance_strength * grad * alpha_t

            progress.set_postfix(r=f"{reward.item():.3f}")

        else:
            # Standard step without guidance
            velocity = prior.compute_velocity_transformed(latents, t_val)
            latents = prior.step(latents, velocity, sigma_t, sigma_prev)

    # Final decode
    waveform = prior.decode_and_trim(latents.detach(), duration, sample_rate)
    final_score = reward_model(waveform, prompt)
    if final_score.dim() > 0:
        final_score = final_score.mean().item()
    else:
        final_score = final_score.item()

    return waveform, final_score, history


# ============================================================================
# 5. Zero-Order Search
# ============================================================================

def inference_zero_order(
    prior,
    reward_model,
    prompt,
    n_candidates=4,
    search_steps=5,
    search_radius=0.95,
    randomize_pivot=True,
    duration=10,
    sample_rate=44100,
    seed=42,
):
    """Zero-Order Search: Local search over the initial noise space.

    Follows "Inference-Time Scaling for Diffusion Models beyond Scaling
    Denoising Steps" (arXiv 2501.09732).

    1. Sample a random Gaussian noise as pivot.
    2. Generate N neighbors via spherical interpolation (same norm, cosine
       similarity = search_radius with pivot).
    3. Run pivot + neighbors through the full ODE solver, decode, and score.
    4. If the best neighbor beats the pivot, update pivot. Otherwise, resample
       a fresh random pivot for the next round.

    Args:
        search_radius: float in (0, 1), the cosine similarity threshold between
                       the pivot and each neighbor (called 'threshold' in the paper).
                       Higher = neighbors are closer to the pivot.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Start with random Gaussian noise as pivot
    pivot = prior.init_latent(1)  # (1, seq, 64)
    num_steps = len(prior.timesteps)

    best_score = float("-inf")
    best_waveform = None
    history = {"steps": [], "rewards": []}

    progress = tqdm(range(search_steps), desc="Zero-Order Search", leave=False)

    for search_step in progress:
        # 2. Generate N neighbors via spherical interpolation
        #    neighbor = ||pivot|| * (threshold * û + sqrt(1 - threshold²) * ŵ)
        #    where û = unit direction of pivot, ŵ = random orthogonal unit direction
        #    This preserves the norm and controls cosine similarity.
        pivot_flat = pivot.view(1, -1).to(torch.float64)  # (1, D)
        pivot_norm = torch.linalg.norm(
            pivot_flat, dim=-1, keepdim=True
        ).unsqueeze(-2)  # (1, 1, 1)
        u = pivot_flat.unsqueeze(-2) / pivot_norm.clamp_min(1e-12)  # (1, 1, D)

        # Random directions
        v = torch.from_numpy(
            np.random.standard_normal(
                size=(1, n_candidates, pivot_flat.shape[-1])
            )
        ).to(device=pivot.device, dtype=torch.float64)  # (1, N, D)

        # Gram-Schmidt: remove component along û, then normalize
        w = F.normalize(v - (v @ u.transpose(-2, -1)) * u, dim=-1)  # (1, N, D)

        # Spherical interpolation
        neighbors_flat = pivot_norm * (
            search_radius * u
            + math.sqrt(1 - search_radius ** 2) * w
        )  # (1, N, D)
        neighbors = neighbors_flat.reshape(
            n_candidates, *pivot.shape[1:]
        ).to(pivot.dtype)  # (N, seq, 64)

        # Combine: pivot (index 0) + neighbors for evaluation
        candidates = torch.cat([pivot, neighbors], dim=0)  # (1+N, seq, 64)
        total_candidates = candidates.shape[0]

        # 3. Run all candidates through the full ODE solver
        latents = candidates
        for step_idx in range(num_steps):
            t_val = prior.timesteps[step_idx].item()
            sigma_t = prior.get_sigma(step_idx)
            sigma_prev = (
                prior.get_sigma(step_idx + 1)
                if step_idx + 1 < len(prior.sigmas)
                else 0.0
            )

            velocity = prior.compute_velocity_transformed(latents, t_val)
            latents = prior.step(latents, velocity, sigma_t, sigma_prev)

        # Decode and score
        rewards_list = []
        with torch.no_grad():
            for i in range(total_candidates):
                waveform = prior.decode_and_trim(
                    latents[i : i + 1], duration, sample_rate
                )
                reward = reward_model(waveform, prompt)
                rewards_list.append(reward.item())

                # Keep track of global best to return
                if reward.item() > best_score:
                    best_score = reward.item()
                    best_waveform = waveform.cpu()

                del waveform
                torch.cuda.empty_cache()

        # 4. Update pivot only if a neighbor improved over the pivot score.
        #    Otherwise reject this round and resample a fresh random pivot.
        pivot_score = rewards_list[0]
        neighbor_scores = rewards_list[1:]  # N neighbor scores
        improved = False

        if neighbor_scores and max(neighbor_scores) > pivot_score:
            best_neighbor_idx = torch.tensor(neighbor_scores).argmax().item()
            pivot = neighbors[best_neighbor_idx : best_neighbor_idx + 1]
            improved = True
        else:
            # No improvement → conditionally reject round, sample fresh random noise
            if randomize_pivot:
                pivot = prior.init_latent(1)

        step_best_score = max(rewards_list)
        history["steps"].append(search_step)
        history["rewards"].append(step_best_score)

        progress.set_postfix(
            step_best_r=f"{step_best_score:.3f}",
            best_r=f"{best_score:.3f}",
            improved="Y" if improved else "N",
        )

    return best_waveform, best_score, history


# ============================================================================
# 6. Particle Swarm Optimization (PSO)
# ============================================================================

def inference_pso(
    prior,
    reward_model,
    prompt,
    n_particles=4,
    search_steps=5,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    duration=10,
    sample_rate=44100,
    seed=42,
):
    """Particle Swarm Optimization over the initial-noise latent space.

    Each particle is a candidate noise latent x_i in R^d (d = seq_len * 64).
    Every iteration we run a full ODE denoise + decode + reward for each
    particle, then update each particle with the standard PSO velocity rule:

        v_i ← w * v_i + c1 * r1 * (p_i - x_i) + c2 * r2 * (g - x_i)
        x_i ← x_i + v_i
        x_i ← x_i * sqrt(d) / ||x_i||      # project back to noise sphere

    where p_i is the particle's personal best, g is the swarm's global best,
    r1, r2 ~ U(0, 1) drawn elementwise. The final projection keeps each
    particle on the typical sphere of a standard d-dim Gaussian, so the
    diffusion model still sees in-distribution noise.

    Args:
        n_particles:   swarm size (NFE per iter = n_particles * num_steps).
        search_steps:  number of PSO iterations.
        inertia:       w in [0, 1]. Higher = more momentum, more exploration.
        cognitive:     c1. Pull toward each particle's own best.
        social:        c2. Pull toward the swarm's global best.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Initialize swarm: N random Gaussian latents, zero velocity.
    particles = torch.cat(
        [prior.init_latent(1) for _ in range(n_particles)], dim=0
    )  # (N, seq, 64)
    velocities = torch.zeros_like(particles)

    # Per-particle best position/score; swarm-global best position/score.
    pbest_pos = particles.clone()
    pbest_score = torch.full(
        (n_particles,), float("-inf"), device=particles.device
    )
    gbest_score = float("-inf")
    gbest_waveform = None
    gbest_pos = particles[0:1].clone()

    # Norm to project particles back onto: typical norm of N(0, I_d).
    d = particles[0].numel()
    target_norm = math.sqrt(d)

    num_steps = len(prior.timesteps)
    history = {"steps": [], "rewards": []}
    progress = tqdm(range(search_steps), desc="PSO Search", leave=False)

    for search_step in progress:
        # 2. Run ODE denoise on every particle.
        latents = particles.clone()
        for step_idx in range(num_steps):
            t_val = prior.timesteps[step_idx].item()
            sigma_t = prior.get_sigma(step_idx)
            sigma_prev = (
                prior.get_sigma(step_idx + 1)
                if step_idx + 1 < len(prior.sigmas)
                else 0.0
            )
            velocity = prior.compute_velocity_transformed(latents, t_val)
            latents = prior.step(latents, velocity, sigma_t, sigma_prev)

        # 3. Decode + score each particle.
        rewards_list = []
        with torch.no_grad():
            for i in range(n_particles):
                waveform = prior.decode_and_trim(
                    latents[i : i + 1], duration, sample_rate
                )
                reward = reward_model(waveform, prompt).item()
                rewards_list.append(reward)

                # Update personal best.
                if reward > pbest_score[i].item():
                    pbest_score[i] = reward
                    pbest_pos[i] = particles[i].clone()

                # Update global best (and keep the audio for return).
                if reward > gbest_score:
                    gbest_score = reward
                    gbest_pos = particles[i : i + 1].clone()
                    gbest_waveform = waveform.cpu()

                del waveform
                torch.cuda.empty_cache()

        # 4. PSO velocity + position update.
        r1 = torch.rand_like(particles)
        r2 = torch.rand_like(particles)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest_pos - particles)
            + social * r2 * (gbest_pos - particles)
        )
        particles = particles + velocities

        # 5. Project each particle back to the noise sphere of radius sqrt(d).
        flat = particles.view(n_particles, -1)
        flat = flat * (target_norm / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        particles = flat.view_as(pbest_pos)

        step_best_score = max(rewards_list)
        history["steps"].append(search_step)
        history["rewards"].append(step_best_score)

        progress.set_postfix(
            step_best_r=f"{step_best_score:.3f}",
            best_r=f"{gbest_score:.3f}",
        )

    return gbest_waveform, gbest_score, history


# ============================================================================
# Dispatch map
# ============================================================================

SCALING_METHODS = {
    "bon": inference_bon,
    "particle_filter": inference_particle_filtering,
    "zero_order": inference_zero_order,
    "pso": inference_pso,
}
