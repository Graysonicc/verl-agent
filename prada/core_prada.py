# Copyright 2026 PRADA contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class PradaLiteConfig:
    top_k: int = 16
    bootstrap_rounds: int = 16
    temporal_band: int = 2
    tau: float = 0.2
    alpha0: float = 1.0
    beta0: float = 1.0
    lambda_u: float = 0.2
    trace_gamma: float = 0.95
    trace_lambda: float = 0.8
    epsilon: float = 1e-6
    hash_dim: int = 4096
    seed: int = 0
    normalize: bool = False
    min_success_reward: Optional[float] = None
    exclude_same_traj: bool = True
    responsibility_mix: float = 0.3


def _as_float_array(values: Iterable, name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PRADA-lite expects numeric `{name}` values.") from exc


def _valid_token_list(ids: torch.Tensor, mask: Optional[torch.Tensor], max_items: int = 512) -> list[int]:
    if mask is not None:
        ids = ids[mask.bool()]
    ids = ids.detach().cpu().to(torch.long)
    if ids.numel() > max_items:
        ids = ids[-max_items:]
    return [int(x) for x in ids.tolist()]


def _hashed_bow(tokens: list[int], hash_dim: int) -> dict[int, float]:
    counts: dict[int, float] = {}
    for tok in tokens:
        if tok < 0:
            continue
        bucket = tok % hash_dim
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    norm = float(np.sqrt(sum(v * v for v in counts.values())))
    if norm <= 0.0:
        return {}
    return {k: v / norm for k, v in counts.items()}


def _sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


def _trajectory_steps(traj_index: np.ndarray) -> np.ndarray:
    counters: dict[object, int] = defaultdict(int)
    steps = np.zeros(len(traj_index), dtype=np.int32)
    for i, traj_uid in enumerate(traj_index):
        steps[i] = counters[traj_uid]
        counters[traj_uid] += 1
    return steps


def _trajectory_outcome_rewards(token_level_rewards: torch.Tensor, traj_index: np.ndarray) -> np.ndarray:
    scores = token_level_rewards.detach().sum(dim=-1).cpu().numpy().astype(np.float32)
    traj_rewards: dict[object, float] = {}
    for score, traj_uid in zip(scores, traj_index):
        traj_rewards.setdefault(traj_uid, float(score))
    return np.asarray([traj_rewards[traj_uid] for traj_uid in traj_index], dtype=np.float32)


def _success_labels(rewards: np.ndarray, min_success_reward: Optional[float]) -> np.ndarray:
    if min_success_reward is not None:
        return (rewards >= float(min_success_reward)).astype(np.float32)
    unique = np.unique(rewards)
    if unique.size <= 2 and np.all((unique >= 0.0) & (unique <= 1.0)):
        return rewards.astype(np.float32)
    max_reward = float(np.max(rewards)) if rewards.size else 0.0
    if max_reward <= 0.0:
        return np.zeros_like(rewards, dtype=np.float32)
    return (rewards > 0.0).astype(np.float32)


def _group_advantages(rewards: np.ndarray, group_index: np.ndarray, traj_index: np.ndarray, epsilon: float) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    for uid in np.unique(group_index):
        idx = np.where(group_index == uid)[0]
        traj_first_rows = []
        seen_traj = set()
        for row in idx:
            traj_uid = traj_index[row]
            if traj_uid in seen_traj:
                continue
            seen_traj.add(traj_uid)
            traj_first_rows.append(row)
        group_rewards = rewards[np.asarray(traj_first_rows, dtype=np.int64)]
        mean = float(np.mean(group_rewards))
        std = float(np.std(group_rewards))
        advantages[idx] = (rewards[idx] - mean) / (std + epsilon)
    return advantages


def _bootstrap_success_proxy(
    neighbor_indices: list[int],
    neighbor_weights: np.ndarray,
    success: np.ndarray,
    cfg: PradaLiteConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(neighbor_indices) == 0:
        prior = cfg.alpha0 / (cfg.alpha0 + cfg.beta0)
        return float(prior), float(prior * (1.0 - prior))

    weights = np.asarray(neighbor_weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    labels = success[np.asarray(neighbor_indices, dtype=np.int64)].astype(np.float64)

    def weighted_rate(sample_idx: np.ndarray) -> float:
        sample_weights = weights[sample_idx]
        sample_labels = labels[sample_idx]
        numerator = cfg.alpha0 + float(np.sum(sample_weights * sample_labels))
        denominator = cfg.alpha0 + cfg.beta0 + float(np.sum(sample_weights))
        return numerator / max(denominator, cfg.epsilon)

    rounds = max(int(cfg.bootstrap_rounds), 0)
    if rounds <= 1 or len(neighbor_indices) == 1:
        p_hat = weighted_rate(np.arange(len(neighbor_indices)))
        eff_n = (weights.sum() ** 2) / max(float(np.sum(weights * weights)), cfg.epsilon)
        variance = p_hat * (1.0 - p_hat) / max(eff_n + cfg.alpha0 + cfg.beta0 + 1.0, 1.0)
        return float(p_hat), float(max(variance, cfg.epsilon))

    estimates = np.zeros(rounds, dtype=np.float64)
    for b in range(rounds):
        sample_idx = rng.choice(len(neighbor_indices), size=len(neighbor_indices), replace=True, p=probs)
        estimates[b] = weighted_rate(sample_idx)
    return float(np.mean(estimates)), float(max(np.var(estimates), cfg.epsilon))


def _build_prefix_features(
    prompts: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    hash_dim: int,
) -> list[dict[int, float]]:
    prompt_len = prompts.size(1)
    features = []
    for i in range(prompts.size(0)):
        prompt_mask = attention_mask[i, :prompt_len] if attention_mask is not None else None
        response_mask = attention_mask[i, prompt_len:] if attention_mask is not None else None
        tokens = _valid_token_list(prompts[i], prompt_mask)
        tokens += _valid_token_list(responses[i], response_mask, max_items=128)
        features.append(_hashed_bow(tokens, hash_dim))
    return features


def _local_success_proxy(
    prompts: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    group_index: np.ndarray,
    traj_index: np.ndarray,
    traj_steps: np.ndarray,
    success: np.ndarray,
    cfg: PradaLiteConfig,
    prefix_embeddings: Optional[torch.Tensor] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_features = None
    features = None
    if prefix_embeddings is not None and prefix_embeddings.ndim == 2 and prefix_embeddings.shape[0] == prompts.shape[0]:
        dense_features = prefix_embeddings.detach().float().cpu().numpy()
        norms = np.linalg.norm(dense_features, axis=-1, keepdims=True)
        dense_features = dense_features / np.maximum(norms, cfg.epsilon)
    else:
        features = _build_prefix_features(prompts, responses, attention_mask, cfg.hash_dim)
    batch_size = int(prompts.shape[0])
    p_hat = np.zeros(batch_size, dtype=np.float32)
    variance = np.zeros(batch_size, dtype=np.float32)
    effective_k = np.zeros(batch_size, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)

    for uid in np.unique(group_index):
        group_idx = np.where(group_index == uid)[0]
        for row in group_idx:
            candidates = [
                int(col)
                for col in group_idx
                if abs(int(traj_steps[col]) - int(traj_steps[row])) <= cfg.temporal_band
                and (not cfg.exclude_same_traj or traj_index[col] != traj_index[row])
            ]
            if len(candidates) == 0:
                candidates = [
                    int(col)
                    for col in group_idx
                    if abs(int(traj_steps[col]) - int(traj_steps[row])) <= cfg.temporal_band
                    and col != row
                ]
            scored: list[tuple[float, int]] = []
            for col in candidates:
                if dense_features is not None:
                    sim = float(np.dot(dense_features[row], dense_features[col]))
                else:
                    sim = _sparse_cosine(features[row], features[col])
                scored.append((sim, col))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: max(int(cfg.top_k), 1)]
            neighbor_indices = [col for _, col in top]
            sims = np.asarray([sim for sim, _ in top], dtype=np.float64)
            sims = np.clip(sims, -1.0, 1.0)
            weights = np.exp(sims / max(float(cfg.tau), cfg.epsilon))
            p_hat[row], variance[row] = _bootstrap_success_proxy(neighbor_indices, weights, success, cfg, rng)
            effective_k[row] = len(neighbor_indices)

    return p_hat, variance, effective_k


def _trace_smooth(evidence: np.ndarray, traj_index: np.ndarray, trace_decay: float) -> np.ndarray:
    smoothed = np.zeros_like(evidence, dtype=np.float32)
    for traj_uid in np.unique(traj_index):
        idx = np.where(traj_index == traj_uid)[0]
        running = 0.0
        for row in idx[::-1]:
            running = float(evidence[row]) + trace_decay * running
            smoothed[row] = running
    return smoothed


def _previous_by_trajectory(values: np.ndarray, traj_index: np.ndarray, default: float) -> np.ndarray:
    previous = np.full_like(values, fill_value=default, dtype=np.float32)
    for traj_uid in np.unique(traj_index):
        idx = np.where(traj_index == traj_uid)[0]
        if len(idx) > 1:
            previous[idx[1:]] = values[idx[:-1]]
    return previous


def _closed_form_projection(
    smoothed_evidence: np.ndarray,
    uncertainty_weight: np.ndarray,
    group_advantage: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    projected = np.zeros_like(smoothed_evidence, dtype=np.float32)
    for traj_uid in np.unique(traj_index):
        idx = np.where(traj_index == traj_uid)[0]
        if len(idx) == 0:
            continue
        evidence_sum = float(np.sum(smoothed_evidence[idx]))
        weight_sum = float(np.sum(uncertainty_weight[idx]))
        target_sum = float(len(idx) * group_advantage[idx[0]])
        correction = (target_sum - evidence_sum) / max(weight_sum, epsilon)
        projected[idx] = smoothed_evidence[idx] + uncertainty_weight[idx] * correction
    return projected


def _normalize_within_prompt_group(
    advantages: np.ndarray,
    group_index: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    normalized = advantages.copy()
    for uid in np.unique(group_index):
        idx = np.where(group_index == uid)[0]
        if len(idx) <= 1:
            continue
        mean = float(np.mean(normalized[idx]))
        std = float(np.std(normalized[idx]))
        normalized[idx] = (normalized[idx] - mean) / (std + epsilon)
    return normalized


def compute_prada_lite_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    prefix_embeddings: Optional[torch.Tensor] = None,
    episode_rewards: Optional[np.ndarray] = None,
    top_k: int = 16,
    bootstrap_rounds: int = 16,
    temporal_band: int = 2,
    tau: float = 0.2,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    lambda_u: float = 0.2,
    trace_gamma: float = 0.95,
    trace_lambda: float = 0.8,
    epsilon: float = 1e-6,
    hash_dim: int = 4096,
    seed: int = 0,
    normalize: bool = False,
    min_success_reward: Optional[float] = None,
    exclude_same_traj: bool = True,
    responsibility_mix: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute PRADA-lite step advantages.

    Equations implemented:
    - Group advantage: A_i^grp = (R_i - mean_G(R)) / (std_G(R) + eps)
    - Bootstrap success proxy: p_hat = (alpha0 + sum omega R) / (alpha0 + beta0 + sum omega)
    - Local evidence: e_t = logit(p_t) - logit(p_{t-1}) + lambda_u sign(A_grp) (v_{t-1} - v_t)
    - Trace smoothing: ebar_t = sum_u (trace_gamma trace_lambda)^(u-t) e_u
    - Projection: A_t = ebar_t + w_t * (T A_grp - sum_u ebar_u) / sum_u w_u
    - Conservative anchoring: final A_t = A_grp + mix * (A_t - A_grp)
    """
    if token_level_rewards.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("PRADA-lite expects token_level_rewards and response_mask to be rank-2 tensors.")
    if prompts.ndim != 2 or responses.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("PRADA-lite expects prompts, responses, and attention_mask to be rank-2 tensors.")

    cfg = PradaLiteConfig(
        top_k=top_k,
        bootstrap_rounds=bootstrap_rounds,
        temporal_band=temporal_band,
        tau=tau,
        alpha0=alpha0,
        beta0=beta0,
        lambda_u=lambda_u,
        trace_gamma=trace_gamma,
        trace_lambda=trace_lambda,
        epsilon=epsilon,
        hash_dim=hash_dim,
        seed=seed,
        normalize=normalize,
        min_success_reward=min_success_reward,
        exclude_same_traj=exclude_same_traj,
        responsibility_mix=responsibility_mix,
    )

    index = np.asarray(index, dtype=object)
    traj_index = np.asarray(traj_index, dtype=object)
    if len(index) != token_level_rewards.shape[0] or len(traj_index) != token_level_rewards.shape[0]:
        raise ValueError("PRADA-lite index arrays must match batch size.")

    # Use shaped token-level rewards for A^grp so existing reward transforms
    # such as invalid-action penalties remain effective. Use raw episode rewards
    # for the success proxy when available, because the proxy estimates terminal
    # task success rather than shaped optimization reward.
    outcome_rewards = _trajectory_outcome_rewards(token_level_rewards, traj_index)
    if episode_rewards is None:
        success_rewards = outcome_rewards
    else:
        success_rewards = _as_float_array(episode_rewards, "episode_rewards")

    success = _success_labels(success_rewards, cfg.min_success_reward)
    group_adv = _group_advantages(outcome_rewards, index, traj_index, cfg.epsilon)
    traj_steps = _trajectory_steps(traj_index)
    p_hat, variance, effective_k = _local_success_proxy(
        prompts=prompts,
        responses=responses,
        attention_mask=attention_mask,
        group_index=index,
        traj_index=traj_index,
        traj_steps=traj_steps,
        success=success,
        cfg=cfg,
        prefix_embeddings=prefix_embeddings,
    )

    p_clamped = np.clip(p_hat, cfg.epsilon, 1.0 - cfg.epsilon)
    logits = np.log(p_clamped / (1.0 - p_clamped)).astype(np.float32)
    prior_p = cfg.alpha0 / (cfg.alpha0 + cfg.beta0)
    prior_logit = float(np.log(prior_p / (1.0 - prior_p)))
    prev_logits = _previous_by_trajectory(logits, traj_index, prior_logit)
    prev_variance = _previous_by_trajectory(variance, traj_index, prior_p * (1.0 - prior_p))
    evidence = (logits - prev_logits) + cfg.lambda_u * np.sign(group_adv) * (prev_variance - variance)

    trace_decay = cfg.trace_gamma * cfg.trace_lambda
    smoothed = _trace_smooth(evidence, traj_index, trace_decay)
    uncertainty_weight = variance + cfg.epsilon
    projected = _closed_form_projection(smoothed, uncertainty_weight, group_adv, traj_index, cfg.epsilon)
    mix = float(np.clip(cfg.responsibility_mix, 0.0, 1.0))
    projected = group_adv + mix * (projected - group_adv)
    if cfg.normalize:
        projected = _normalize_within_prompt_group(projected, index, cfg.epsilon)

    advantages = torch.as_tensor(projected, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    advantages = advantages.unsqueeze(-1) * response_mask
    metrics = {
        "prada_lite/proxy_success_mean": float(np.mean(p_hat)),
        "prada_lite/proxy_success_std": float(np.std(p_hat)),
        "prada_lite/bootstrap_var_mean": float(np.mean(variance)),
        "prada_lite/effective_neighbors_mean": float(np.mean(effective_k)),
        "prada_lite/used_policy_hidden_embeddings": float(prefix_embeddings is not None),
        "prada_lite/evidence_mean": float(np.mean(evidence)),
        "prada_lite/responsibility_residual_std": float(np.std(projected - group_adv)),
        "prada_lite/responsibility_mix": mix,
        "prada_lite/projected_step_adv_mean": float(np.mean(projected)),
        "prada_lite/global_consistency_error": float(
            np.mean(
                [
                    abs(float(np.mean(projected[np.where(traj_index == traj_uid)[0]])) - float(group_adv[np.where(traj_index == traj_uid)[0][0]]))
                    for traj_uid in np.unique(traj_index)
                ]
            )
            if len(traj_index) > 0
            else 0.0
        ),
    }
    return advantages, advantages, metrics
