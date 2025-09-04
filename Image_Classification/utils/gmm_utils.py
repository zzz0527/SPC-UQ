import gc
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = [
    "DOUBLE_INFO",
    "JITTERS",
    "centered_cov_torch",
    "get_embeddings",
    "gmm_forward",
    "gmm_evaluate",
    "gmm_get_logits",
    "gmm_fit",
]

# Numeric limits and jitter candidates for stabilizing covariance matrices
DOUBLE_INFO = torch.finfo(torch.double)
JITTERS: List[float] = [0.0, float(DOUBLE_INFO.tiny)] + [10.0 ** exp for exp in range(-10, 0)]


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------
def centered_cov_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute sample covariance for a centered matrix X (N, D).

    Args:
        x: Centered data matrix of shape (N, D), i.e., mean(x, dim=0) ≈ 0.

    Returns:
        Covariance matrix of shape (D, D).
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    return (x.T @ x) / (n - 1)


def _get_features(net: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Forward data through `net` and return feature tensor.

    Supports models that:
      - expose the last-forwarded features via a `.feature` attribute, OR
      - directly return features from `forward`.

    Handles `nn.DataParallel` transparently.

    Args:
        net: PyTorch module (possibly wrapped in DataParallel).
        inputs: Input batch tensor.

    Returns:
        Feature tensor of shape (B, D).
    """
    module = net.module if isinstance(net, nn.DataParallel) else net
    out = module(inputs)  # refresh any internal feature buffer

    feats = getattr(module, "feature", out)
    if not isinstance(feats, torch.Tensor):
        raise TypeError("Model features are not a torch.Tensor.")

    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    return feats


# ---------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------
@torch.inference_mode()
def get_embeddings(
    net: nn.Module,
    loader: DataLoader,
    num_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    storage_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embeddings and labels for an entire dataset.

    Args:
        net: Trained network providing features (via return value or `.feature`).
        loader: DataLoader providing (inputs, labels).
        num_dim: Embedding dimensionality.
        dtype: Desired dtype for saved embeddings.
        device: Compute device for forward passes.
        storage_device: Device to store the resulting tensors (e.g., 'cpu').

    Returns:
        embeddings: Tensor of shape (N, D).
        labels: Tensor of shape (N,) with dtype torch.long.
    """
    net.eval()

    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.long, device=storage_device)

    start = 0
    for data, label in tqdm(loader, disable=True):
        data = data.to(device, non_blocking=True)
        feats = _get_features(net, data)

        end = start + data.size(0)
        embeddings[start:end].copy_(feats.to(storage_device), non_blocking=True)
        labels[start:end].copy_(label.to(storage_device), non_blocking=True)
        start = end

    return embeddings, labels


# ---------------------------------------------------------------------
# GMM scoring (forward/evaluate)
# ---------------------------------------------------------------------
DistributionOrParams = Union[
    torch.distributions.MultivariateNormal,
    Dict[str, torch.Tensor],  # {'mean', 'cov'} or {'mean', 'scale_tril'}
]


@torch.inference_mode()
def gmm_forward(
    net: nn.Module,
    gaussians_model: Sequence[DistributionOrParams],
    data_B_X: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute class-wise log-probabilities for a batch using per-class Gaussians.

    Accepts either:
      - a sequence of torch.distributions.MultivariateNormal (one per class), OR
      - a sequence of dicts with keys {'mean', 'cov'} or {'mean', 'scale_tril'}.

    Args:
        net: Feature-extracting model.
        gaussians_model: Per-class Gaussian distributions or parameter dicts.
        data_B_X: Input batch (B, ...).
        eps: Diagonal jitter for covariance stabilization when dicts have 'cov'.

    Returns:
        log_prob_B_C: Log-prob tensor of shape (B, C).
    """
    net.eval()
    feats = _get_features(net, data_B_X)

    if torch.isnan(feats).any() or torch.isinf(feats).any():
        raise ValueError("Features contain NaN or Inf.")

    device = feats.device
    log_probs: List[torch.Tensor] = []

    for g in gaussians_model:
        if isinstance(g, torch.distributions.MultivariateNormal):
            dist = g
            if dist.loc.device != device:
                # Rebuild on the correct device
                dist = torch.distributions.MultivariateNormal(
                    loc=dist.loc.to(device),
                    covariance_matrix=getattr(dist, "covariance_matrix", None),
                    scale_tril=getattr(dist, "scale_tril", None),
                )
        else:
            mean = g["mean"].to(device)
            if "cov" in g:
                cov = g["cov"].to(device)
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    raise ValueError("Covariances contain NaN or Inf.")
                cov_stable = cov + eps * torch.eye(cov.shape[-1], device=device, dtype=cov.dtype)
                dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_stable)
            elif "scale_tril" in g:
                scale_tril = g["scale_tril"].to(device)
                dist = torch.distributions.MultivariateNormal(loc=mean, scale_tril=scale_tril)
            else:
                raise ValueError("Each GMM spec must contain 'cov' or 'scale_tril'.")

        log_probs.append(dist.log_prob(feats))

    log_prob_B_C = torch.stack(log_probs, dim=1)  # (B, C)
    return log_prob_B_C


@torch.inference_mode()
def gmm_evaluate(
    net: nn.Module,
    gaussians_model: Sequence[DistributionOrParams],
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    storage_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate GMM log-probabilities across a dataset.

    Args:
        net: Feature-extracting model.
        gaussians_model: Per-class Gaussian distributions or parameter dicts.
        loader: DataLoader with (inputs, labels).
        device: Compute device for forward passes.
        num_classes: Number of classes (C).
        storage_device: Device to store logits/labels (e.g., 'cpu').

    Returns:
        logits_N_C: Log-probabilities (N, C).
        labels_N: Labels (N,).
    """
    net.eval()

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float32, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.long, device=storage_device)

    start = 0
    for data, label in tqdm(loader, disable=True):
        data = data.to(device, non_blocking=True)
        logit_B_C = gmm_forward(net, gaussians_model, data)

        end = start + data.size(0)
        logits_N_C[start:end].copy_(logit_B_C.to(storage_device), non_blocking=True)
        labels_N[start:end].copy_(label.to(storage_device), non_blocking=True)
        start = end

    return logits_N_C, labels_N


def gmm_get_logits(
    gmms: Sequence[torch.distributions.MultivariateNormal],
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute class-wise log-probabilities given embeddings and per-class Gaussians.

    Args:
        gmms: Sequence of MultivariateNormal, one per class.
        embeddings: Embedding tensor (B, D).

    Returns:
        log_probs_B_C: Tensor (B, C) with log-probabilities.
    """
    log_probs = [gmm.log_prob(embeddings) for gmm in gmms]
    return torch.stack(log_probs, dim=1)


# ---------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------
@torch.inference_mode()
def gmm_fit(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    use_cholesky: bool = False,
) -> Tuple[List[Dict[str, torch.Tensor]], float]:
    """
    Memory-friendly construction of per-class Gaussian parameters.

    For each class c:
      - mean_c = mean(embeddings_c)
      - cov_c  = sample covariance of centered embeddings_c
      - add jitter to ensure positive definiteness
      - return either {'mean', 'cov'} or {'mean', 'scale_tril'}

    All computations are done on CPU for memory safety; outputs are returned on CPU.
    Move to GPU only when scoring.

    Args:
        embeddings: (N, D) embeddings in any device; will be moved to CPU float32.
        labels: (N,) integer labels; will be moved to CPU long.
        num_classes: Number of classes C.
        device: (Kept for API symmetry; not used in fitting).
        use_cholesky: If True, return `scale_tril` instead of `cov`.

    Returns:
        gmm_params: List of dicts per class with {'mean', 'cov'} or {'mean', 'scale_tril'}.
        best_jitter: The jitter value that produced valid PD covariances for all classes.
    """
    # Move to CPU and free GPU memory if these came from GPU
    embeddings_cpu = embeddings.detach().to("cpu", dtype=torch.float32)
    labels_cpu = labels.detach().to("cpu", dtype=torch.long)

    # Proactively release references to source tensors to help GC
    del embeddings, labels
    torch.cuda.empty_cache()
    gc.collect()

    # Try a sequence of jitter values until all class covariances are PD
    for jitter_eps in JITTERS:
        successful = True
        gmm_params: List[Dict[str, torch.Tensor]] = []

        for c in range(num_classes):
            mask = labels_cpu == c
            emb_c = embeddings_cpu[mask]
            if emb_c.numel() == 0:
                successful = False
                break

            mean_c = emb_c.mean(dim=0)
            cov_c = centered_cov_torch(emb_c - mean_c)

            # Stabilize covariance
            d = cov_c.shape[0]
            jitter = jitter_eps * torch.eye(d, dtype=cov_c.dtype)
            cov_adj = cov_c + jitter

            # Positive definiteness check
            try:
                min_eig = torch.linalg.eigvalsh(cov_adj).min()
                if torch.isnan(min_eig) or (min_eig <= 0):
                    raise ValueError("Covariance not PD after jitter.")
            except Exception:
                successful = False
                break

            if use_cholesky:
                try:
                    scale_tril = torch.linalg.cholesky(cov_adj)
                except RuntimeError:
                    successful = False
                    break
                gmm_params.append({"mean": mean_c, "scale_tril": scale_tril})
            else:
                gmm_params.append({"mean": mean_c, "cov": cov_adj})

            # Cleanup per-class temporaries aggressively
            del mask, emb_c, mean_c, cov_c, jitter, cov_adj

        if successful and len(gmm_params) == num_classes:
            return gmm_params, float(jitter_eps)

        # If failed, try the next jitter value
        del gmm_params
        gc.collect()

    raise RuntimeError("Failed to build valid GMMs with all provided jitter values.")
