import torch
from torch import nn
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-10, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader, disable=True):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


# def gmm_forward(net, gaussians_model, data_B_X):
#
#     if isinstance(net, nn.DataParallel):
#         features_B_Z = net.module(data_B_X)
#         features_B_Z = net.module.feature
#     else:
#         features_B_Z = net(data_B_X)
#         features_B_Z = net.feature
#
#     log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])
#
#     return log_probs_B_Y

# def gmm_forward(net, gaussians_model, data_B_X, eps=1e-6):
#
#     net.eval()
#     with torch.no_grad():
#         if isinstance(net, nn.DataParallel):
#             _ = net.module(data_B_X)
#             features_B_Z = net.module.feature
#         else:
#             _ = net(data_B_X)
#             features_B_Z = net.feature
#
#         if torch.isnan(features_B_Z).any() or torch.isinf(features_B_Z).any():
#             raise ValueError("Features contain NaN or Inf")
#
#         means = gaussians_model.loc
#         covariances = gaussians_model.covariance_matrix
#
#         if torch.isnan(covariances).any() or torch.isinf(covariances).any():
#             raise ValueError("Covariances contain NaN or Inf")
#
#         stable_covariances = covariances + eps * torch.eye(covariances.shape[-1], device=covariances.device)
#
#         stable_gaussians_model = torch.distributions.MultivariateNormal(means, stable_covariances)
#
#         features_B_Z = features_B_Z.unsqueeze(1) if features_B_Z.dim() == 2 else features_B_Z
#         log_prob_B = stable_gaussians_model.log_prob(features_B_Z)
#
#     return log_prob_B

def gmm_forward(net, gaussians_model, data_B_X, eps=1e-6):
    net.eval()
    with torch.no_grad():
        if isinstance(net, nn.DataParallel):
            _ = net.module(data_B_X)
            features_B_Z = net.module.feature
        else:
            _ = net(data_B_X)
            features_B_Z = net.feature

        if torch.isnan(features_B_Z).any() or torch.isinf(features_B_Z).any():
            raise ValueError("Features contain NaN or Inf")

        device = features_B_Z.device

        log_probs = []
        for gmm in gaussians_model:
            mean = gmm["mean"].to(device)

            if "cov" in gmm:
                cov = gmm["cov"].to(device)
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    raise ValueError("Covariances contain NaN or Inf")
                cov_stable = cov + eps * torch.eye(cov.shape[-1], device=device)
                stable_gmm = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_stable)

            elif "scale_tril" in gmm:
                scale_tril = gmm["scale_tril"].to(device)
                stable_gmm = torch.distributions.MultivariateNormal(loc=mean, scale_tril=scale_tril)

            else:
                raise ValueError("GMM dict must contain either 'cov' or 'scale_tril'.")

            log_prob = stable_gmm.log_prob(features_B_Z)
            log_probs.append(log_prob)

        log_prob_B_C = torch.stack(log_probs, dim=1)
        return log_prob_B_C



def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader, disable=True):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


# def gmm_get_logits(gmm, embeddings):
#     log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
#     return log_probs_B_Y

def gmm_get_logits(gmms, embeddings):
    # gmms 是一个长度为 num_classes 的 list
    log_probs = []
    for gmm in gmms:
        log_prob = gmm.log_prob(embeddings)  # shape: (B,)
        log_probs.append(log_prob)
    log_probs_B_C = torch.stack(log_probs, dim=1)  # shape: (B, C)
    return log_probs_B_C

# def gmm_fit(embeddings, labels, num_classes):
#     with torch.no_grad():
#         classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
#         classwise_cov_features = torch.stack(
#             [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
#         )
#         del embeddings, labels
#         torch.cuda.empty_cache()
#
#         classwise_mean_features=classwise_mean_features.to('cuda')
#         classwise_cov_features=classwise_cov_features.to('cuda')
#
#         allocated = torch.cuda.memory_allocated()
#         reserved = torch.cuda.memory_reserved()
#         # 转为 MB 输出
#         print(f"Allocated: {allocated / 1024 ** 2:.2f} MB")
#         print(f"Reserved: {reserved / 1024 ** 2:.2f} MB")
#
#         torch.cuda.memory_summary()
#
#         # with torch.no_grad():
#     #     classwise_mean_features = torch.stack([
#     #         torch.mean(embeddings[labels == c], dim=0).cpu() for c in range(num_classes)
#     #     ])
#     #     classwise_cov_features = torch.stack([
#     #         centered_cov_torch(embeddings[labels == c].cpu() - classwise_mean_features[c]) for c in range(num_classes)
#     #     ])
#
#         for jitter_eps in JITTERS:
#             gmm=None
#             try:
#                 jitter = jitter_eps * torch.eye(
#                     classwise_cov_features.shape[1], device=classwise_cov_features.device,
#                 ).unsqueeze(0)
#
#                 gmm = torch.distributions.MultivariateNormal(
#                     loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
#                 )
#
#                 # jitter = jitter_eps * torch.eye(
#                 #     classwise_cov_features.shape[1]
#                 # ).unsqueeze(0)
#                 #
#                 # gmm = torch.distributions.MultivariateNormal(
#                 #     loc=classwise_mean_features.to('cuda'), covariance_matrix=(classwise_cov_features.to('cuda') + jitter.to('cuda')),
#                 # )
#
#             except RuntimeError as e:
#                 if "cholesky" in str(e):
#                     continue
#             except ValueError as e:
#                 if "The parameter covariance_matrix has invalid values" in str(e):
#                     continue
#             min_eigval = torch.min(torch.linalg.eigvalsh(classwise_cov_features + jitter))
#             print(f"jitter_eps={jitter_eps}, min_eigval: {min_eigval}")
#             if gmm is not None:
#                 print(gmm)
#                 break
#
#     return gmm, jitter_eps


# def gmm_fit(embeddings, labels, num_classes):
#     with torch.no_grad():
#         embeddings_cpu = embeddings.cpu()
#         labels_cpu = labels.cpu()
#
#         classwise_mean_features = []
#         classwise_cov_features = []
#
#         for c in range(num_classes):
#             mask = labels_cpu == c
#             emb_c = embeddings_cpu[mask]
#             mean_c = torch.mean(emb_c, dim=0)
#             cov_c = centered_cov_torch(emb_c - mean_c)
#             classwise_mean_features.append(mean_c)
#             classwise_cov_features.append(cov_c)
#
#             gpu_id = torch.cuda.current_device()
#             allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 2
#             reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 2
#             print(c)
#             print(f"Allocated memory2: {allocated:.2f} MB")
#             print(f"Reserved memory2: {reserved:.2f} MB")
#
#         del embeddings, labels
#         torch.cuda.empty_cache()
#
#         gpu_id = torch.cuda.current_device()
#         allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 2
#         reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 2
#         print(f"Allocated memory2: {allocated:.2f} MB")
#         print(f"Reserved memory2: {reserved:.2f} MB")
#
#         # 构建 GMM 时逐个进行
#         for jitter_eps in JITTERS:
#             print(f"Trying jitter_eps = {jitter_eps}")
#             successful = True
#             gmm_list = []
#
#             for c in range(num_classes):
#                 print(f"[Class {c}] Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
#
#                 try:
#                     mean = classwise_mean_features[c]
#                     cov = classwise_cov_features[c]
#                     jitter = jitter_eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
#                     adjusted_cov = cov + jitter
#
#                     min_eigval = torch.min(torch.linalg.eigvalsh(adjusted_cov))
#                     if min_eigval <= 0:
#                         raise ValueError(f"min eigval = {min_eigval.item():.2e}, not PD")
#
#                     gmm_c = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=adjusted_cov)
#                     gmm_list.append(gmm_c)
#
#                 except (RuntimeError, ValueError) as e:
#                     print(f"[Class {c}] Failed at jitter_eps={jitter_eps}: {str(e)}")
#                     successful = False
#                     break
#
#             if successful:
#                 print(f"✅ Successfully built GMMs with jitter_eps = {jitter_eps}")
#                 return gmm_list, jitter_eps
#
#         raise RuntimeError("❌ Failed to build valid GMMs with all provided jitter values.")
#
#     # 如果所有 jitter 都失败了，返回空
#     return None, None


import gc
def gmm_fit(embeddings, labels, num_classes, device, use_cholesky=False):
    """
    内存友好的 GMM 参数构建函数。

    Args:
        embeddings (Tensor): (N, D) embedding 表示
        labels (Tensor): (N,) 对应的类别标签
        num_classes (int): 类别数
        jitter_candidates (List[float]): 尝试的 jitter 值
        use_cholesky (bool): 是否使用 scale_tril 模式构造 GMM（推荐）

    Returns:
        gmm_params (List[Dict]): 每类的 {mean, cov} 或 {mean, scale_tril}
        best_jitter (float): 成功使用的 jitter 值
    """
    with torch.no_grad():
        embeddings_cpu = embeddings.cpu().float()
        labels_cpu = labels.cpu()

        # embeddings_cpu = embeddings.float()
        # labels_cpu = labels

        # classwise_mean_features = []
        # classwise_cov_features = []
        #
        # for c in range(num_classes):
        #     mask = labels_cpu == c
        #     emb_c = embeddings_cpu[mask]
        #     mean_c = torch.mean(emb_c, dim=0)
        #     cov_c = centered_cov_torch(emb_c - mean_c)
        #     classwise_mean_features.append(mean_c)
        #     classwise_cov_features.append(cov_c)

        del embeddings, labels
        torch.cuda.empty_cache()
        gc.collect()

        import psutil
        import os

        def get_cpu_memory_mb():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss / 1024 ** 2

        for jitter_eps in JITTERS:
            print(f"Trying jitter_eps = {jitter_eps}")
            successful = True
            gmm_params = []

            for c in range(num_classes):
                try:
                    # mean = classwise_mean_features[c]
                    # cov = classwise_cov_features[c]
                    # dim = cov.shape[0]

                    mask = labels_cpu == c
                    emb_c = embeddings_cpu[mask]
                    mean_c = torch.mean(emb_c, dim=0)
                    cov_c = centered_cov_torch(emb_c - mean_c)
                    dim = cov_c.shape[0]

                    jitter = jitter_eps * torch.eye(dim, dtype=cov_c.dtype)
                    adjusted_cov = cov_c + jitter

                    # 正定性检查
                    min_eigval = torch.min(torch.linalg.eigvalsh(adjusted_cov))
                    if min_eigval <= 0:
                        raise ValueError(f"min eigval = {min_eigval.item():.2e}, not PD")

                    if use_cholesky:
                        scale_tril = torch.linalg.cholesky(adjusted_cov)
                        gmm_params.append({'mean': mean_c, 'scale_tril': scale_tril})
                    else:
                        gmm_params.append({'mean': mean_c, 'cov': adjusted_cov})

                    # torch.cuda.synchronize()
                    # gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
                    # gpu_max = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    # cpu_mem = get_cpu_memory_mb()
                    # print(f"[Class {c}] GPU: {gpu_mem:.2f} MB (Max: {gpu_max:.2f} MB) | CPU: {cpu_mem:.2f} MB")

                    del mask, emb_c, mean_c, cov_c, jitter, adjusted_cov
                    gc.collect()


                except (RuntimeError, ValueError) as e:
                    print(f"[Class {c}] Failed at jitter_eps={jitter_eps}: {str(e)}")
                    successful = False
                    break

            if successful:
                print(f"Successfully built GMM parameters with jitter_eps = {jitter_eps}")
                return gmm_params, jitter_eps

        raise RuntimeError("Failed to build valid GMMs with all provided jitter values.")
