import torch
import torch.nn.functional as F
from torch import nn


class MCDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=True)


class AleatoricClassificationLoss(torch.nn.Module):
    def __init__(self, num_samples=100):
        super(AleatoricClassificationLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, logits, targets, log_std):
        return aleatoric_loss(logits, targets, log_std, num_samples=self.num_samples)


def aleatoric_loss(logits, targets, log_std, num_samples=100):
    # std = torch.exp(log_std)
    std = log_std
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples, device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise

    targets = targets.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples).squeeze(0)
    mc_x = torch.nn.functional.cross_entropy(prd, targets, reduction='none')
    # mean across mc samples
    mc_x = mc_x.mean(-1)
    # mean across every thing else
    mc_x_mean = mc_x.mean()
    # assert is not inf or nan
    assert not torch.isfinite(mc_x_mean).sum() == 0, f"Loss is inf: {mc_x_mean}"
    return mc_x.mean()


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


class AvgTrustedLoss(nn.Module):
    def __init__(self, num_views: int, annealing_start=50, gamma=1):
        super(AvgTrustedLoss, self).__init__()
        self.num_views = num_views
        self.annealing_step = 0
        self.annealing_start = annealing_start
        self.gamma = gamma

    def forward(self, evidences, target, evidence_a, **kwargs):
        num_classes = evidences.shape[-1]
        target = F.one_hot(target, num_classes)
        alphas = evidences + 1
        loss_acc = edl_digamma_loss(alphas, target, self.annealing_step, num_classes, self.annealing_start,
                                    evidence_a.device)
        for v in range(len(evidences)):
            alpha = evidences[v] + 1
            loss_acc += edl_digamma_loss(alpha, target, self.annealing_step, num_classes, self.annealing_start,
                                         evidence_a.device)
        loss_acc = loss_acc / (len(evidences) + 1)
        loss = loss_acc + self.gamma * get_dc_loss(evidences, evidence_a.device)
        return loss


def sampling_softmax(logits, log_sigma, num_samples=100):
    std = torch.exp(log_sigma)
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples, device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise
    return torch.softmax(prd, dim=0).mean(-1)


def compute_uncertainty(outputs, log_sigmas_ale, log_sigmas_ep, num_samples=100):
    p_ale = sampling_softmax(outputs, log_sigmas_ale, num_samples)
    entropy_ale = -torch.sum(p_ale * torch.log(p_ale + 1e-6), dim=-1)
    p_ep = sampling_softmax(outputs, log_sigmas_ep, num_samples)
    entropy_ep = -torch.sum(p_ep * torch.log(p_ep + 1e-6), dim=-1)
    return entropy_ale, entropy_ep


def mar_loss(outputs, labels, num_classes):
    # labels = F.one_hot(labels, num_classes=num_classes)
    labels = labels.to(torch.long)  # 转换为 LongTensor
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float() #* num_classes
    cls, mar, mar_up, mar_down = outputs
    criterion = nn.MSELoss()
    criterion_no_reduction = torch.nn.MSELoss(reduction='none')

    mean = F.softmax(cls, dim=1)  # *num_classes
    # confidences, predictions = torch.max(mean, dim=1)
    # preds_one_hot = torch.nn.functional.one_hot(predictions, num_classes=num_classes).float() * num_classes
    mar_targets = abs(labels_one_hot - mean.detach())
    loss_mar = criterion(mar, mar_targets)

    # loss_values_mar = criterion_no_reduction(mar, mar_targets)
    # weights = torch.ones_like(mar_targets)
    # # weights += targets*num_classes
    # # weights+=preds_one_hot
    # # print(weights)
    # loss_mar = (loss_values_mar * weights).mean()

    mask_up = (labels_one_hot == 1).float()
    mask_down = (labels_one_hot == 0).float()
    mae_up_targets = (labels_one_hot - mean).detach() * mask_up
    mae_down_targets = (mean).detach() * mask_down

    loss_mar_up = criterion(mar_up, mae_up_targets)
    loss_mar_down = criterion(mar_down, mae_down_targets)

    # loss_mar_up = criterion_no_reduction(mar_up, mae_up_targets)
    # # weights = torch.ones_like(mae_up_targets, device=device)
    # # weights += targets*num_classes
    # loss_mar_up = (loss_mar_up * weights).mean()
    #
    # loss_mar_down = criterion_no_reduction(mar_down, mae_down_targets)
    # # weights = torch.ones_like(mae_down_targets, device=device)#*(num_classes+1)
    # # weights -= targets*num_classes
    # # weights += targets*num_classes
    # loss_mar_down = (loss_mar_down * weights).mean()

    loss = loss_mar + loss_mar_up + loss_mar_down

    return loss


def mar_uncertainty(outputs):
    cls, mar, mar_up, mar_down = outputs
    logits = torch.nn.functional.softmax(cls, dim=1)
    confidences, predictions = torch.max(logits, dim=1)
    target=2*logits*(1-logits)
    uncertainty=abs(mar-target)

    uncertainty = torch.sum(uncertainty, dim=1)
    return uncertainty


def cross_entropy_loss(outputs, labels):
    loss = F.cross_entropy(outputs, labels)
    return loss

def softmax_entropy(logits):
    delta = torch.min(torch.min(logits[2] - logits[3], logits[1] - 2 * logits[3]), 2 * logits[2] - logits[1])
    c_uncertainty = abs(logits[2] + logits[3] - logits[1])
    threshold = 0.01
    # print(threshold)
    mask = (c_uncertainty < threshold).float()
    delta = delta * mask

    p = F.softmax(logits[0], dim=1)
    # logp = F.log_softmax(logits[0], dim=1)
    # plogp = p * logp

    calib_prob = p + delta
    calib_prob = torch.clamp(calib_prob, min=1e-8)
    log_prob = torch.log(calib_prob)
    plogp = calib_prob * log_prob

    entropy = -torch.sum(plogp, dim=1)
    return entropy