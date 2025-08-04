"""
Metrics measuring either uncertainty or confidence of a model.
"""
import torch
import torch.nn.functional as F


def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)


def confidence(logits):
    p = F.softmax(logits, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence


def entropy_prob(probs):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def mutual_information_prob(probs):
    mean_output = torch.mean(probs, dim=0)
    predictive_entropy = entropy_prob(mean_output)

    # Computing expectation of entropies
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

    # Computing mutual information
    mi = predictive_entropy - exp_entropies
    return mi

def self_consistency_o(maes):
    logits=maes[0]
    logits = torch.nn.functional.softmax(logits, dim=1)
    # logits=torch.round(logits * 100) / 100
    confidences, predictions = torch.max(logits, dim=1)

    mae=maes[1].squeeze()
    mae_up=maes[2].squeeze()
    mae_down=maes[3].squeeze()

    # mask = mae > 0.1
    # print(mae[mask])

    # uncertainty = torch.sqrt(abs(2 * mae_up * mae_down - mae * (mae_up + mae_down)))
    # uncertainty = (abs(2 * mae_up * mae_down - mae * (mae_up + mae_down)))

    target=2*logits*(1-logits)

    uncertainty1=abs(mae-target)

    uncertainty2= abs(mae_down + mae_up - target)

    uncertainty3 = abs(mae_down + mae_up - mae)

    uncertainty4=abs(mae_up-mae_down)

    uncertainty5=target

    uncertainty=uncertainty1 #+ uncertainty3#2 + uncertainty3 #- uncertainty4

    # num_classes = logits.size(1)
    # weights = torch.ones_like(uncertainty1)
    # weights[torch.arange(logits.size(0)), predictions] = num_classes-1
    # uncertainty = uncertainty * weights

    # batch_indices = torch.arange(uncertainty.size(0))  # shape: [B]
    # uncertainty = uncertainty[batch_indices, predictions]  # shape: [B]
    uncertainty = torch.sum(uncertainty, dim=1)
    # print(predictions)
    # print(uncertainty)
    # print(uncertainty5)
    # print(2*confidences*(1-confidences))
    # print(confidences)

    return uncertainty

def self_consistency_e(maes):
    logits=maes[0]
    logits = torch.nn.functional.softmax(logits, dim=1)
    confidences, predictions = torch.max(logits, dim=1)


    mae=maes[1]
    mae_up=maes[2]
    mae_down=maes[3]

    # uncertainty = torch.sqrt(abs(2 * mae_up * mae_down - mae * (mae_up + mae_down)))
    # uncertainty = (abs(2 * mae_up * mae_down - mae * (mae_up + mae_down)))

    target=2*logits*(1-logits)
    uncertainty1=abs(mae-target)

    # print(uncertainty1)
    # print(target)
    # print(mae)

    uncertainty2= abs(mae_down + mae_up - target)

    uncertainty3 = abs(mae_down + mae_up - mae)

    uncertainty4=abs(mae_up-mae_down)

    uncertainty5=target

    uncertainty=uncertainty1#+uncertainty3+uncertainty4


    num_classes = logits.size(1)
    weights = torch.ones_like(uncertainty1)
    weights[torch.arange(logits.size(0)), predictions] = num_classes-1
    uncertainty = uncertainty * weights

    # batch_indices = torch.arange(uncertainty.size(0))  # shape: [B]
    # uncertainty = uncertainty[batch_indices, predictions]  # shape: [B]
    uncertainty = torch.sum(uncertainty, dim=1)
    # print(predictions)
    # print(uncertainty)
    # print(2*confidences*(1-confidences))
    # print(confidences)

    return uncertainty

def edl_unc(logits):
    num_classes = logits.shape[1]
    uncertainty = num_classes / logits.sum(dim=1)
    return uncertainty

def certificate(OCs):
    return OCs