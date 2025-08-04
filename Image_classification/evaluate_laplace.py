"""
Script to evaluate the Laplace Approximation.
"""

import os
import json
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import data loaders and networks
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.imagenet as imagenet
import data.ood_detection.tinyimagenet as tinyimagenet
import data.ood_detection.imagenet_o as imagenet_o
import data.ood_detection.imagenet_a as imagenet_a
import data.ood_detection.ood_union as ood_union

from net.resnet import resnet50
from net.resnet_edl import resnet50_edl
from net.wide_resnet import wrn
from net.wide_resnet_edl import wrn_edl
from net.vgg import vgg16
from net.vgg_edl import vgg16_edl
from net.inception import inception_v3
from net.imagenet_wide import imagenet_wide
from net.imagenet_vgg import imagenet_vgg16
from net.imagenet_vit import imagenet_vit

from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_logits,
    test_classification_net_ensemble,
    test_classification_net_edl,
    create_adversarial_dataloader
)
from metrics.calibration_metrics import expected_calibration_error

from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import laplace_eval_args

from laplace import Laplace
from sklearn import metrics as M
from laplace.curvature import AsdlGGN, AsdlEF, BackPackGGN, BackPackEF

import warnings
warnings.filterwarnings('ignore')

# Dataset mapping and config
DATASET_NUM_CLASSES = {
    "cifar10": 10, "cifar100": 100, "svhn": 10, "imagenet": 1000,
    "tinyimagenet": 200, "imagenet_o": 200, "imagenet_a": 200
}

DATASET_LOADER = {
    "cifar10": cifar10, "cifar100": cifar100, "svhn": svhn, "imagenet": imagenet,
    "tinyimagenet": tinyimagenet, "imagenet_o": imagenet_o, "imagenet_a": imagenet_a
}

MODELS = {
    "resnet50": resnet50, "resnet50_edl": resnet50_edl,
    "wide_resnet": wrn, "wide_resnet_edl": wrn_edl,
    "vgg16": vgg16, "vgg16_edl": vgg16_edl, "inception": inception_v3,
    "imagenet_wide": imagenet_wide
}

MODEL_TO_NUM_DIM = {
    "resnet50": 2048, "resnet50_edl": 2048, "wide_resnet": 640, "wide_resnet_edl": 640,
    "vgg16": 512, "vgg16_edl": 512, "inception": 2048, "imagenet_wide": 2048,
    "imagenet_vgg16": 4096, "imagenet_vit": 768
}

MODEL_TO_INPUT_DIM = {
    "resnet50": 32, "resnet50_edl": 32, "wide_resnet": 32, "wide_resnet_edl": 32,
    "vgg16": 32, "inception": 128, "vgg16_edl": 32, "imagenet_wide": 224,
    "imagenet_vgg16": 224, "imagenet_vit": 224
}

MODEL_TO_LAST_LAYER = {
    "resnet50": "module.fc", "wide_resnet": "module.linear", "vgg16": "module.classifier",
    "inception": "module.fc", "imagenet_wide": "module.linear",
    "imagenet_vgg16": "module.classifier", "imagenet_vit": "module.linear"
}

def get_backend(backend, approx_type):
    if backend == 'kazuki':
        return AsdlGGN if approx_type == 'ggn' else AsdlEF
    elif backend == 'backpack':
        return BackPackGGN if approx_type == 'ggn' else BackPackEF
    else:
        raise ValueError(f"Unknown backend: {backend}")

def get_cpu_memory_mb():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2

def print_metrics(mean, std, name):
    print(f"{name}: {mean * 100:.2f} ± {std * 100:.2f}")

if __name__ == "__main__":

    args = laplace_eval_args().parse_args()

    # Set random seed and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Parsed args:", args)
    print("Seed:", args.seed)

    num_classes = DATASET_NUM_CLASSES[args.dataset]
    test_loader = DATASET_LOADER[args.dataset].get_test_loader(
        batch_size=args.batch_size, imagesize=MODEL_TO_INPUT_DIM[args.model], pin_memory=args.gpu
    )
    if args.ood_dataset == 'ood_union':
        ood_test_loader = ood_union.get_combined_ood_test_loader(
            batch_size=args.batch_size, sample_seed=args.seed,
            imagesize=MODEL_TO_INPUT_DIM[args.model], pin_memory=args.gpu
        )
    else:
        ood_test_loader = DATASET_LOADER[args.ood_dataset].get_test_loader(
            batch_size=args.batch_size, imagesize=MODEL_TO_INPUT_DIM[args.model], pin_memory=args.gpu
        )

    # Prepare metric accumulators
    accuracies, eces, ood_aurocs, err_aurocs, adv_aurocs = [], [], [], [], []
    err_aurocs, adv_aurocs = [], []
    adv_unc = np.zeros((args.runs, 9))
    adv_acc = np.zeros((args.runs, 9))
    adv_ep = 0.02

    for i in range(args.runs):
        # Load training/validation splits
        train_loader, val_loader = DATASET_LOADER[args.dataset].get_train_valid_loader(
            batch_size=args.batch_size, imagesize=MODEL_TO_INPUT_DIM[args.model], augment=args.data_aug,
            val_seed=(args.seed + i), val_size=args.val_size, pin_memory=args.gpu
        )
        mixture_components = []
        for model_idx in range(args.nr_components):
            if args.dataset == 'imagenet':
                net = MODELS[args.model](pretrained=True, num_classes=1000).cuda()
                net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                cudnn.benchmark = True
            else:
                if args.val_size == 0.1 or not args.crossval:
                    saved_model_name = os.path.join(
                        args.load_loc, f"Run{i+1}",
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350.model",
                    )
                else:
                    saved_model_name = os.path.join(
                        args.load_loc, f"Run{i+1}",
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i)
                        + f"_350_0{int(args.val_size * 10)}.model"
                    )
                print('Loading:', saved_model_name)
                net = MODELS[args.model](
                    spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, temp=1.0
                )
                if args.gpu:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                    cudnn.benchmark = True
                net.load_state_dict(torch.load(str(saved_model_name)))

            # Laplace backend and fit
            args.prior_precision = 1.0 if isinstance(args.prior_precision, float) else torch.load(args.prior_precision, map_location=device)
            Backend = get_backend(args.backend, args.approx_type)
            args.last_layer_name = MODEL_TO_LAST_LAYER[args.model]
            optional_args = {"last_layer_name": args.last_layer_name} if args.subset_of_weights == 'last_layer' else {}

            print('Fitting Laplace approximation...')
            model = Laplace(
                net, args.likelihood, subset_of_weights=args.subset_of_weights,
                hessian_structure=args.hessian_structure, prior_precision=args.prior_precision,
                temperature=args.temperature, backend=Backend, **optional_args
            )
            model.fit(val_loader if args.crossval else train_loader)

            # Optional: Optimize prior precision
            if (args.optimize_prior_precision is not None) and (args.method == 'laplace'):
                n = model.n_params if args.prior_structure == 'all' else model.n_layers
                prior_precision = args.prior_precision * torch.ones(n, device=device)
                print('Optimizing prior precision...')
                model.optimize_prior_precision(
                    method=args.optimize_prior_precision, init_prior_prec=prior_precision,
                    val_loader=val_loader, pred_type=args.pred_type, link_approx=args.link_approx,
                    n_samples=args.n_samples, verbose=(args.prior_structure == 'scalar')
                )
            mixture_components.append(model)

        model = mixture_components[0]
        loss_fn = nn.NLLLoss()

        # Evaluate ID data
        id_y_true, id_y_prob = [], []
        for data in tqdm(test_loader, desc='Evaluating ID data'):
            x, y = data[0].to(device), data[1].to(device)
            id_y_true.append(y.cpu())
            y_prob = model(x, pred_type=args.pred_type, link_approx=args.link_approx, n_samples=args.n_samples)
            id_y_prob.append(y_prob.cpu())
        id_y_prob = torch.cat(id_y_prob, dim=0)
        id_y_true = torch.cat(id_y_true, dim=0)
        c, preds = torch.max(id_y_prob, 1)
        metrics = {}
        metrics['conf'] = c.mean().item()
        metrics['nll'] = loss_fn(id_y_prob.log(), id_y_true).item()
        metrics['acc'] = (id_y_true == preds).float().mean().item()
        accuracy = metrics['acc']
        id_confidences = id_y_prob.max(dim=1)[0].numpy()
        ece = expected_calibration_error(id_confidences, preds.numpy(), id_y_true.numpy(), num_bins=15)
        t_ece = ece
        metrics['ece'] = ece
        print(metrics)

        # Evaluate OOD data
        ood_y_true, ood_y_prob = [], []
        for data in tqdm(ood_test_loader, desc='Evaluating OOD data'):
            x, y = data[0].to(device), data[1].to(device)
            ood_y_true.append(y.cpu())
            y_prob = model(x, pred_type=args.pred_type, link_approx=args.link_approx, n_samples=args.n_samples)
            ood_y_prob.append(y_prob.cpu())
        ood_y_prob = torch.cat(ood_y_prob, dim=0)
        ood_y_true = torch.cat(ood_y_true, dim=0)
        ood_confidences = ood_y_prob.max(dim=1)[0].numpy()

        # OOD AUROC/AUPRC metrics
        bin_labels = np.concatenate([
            np.zeros(id_confidences.shape[0]),
            np.ones(ood_confidences.shape[0])
        ])
        scores = np.concatenate([id_confidences, ood_confidences])
        fpr, tpr, thresholds = M.roc_curve(bin_labels, scores)
        precision, recall, prc_thresholds = M.precision_recall_curve(bin_labels, scores)
        ood_auroc = M.roc_auc_score(bin_labels, scores)
        auprc = M.average_precision_score(bin_labels, scores)
        print(f"OOD AUROC: {ood_auroc:.4f}, AUPRC: {auprc:.4f}")

        # Error AUROC/AUPRC (in-distribution: correct vs incorrect)
        labels_array = np.array(id_y_true)
        pred_array = np.array(preds)
        correct_mask = labels_array == pred_array
        confidences_right = id_confidences[correct_mask]
        confidences_wrong = id_confidences[~correct_mask]
        bin_labels = np.concatenate([
            np.zeros(confidences_right.shape[0]),
            np.ones(confidences_wrong.shape[0])
        ])
        scores = np.concatenate([confidences_right, confidences_wrong])
        err_auroc = M.roc_auc_score(bin_labels, scores)
        err_auprc = M.average_precision_score(bin_labels, scores)
        print(f"Error AUROC: {err_auroc:.4f}, AUPRC: {err_auprc:.4f}")

        # Adversarial robustness
        adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep, batch_size=args.batch_size)
        adv_y_prob, adv_y_true = [], []
        for data in tqdm(adv_loader, desc='Adversarial evaluation'):
            x, y = data[0].to(device), data[1].to(device)
            y_prob = model(x, pred_type=args.pred_type, link_approx=args.link_approx, n_samples=args.n_samples)
            adv_y_true.append(y.cpu())
            adv_y_prob.append(y_prob.cpu())
        adv_y_prob = torch.cat(adv_y_prob, dim=0)
        adv_y_true = torch.cat(adv_y_true, dim=0).numpy()
        _, adv_predictions = torch.max(adv_y_prob, 1)
        adv_accuracy = (adv_y_true == adv_predictions).mean()
        adv_confidences = adv_y_prob.max(dim=1)[0].numpy()
        bin_labels = np.concatenate([
            np.zeros(id_confidences.shape[0]),
            np.ones(adv_confidences.shape[0])
        ])
        adv_scores = np.concatenate([id_confidences, adv_confidences])
        adv_auroc = M.roc_auc_score(bin_labels, adv_scores)
        adv_auprc = M.average_precision_score(bin_labels, adv_scores)
        print(f"Adversarial AUROC: {adv_auroc:.4f}, AUPRC: {adv_auprc:.4f}")

        # If sample_noise: save/plot noise-uncertainty and accuracy curves
        if args.sample_noise:
            adv_eps = np.linspace(0, 0.4, 9)
            for idx_ep, ep in enumerate(adv_eps):
                adv_loader = create_adversarial_dataloader(
                    net, test_loader, device, epsilon=ep, batch_size=args.batch_size
                )
                adv_y_prob, adv_y_true = [], []
                for data in tqdm(adv_loader, desc=f"Adv evaluation ep={ep:.2f}"):
                    x, y = data[0].to(device), data[1].to(device)
                    y_prob = model(x, pred_type=args.pred_type, link_approx=args.link_approx, n_samples=args.n_samples)
                    adv_y_true.append(y.cpu())
                    adv_y_prob.append(y_prob.cpu())
                adv_y_prob = torch.cat(adv_y_prob, dim=0)
                adv_y_true = torch.cat(adv_y_true, dim=0).numpy()
                _, predictions = torch.max(adv_y_prob, 1)
                adv_accuracy = (adv_y_true == predictions).mean()
                uncertainties = 1 - adv_y_prob.max(dim=1)[0].numpy()
                adv_unc[i][idx_ep] = uncertainties.mean()
                adv_acc[i][idx_ep] = adv_accuracy
            # Save/plot uncertainty/accuracy curves as in your original

        # Accumulate results
        accuracies.append(accuracy)
        eces.append(ece)
        ood_aurocs.append(ood_auroc)
        err_aurocs.append(err_auroc)
        adv_aurocs.append(adv_auroc)

        del model, mixture_components
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Final result reporting and saving

    def mean_std(x):
        arr = torch.tensor(x)
        return arr.mean().item(), arr.std().item() / math.sqrt(arr.shape[0])

    # Print summary
    print_metrics(*mean_std(accuracies), "Accuracy")
    print_metrics(*mean_std(eces), "ECE")
    print_metrics(*mean_std(adv_aurocs), "Adv AUROC")
    print_metrics(*mean_std(err_aurocs), "Error AUROC")
    print_metrics(*mean_std(ood_aurocs), "OOD AUROC")

    # Store only required metrics
    result_json = {}
    for key, arr in [
        ("accuracy", accuracies),
        ("ece", eces),
        ("adv_auroc", adv_aurocs),
        ("err_auroc", err_aurocs),
        ("ood_auroc", ood_aurocs)
    ]:
        mean, std = mean_std(arr)
        result_json[key] = {
            "mean": mean,
            "std": std,
            "values": [float(v) for v in arr]
        }

    result_file = (
        "res_" + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
        + "_laplace_" + args.dataset + "_" + args.ood_dataset + ".json"
    )
    with open(result_file, "w") as f:
        json.dump(result_json, f, indent=2)