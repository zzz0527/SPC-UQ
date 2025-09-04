"""
Script to evaluate a single model. 
"""
import os
import gc
import json
import math
import torch
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.imagenet as imagenet
import data.ood_detection.tinyimagenet as tinyimagenet
import data.ood_detection.imagenet_o as imagenet_o
import data.ood_detection.imagenet_a as imagenet_a
import data.ood_detection.ood_union as ood_union

# Import network models
from net.resnet import resnet50
from net.resnet_edl import resnet50_edl
from net.wide_resnet import wrn
from net.wide_resnet_edl import wrn_edl
from net.wide_resnet_uq import wrn_uq
from net.vgg import vgg16
from net.vgg_edl import vgg16_edl
from net.vgg_uq import vgg16_uq
from net.imagenet_wide import imagenet_wide
from net.imagenet_vgg import imagenet_vgg16
from net.imagenet_vit import imagenet_vit

# Import metrics to compute
from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_logits,
    test_classification_uq,
    test_classification_net_ensemble,
    test_classification_net_edl,
    create_adversarial_dataloader,
    test_classification_net_logits_edl,
    test_classification_net_softmax
)
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp, self_consistency, edl_unc, certificate
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble, get_unc_ensemble, get_roc_auc_uncs
from metrics.classification_metrics import get_logits_labels
from metrics.classification_metrics import get_logits_labels_uq

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.ensemble_utils import load_ensemble, Ensemble_fit, Ensemble_evaluate, Ensemble_load
from utils.oc_utils import oc_fit, oc_evaluate
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import eval_args

# Import SPC utils
from utils.spc_utils import SPC_fit, SPC_load, SPC_evaluate

# Import EDL utils
from utils.edl_utils import EDL_fit, EDL_load, EDL_evaluate

# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "imagenet": 1000, "tinyimagenet": 200, "imagenet_o":200, "imagenet_a":200}

dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn, "imagenet": imagenet, "tinyimagenet": tinyimagenet, "imagenet_o":imagenet_o, "imagenet_a":imagenet_a}

# Mapping model name to model function
models = {"resnet50": resnet50, "resnet50_edl":resnet50_edl, "wide_resnet": wrn, "wide_resnet_edl": wrn_edl, "wide_resnet_uq": wrn_uq, "vgg16": vgg16, "vgg16_edl": vgg16_edl, "vgg16_uq": vgg16_uq, "imagenet_wide":imagenet_wide, "imagenet_vgg16":imagenet_vgg16, "imagenet_vit":imagenet_vit}

model_to_num_dim = {"resnet50": 2048, "resnet50_edl":2048, "wide_resnet": 640, "wide_resnet_edl": 640, "wide_resnet_uq": 640, "vgg16": 512, "vgg16_edl": 512, "vgg16_uq": 512, "imagenet_wide":2048, "imagenet_vgg16":4096, "imagenet_vit":768}

model_to_input_dim = {"resnet50": 32, "resnet50_edl": 32, "wide_resnet": 32, "wide_resnet_edl": 32, "wide_resnet_uq": 32, "vgg16": 32, "vgg16_edl": 32, "vgg16_uq": 32, "imagenet_wide":224, "imagenet_vgg16":224, "imagenet_vit":224}

model_to_last_layer = {"resnet50": "module.fc", "wide_resnet": "module.linear", "vgg16": "module.classifier", "imagenet_wide": "module.linear", "imagenet_vgg16": "module.classifier", "imagenet_vit": "module.linear"}

if __name__ == "__main__":

    args = eval_args().parse_args()

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, imagesize=model_to_input_dim[args.model], pin_memory=args.gpu)

    if args.ood_dataset=='ood_union':
        ood_test_loader = ood_union.get_combined_ood_test_loader(batch_size=args.batch_size, sample_seed=args.seed, imagesize=model_to_input_dim[args.model], pin_memory=args.gpu)
    else:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size,
                                                                           imagesize=model_to_input_dim[args.model],
                                                                           pin_memory=args.gpu)

    # Evaluating the models
    accuracies = []
    c_accuracies = []

    # Pre temperature scaling
    # m1 - Uncertainty/Confidence Metric 1
    #      for deterministic model: logsumexp, for ensemble: entropy
    # m2 - Uncertainty/Confidence Metric 2
    #      for deterministic model: entropy, for ensemble: MI
    eces = []
    ood_m1_aurocs = []
    ood_m1_auprcs = []
    ood_m2_aurocs = []
    ood_m2_auprcs = []

    err_m1_aurocs = []
    err_m1_auprcs = []
    err_m2_aurocs = []
    err_m2_auprcs = []

    adv_ep = 0.02
    adv_m1_aurocs = []
    adv_m1_auprcs = []
    adv_m2_aurocs = []
    adv_m2_auprcs = []

    # Post temperature scaling
    t_eces = []
    t_m1_aurocs = []
    t_m1_auprcs = []
    t_m2_aurocs = []
    t_m2_auprcs = []

    c_eces = []

    adv_unc = np.zeros((args.runs, 9))
    adv_acc = np.zeros((args.runs, 9))

    topt = None

    for i in range(args.runs):
        print (f"Evaluating run: {(i+1)}")
        # Loading the model(s)
        if args.model_type == "ensemble":
            if args.dataset == 'imagenet':
                train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                    batch_size=args.batch_size, imagesize=model_to_input_dim[args.model], augment=args.data_aug,
                    val_seed=(args.seed + i), val_size=args.val_size, pin_memory=args.gpu)
                net = models[args.model](pretrained=True, num_classes=1000).cuda()
                if args.gpu:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                    cudnn.benchmark = True
                net.eval()

            else:
                val_loaders = []
                for j in range(args.ensemble):
                    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                        batch_size=args.batch_size, imagesize=model_to_input_dim[args.model], augment=args.data_aug, val_seed=(args.seed+(5*i)+j), val_size=0.1, pin_memory=args.gpu,
                    )
                    val_loaders.append(val_loader)
                # Evaluate an ensemble
                ensemble_loc = args.load_loc
                net_ensemble = load_ensemble(
                    ensemble_loc=ensemble_loc,
                    model_name=args.model,
                    device=device,
                    num_classes=num_classes,
                    spectral_normalization=args.sn,
                    mod=args.mod,
                    coeff=args.coeff,
                    seed=(i)
                )

        else:
            train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                batch_size=args.batch_size, imagesize=model_to_input_dim[args.model], augment=args.data_aug, val_seed=(args.seed+i), val_size=args.val_size, pin_memory=args.gpu,
            )
            if args.dataset == 'imagenet':
                net = models[args.model](pretrained=True, num_classes=1000).cuda()
                if args.gpu:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                    cudnn.benchmark = True
                net.eval()

            else:
                if args.val_size==0.1 or (not args.crossval):
                    saved_model_name = os.path.join(
                        args.load_loc,
                        "Run" + str(i + 1),
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350.model",
                    )
                else:
                    saved_model_name = os.path.join(
                        args.load_loc,
                        "Run" + str(i + 1),
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350_0"+str(int(args.val_size*10))+".model",
                    )
                print(saved_model_name)
                net = models[args.model](
                    spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, temp=1.0,
                )
                if args.gpu:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                    cudnn.benchmark = True
                net.load_state_dict(torch.load(str(saved_model_name)))
                net.eval()


        # Evaluating the UQ method
        if args.model_type == "ensemble":
            if args.dataset == 'imagenet':
                ensemble_model_path = os.path.join(
                    args.load_loc,
                    "Run" + str(i + 1),
                    model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350_ensemble_model.pth",
                )

                if os.path.exists(ensemble_model_path):
                    print(f"Loading existing ensemble_model from {ensemble_model_path}")
                    Ensemble_model = Ensemble_load(ensemble_model_path, model_to_num_dim[args.model],
                                                         num_classes, device)
                else:
                    if args.model == 'imagenet_vgg16':
                        embed_path = 'data/imagenet_train_vgg_embedding.pt'
                        # embed_path = 'data/imagenet_val_vgg_embedding.pt'
                    if args.model == 'imagenet_wide':
                        embed_path = 'data/imagenet_train_wide_embedding.pt'
                        # embed_path = 'data/imagenet_val_wide_embedding.pt'
                    if args.model == 'imagenet_vit':
                        embed_path = 'data/imagenet_train_vit_embedding.pt'
                        # embed_path = 'data/imagenet_val_vit_embedding.pt'
                    if os.path.exists(embed_path):
                        data = torch.load(embed_path, map_location=device)
                        embeddings = data['embeddings']
                        labels = data['labels']
                    else:
                        embeddings, labels = get_embeddings(
                            net,
                            train_loader,
                            num_dim=model_to_num_dim[args.model],
                            dtype=torch.double,
                            device=device,
                            storage_device=device,
                        )
                        torch.save({'embeddings': embeddings, 'labels': labels}, embed_path)
                    Ensemble_model = Ensemble_fit(embeddings, labels, model_to_num_dim[args.model], num_classes, device)
                    torch.save(Ensemble_model.state_dict(), ensemble_model_path)
                    print(f"Model saved at {ensemble_model_path}")

                logits, predictive_entropy, mut_info, labels = Ensemble_evaluate(net, Ensemble_model, test_loader, model_to_num_dim[args.model], device)
                ood_logits, ood_predictive_entropy, ood_mut_info, ood_labels = Ensemble_evaluate(net, Ensemble_model, ood_test_loader, model_to_num_dim[args.model], device)
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_softmax(logits, labels)
                t_accuracy = accuracy
                ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
                t_ece = ece
                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_uncs(predictive_entropy, ood_predictive_entropy, device)
                (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_uncs(mut_info, ood_mut_info, device)

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                entropy_right = predictive_entropy[correct_mask]
                entropy_wrong = predictive_entropy[~correct_mask]
                mut_info_right = mut_info[correct_mask]
                mut_info_wrong = mut_info[~correct_mask]
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc =  get_roc_auc_uncs(entropy_right, entropy_wrong, device)
                (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc =  get_roc_auc_uncs(mut_info_right, mut_info_wrong, device)

                adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                               imagesize=model_to_input_dim[args.model],
                                                                               pin_memory=args.gpu)

                adv_logits, adv_predictive_entropy, adv_mut_info, adv_labels = Ensemble_evaluate(net, Ensemble_model, adv_test_loader, model_to_num_dim[args.model], device)


                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_uncs(predictive_entropy, adv_predictive_entropy, device)
                (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_uncs(mut_info, adv_mut_info, device)

                print('adv_m1_auroc', adv_m1_auroc)

                t_m1_auroc = ood_m1_auroc
                t_m1_auprc = ood_m1_auprc
                t_m2_auroc = ood_m2_auroc
                t_m2_auprc = ood_m2_auprc


            else:
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_ensemble(
                    net_ensemble, test_loader, device
                )
                ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_ensemble(
                    net_ensemble, test_loader, ood_test_loader, "entropy", device
                )
                (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_ensemble(
                    net_ensemble, test_loader, ood_test_loader, "mutual_information", device
                )

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                from torch.utils.data import Subset, DataLoader
                dataset = test_loader.dataset
                correct_indices = np.where(correct_mask)[0]
                right_subset = Subset(dataset, correct_indices)
                right_loader = DataLoader(right_subset, batch_size=test_loader.batch_size, shuffle=False)
                wrong_indices = np.where(~correct_mask)[0]
                wrong_subset = Subset(dataset, wrong_indices)
                wrong_loader = DataLoader(wrong_subset, batch_size=test_loader.batch_size, shuffle=False)
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_ensemble(
                    net_ensemble, right_loader, wrong_loader, "entropy", device
                )
                (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_ensemble(
                    net_ensemble, right_loader, wrong_loader, "mutual_information", device
                )

                adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,batch_size=args.batch_size)
                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_ensemble(
                    net_ensemble, test_loader, adv_test_loader, "entropy", device
                )
                (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_ensemble(
                    net_ensemble, test_loader, adv_test_loader, "mutual_information", device
                )
                print('adv_m1_auroc,adv_m2_auroc', adv_m1_auroc, adv_m2_auroc)

                if args.sample_noise:
                    adv_eps = np.linspace(0, 0.4, 9)
                    print(adv_eps)
                    for idx_ep, ep in enumerate(adv_eps):
                        adv_loader = create_adversarial_dataloader(net_ensemble[0], test_loader, device, epsilon=ep,
                                                                   batch_size=args.batch_size)
                        (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions, adv_confidences) = test_classification_net_ensemble(
                            net_ensemble, adv_loader, device
                        )
                        uncertainties = get_unc_ensemble(net_ensemble, adv_loader, "entropy", device).detach().cpu().numpy()
                        quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                        quantiles = np.delete(quantiles, 0)
                        unc_list = []
                        accuracy_list = []
                        for threshold in quantiles:
                            cer_indices = (uncertainties < threshold)
                            unc_indices = ~cer_indices
                            labels_list = np.array(adv_labels_list)
                            targets_cer = labels_list[cer_indices]
                            predictions = np.array(adv_predictions)
                            pred_cer = predictions[cer_indices]
                            targets_unc = labels_list[unc_indices]
                            pred_unc = predictions[unc_indices]
                            cer_right = np.sum(targets_cer == pred_cer)
                            cer = len(targets_cer)
                            unc_right = np.sum(targets_unc == pred_unc)
                            unc = len(targets_unc)
                            accuracy_cer = cer_right / cer
                            accuracy_unc = unc_right / unc
                            unc_list.append(threshold)
                            accuracy_list.append(accuracy_cer)
                            print('ACC:', accuracy_cer, accuracy_unc)
                        from scipy.stats import spearmanr

                        Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                        print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                        adv_unc[i][idx_ep] = uncertainties.mean()
                        adv_acc[i][idx_ep] = adv_accuracy

                        (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_ensemble(
                            net_ensemble, test_loader, adv_loader, "entropy", device
                        )
                        (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_ensemble(
                            net_ensemble, test_loader, adv_loader, "mutual_information", device
                        )
                        print('adv_m1_auroc,adv_m2_auroc', adv_m1_auroc, adv_m2_auroc)

                # Temperature scale the ensemble
                t_ensemble = []
                for model, val_loader in zip(net_ensemble, val_loaders):
                    t_model = ModelWithTemperature(model)
                    t_model.set_temperature(val_loader)
                    t_ensemble.append(t_model)

                (
                    t_conf_matrix,
                    t_accuracy,
                    t_labels_list,
                    t_predictions,
                    t_confidences,
                ) = test_classification_net_ensemble(t_ensemble, test_loader, device)
                t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

                (_, _, _), (_, _, _), t_m1_auroc, t_m1_auprc = get_roc_auc_ensemble(
                    t_ensemble, test_loader, ood_test_loader, "entropy", device
                )
                (_, _, _), (_, _, _), t_m2_auroc, t_m2_auprc = get_roc_auc_ensemble(
                    t_ensemble, test_loader, ood_test_loader, "mutual_information", device
                )

        elif args.model_type == "edl":
            if args.dataset == 'imagenet':
                edl_model_path = os.path.join(
                    args.load_loc,
                    "Run" + str(i + 1),
                    model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350_edl_model.pth",
                )

                if os.path.exists(edl_model_path):
                    print(f"Loading existing edl_model from {edl_model_path}")
                    EDL_model = EDL_load(edl_model_path, model_to_num_dim[args.model],
                                                         num_classes, device)
                else:
                    if args.model=='imagenet_vgg16':
                        embed_path = 'data/imagenet_train_vgg_embedding.pt'
                        # embed_path = 'data/imagenet_val_vgg_embedding.pt'
                    if args.model=='imagenet_wide':
                        embed_path = 'data/imagenet_train_wide_embedding.pt'
                        # embed_path = 'data/imagenet_val_wide_embedding.pt'
                    if args.model=='imagenet_vit':
                        embed_path = 'data/imagenet_train_vit_embedding.pt'
                        # embed_path = 'data/imagenet_val_vit_embedding.pt'
                    if os.path.exists(embed_path):
                        data = torch.load(embed_path, map_location=device)
                        embeddings = data['embeddings']
                        labels = data['labels']
                    else:
                        embeddings, labels = get_embeddings(
                            net,
                            train_loader,
                            num_dim=model_to_num_dim[args.model],
                            dtype=torch.double,
                            device=device,
                            storage_device=device,
                        )
                        torch.save({'embeddings': embeddings, 'labels': labels}, embed_path)
                    EDL_model = EDL_fit(embeddings, labels, model_to_num_dim[args.model], num_classes,
                                                        device)
                    torch.save(EDL_model.state_dict(), edl_model_path)
                    print(f"Model saved at {edl_model_path}")

                logits, labels = EDL_evaluate(net, EDL_model, test_loader, model_to_num_dim[args.model], device)
                ood_logits, ood_labels = EDL_evaluate(net, EDL_model, ood_test_loader, model_to_num_dim[args.model], device)
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_logits_edl(logits, labels)
                t_accuracy = accuracy

                ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
                t_ece = ece

                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_logits(logits, ood_logits, edl_unc, device)

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                # logits, _ = get_logits_labels(net, test_loader, device)
                logits_right = logits[correct_mask]
                logits_wrong = logits[~correct_mask]
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(logits_right, logits_wrong,
                                                                                      edl_unc, device)

                adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                                   imagesize=model_to_input_dim[args.model],
                                                                                   pin_memory=args.gpu)
                # (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,adv_confidences,) = test_classification_net_edl(net, adv_test_loader, device)
                # adv_logits, _ = get_logits_labels(net, adv_test_loader, device)
                adv_logits, adv_labels = EDL_evaluate(net, EDL_model, adv_test_loader, model_to_num_dim[args.model], device)

                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(logits, adv_logits, edl_unc, device)
                print('adv_m1_auroc', adv_m1_auroc)

                if args.sample_noise:
                    adv_eps = np.linspace(0, 0.4, 9)
                    for idx_ep, ep in enumerate(adv_eps):
                        adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                   batch_size=args.batch_size)
                        (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                         adv_confidences,) = test_classification_net(net, adv_loader, device)
                        adv_logits, adv_labels = EDL_evaluate(net, EDL_model, adv_loader,
                                                                    model_to_num_dim[args.model], device)
                        uncertainties = edl_unc(adv_logits).detach().cpu().numpy()
                        quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                        quantiles = np.delete(quantiles, 0)
                        unc_list = []
                        accuracy_list = []
                        for threshold in quantiles:
                            cer_indices = (uncertainties < threshold)
                            unc_indices = ~cer_indices
                            labels_list = np.array(adv_labels_list)
                            targets_cer = labels_list[cer_indices]
                            predictions = np.array(adv_predictions)
                            pred_cer = predictions[cer_indices]
                            targets_unc = labels_list[unc_indices]
                            pred_unc = predictions[unc_indices]
                            cer_right = np.sum(targets_cer == pred_cer)
                            cer = len(targets_cer)
                            unc_right = np.sum(targets_unc == pred_unc)
                            unc = len(targets_unc)
                            accuracy_cer = cer_right / cer
                            accuracy_unc = unc_right / unc
                            unc_list.append(threshold)
                            accuracy_list.append(accuracy_cer)
                            print('ACC:', accuracy_cer, accuracy_unc)
                        from scipy.stats import spearmanr

                        Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                        print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                        adv_unc[i][idx_ep] = uncertainties.mean()
                        adv_acc[i][idx_ep] = adv_accuracy


            else:
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_edl(
                    net, test_loader, device
                )
                t_accuracy = accuracy
                ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
                t_ece=ece

                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc(net, test_loader, ood_test_loader, edl_unc, device)

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                logits, _ = get_logits_labels(net, test_loader, device)
                logits_right = logits[correct_mask]
                logits_wrong = logits[~correct_mask]
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(logits_right, logits_wrong, edl_unc, device)

                adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,
                                                           batch_size=args.batch_size, edl=True)
                (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                 adv_confidences,) = test_classification_net_edl(net, adv_loader, device)
                adv_logits, _ = get_logits_labels(net, adv_loader, device)

                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(logits, adv_logits, edl_unc, device)
                print('adv_m1_auroc', adv_m1_auroc)


                if args.sample_noise:
                    adv_eps = np.linspace(0, 0.4, 9)
                    for idx_ep, ep in enumerate(adv_eps):
                        adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                   batch_size=args.batch_size, edl=True)
                        (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                         adv_confidences,) = test_classification_net_edl(net, adv_loader, device)
                        adv_logits, _ = get_logits_labels(net, adv_loader, device)
                        uncertainties = edl_unc(adv_logits).detach().cpu().numpy()
                        quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                        quantiles = np.delete(quantiles, 0)
                        unc_list = []
                        accuracy_list = []
                        for threshold in quantiles:
                            cer_indices = (uncertainties < threshold)
                            unc_indices = ~cer_indices
                            labels_list = np.array(adv_labels_list)
                            targets_cer = labels_list[cer_indices]
                            predictions = np.array(adv_predictions)
                            pred_cer = predictions[cer_indices]
                            targets_unc = labels_list[unc_indices]
                            pred_unc = predictions[unc_indices]
                            cer_right = np.sum(targets_cer == pred_cer)
                            cer = len(targets_cer)
                            unc_right = np.sum(targets_unc == pred_unc)
                            unc = len(targets_unc)
                            accuracy_cer = cer_right / cer
                            accuracy_unc = unc_right / unc
                            unc_list.append(threshold)
                            accuracy_list.append(accuracy_cer)
                            print('ACC:', accuracy_cer, accuracy_unc)
                        from scipy.stats import spearmanr

                        Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                        print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                        adv_unc[i][idx_ep] = uncertainties.mean()
                        adv_acc[i][idx_ep] = adv_accuracy

            ood_m2_auroc=ood_m1_auroc
            ood_m2_auprc = ood_m1_auprc
            err_m2_auroc = err_m1_auroc
            err_m2_auprc = err_m1_auprc
            adv_m2_auroc = adv_m1_auroc
            adv_m2_auprc = adv_m1_auprc
            t_m1_auroc=ood_m1_auroc
            t_m1_auprc=ood_m1_auprc
            t_m2_auroc=ood_m1_auroc
            t_m2_auprc=ood_m1_auprc

        elif args.model_type == "joint":
            (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_uq(
                net, test_loader, device
            )
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
            print(accuracy)
            print('ece', ece)

            t_ece=ece
            t_accuracy=accuracy

            print("SPC Model")

            logits, labels = get_logits_labels_uq(net, test_loader, device)

            soft = torch.nn.functional.softmax(logits[0], dim=1)
            delta = torch.min(torch.min(logits[2] - logits[3], logits[1] - 2 * logits[3]), 2 * logits[2] - logits[1])

            uncertainty = abs(logits[2] + logits[3] - logits[1])

            threshold = 0.05
            mask = (uncertainty < threshold).float()
            delta = delta * mask
            softmax_prob = soft + delta

            c_confidences, c_predictions = torch.max(softmax_prob, dim=1)
            c_predictions = c_predictions.tolist()
            c_confidences = c_confidences.tolist()
            c_ece = expected_calibration_error(c_confidences, c_predictions, labels_list, num_bins=15)
            print('ece', ece, 't_ece', t_ece, 'c_ece', c_ece)
            c_accuracy = accuracy_score(labels_list, c_predictions)
            print(accuracy, t_accuracy, c_accuracy)

            ood_logits, ood_labels = get_logits_labels_uq(net, ood_test_loader, device)

            (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_logits(logits, ood_logits, self_consistency,
                                                                                  device)
            (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_logits(logits[0], ood_logits[0], entropy, device)

            labels_array = np.array(labels_list)
            pred_array = np.array(predictions)
            correct_mask = labels_array == pred_array
            logits_right = [m[correct_mask] for m in logits]
            logits_wrong = [m[~correct_mask] for m in logits]
            (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(logits_right, logits_wrong,
                                                                                  self_consistency, device)
            (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_logits(logits_right[0], logits_wrong[0], entropy,
                                                                                  device)

            if args.dataset == 'imagenet':
                adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                               imagesize=model_to_input_dim[args.model],
                                                                               pin_memory=args.gpu)
            else:
                adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,
                                                                batch_size=args.batch_size, joint=True)

            adv_logits, adv_labels = get_logits_labels_uq(net, adv_test_loader, device)

            (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(logits, adv_logits, self_consistency, device)
            (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits[0], adv_logits[0], entropy, device)

            t_m1_auroc = ood_m1_auroc
            t_m1_auprc = ood_m1_auprc
            t_m2_auroc = ood_m2_auroc
            t_m2_auprc = ood_m2_auprc

        else:
            (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net(
                net, test_loader, device
            )
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
            print(accuracy)
            print('ece',ece)

            temp_scaled_net = ModelWithTemperature(net)
            temp_scaled_net.set_temperature(val_loader)
            # temp_scaled_net.set_temperature(train_loader)
            topt = temp_scaled_net.temperature

            (t_conf_matrix, t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net(
                temp_scaled_net, test_loader, device
            )
            t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)
            print('t_ece',t_ece)

            if (args.model_type == "gmm"):
                # Evaluate a GMM model
                print("GMM Model")

                if args.crossval:
                    embeddings, labels = get_embeddings(
                        net,
                        val_loader,
                        num_dim=model_to_num_dim[args.model],
                        dtype=torch.double,
                        device=device,
                        storage_device=device,
                    )
                else:
                    if args.dataset == 'imagenet':
                        if args.model == 'imagenet_vgg16':
                            embed_path = 'data/imagenet_train_vgg_embedding.pt'
                            # embed_path = 'data/imagenet_val_vgg_embedding.pt'
                        if args.model == 'imagenet_wide':
                            embed_path = 'data/imagenet_train_wide_embedding.pt'
                            # embed_path = 'data/imagenet_val_wide_embedding.pt'
                        if args.model == 'imagenet_vit':
                            embed_path = 'data/imagenet_train_vit_embedding.pt'
                            # embed_path = 'data/imagenet_val_vit_embedding.pt'
                        if os.path.exists(embed_path):
                            data = torch.load(embed_path, map_location=device)
                            embeddings = data['embeddings']
                            labels = data['labels']
                        else:
                            embeddings, labels = get_embeddings(
                                net,
                                train_loader,
                                num_dim=model_to_num_dim[args.model],
                                dtype=torch.double,
                                device=device,
                                storage_device=device,
                            )
                            torch.save({'embeddings': embeddings, 'labels': labels}, embed_path)
                    else:
                        embeddings, labels = get_embeddings(
                            net,
                            train_loader,
                            num_dim=model_to_num_dim[args.model],
                            dtype=torch.double,
                            device=device,
                            storage_device=device,
                        )

                try:
                    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes, device=device)
                    logits, labels = gmm_evaluate(
                        net, gaussians_model, test_loader, device=device, num_classes=num_classes, storage_device=device,
                    )

                    ood_logits, ood_labels = gmm_evaluate(
                        net, gaussians_model, ood_test_loader, device=device, num_classes=num_classes, storage_device=device,
                    )

                    (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_logits(
                        logits, ood_logits, logsumexp, device, confidence=True
                    )
                    (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)

                    labels_array = np.array(labels_list)
                    pred_array = np.array(predictions)
                    correct_mask = labels_array == pred_array
                    logits_right = logits[correct_mask]
                    logits_wrong = logits[~correct_mask]
                    (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(logits_right, logits_wrong,logsumexp, device, confidence=True)
                    (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_logits(logits_right, logits_wrong,entropy, device, confidence=True)

                    if args.dataset == 'imagenet':
                        adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                                       imagesize=model_to_input_dim[args.model],
                                                                                       pin_memory=args.gpu)
                    else:
                        adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,batch_size=args.batch_size)
                    (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions, adv_confidences,) = test_classification_net(net, adv_test_loader, device)
                    adv_logits, adv_labels = gmm_evaluate(net, gaussians_model, adv_test_loader, device=device, num_classes=num_classes, storage_device=device, )
                    (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(logits, adv_logits, logsumexp, device, confidence=True)
                    (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy, device)

                    if args.sample_noise:
                        adv_eps = np.linspace(0, 0.4, 9)
                        for idx_ep, ep in enumerate(adv_eps):
                            adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                       batch_size=args.batch_size)
                            (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                             adv_confidences,) = test_classification_net(net, adv_loader, device)
                            adv_logits, adv_labels = gmm_evaluate(net, gaussians_model, adv_loader, device=device, num_classes=num_classes,storage_device=device,)
                            uncertainties = -logsumexp(adv_logits).detach().cpu().numpy()
                            quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                            quantiles = np.delete(quantiles, 0)
                            unc_list = []
                            accuracy_list = []
                            for threshold in quantiles:
                                cer_indices = (uncertainties < threshold)
                                unc_indices = ~cer_indices
                                labels_list = np.array(adv_labels_list)
                                targets_cer = labels_list[cer_indices]
                                predictions = np.array(adv_predictions)
                                pred_cer = predictions[cer_indices]
                                targets_unc = labels_list[unc_indices]
                                pred_unc = predictions[unc_indices]
                                cer_right = np.sum(targets_cer == pred_cer)
                                cer = len(targets_cer)
                                unc_right = np.sum(targets_unc == pred_unc)
                                unc = len(targets_unc)
                                accuracy_cer = cer_right / cer
                                accuracy_unc = unc_right / unc
                                unc_list.append(threshold)
                                accuracy_list.append(accuracy_cer)
                                print('ACC:', accuracy_cer, accuracy_unc)
                            from scipy.stats import spearmanr
                            Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                            print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                            adv_unc[i][idx_ep]=uncertainties.mean()
                            adv_acc[i][idx_ep] = adv_accuracy

                            (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(logits, adv_logits, logsumexp, device, confidence=True)
                            (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy, device)
                            print('adv_m1_auroc,adv_m2_auroc', adv_m1_auroc, adv_m2_auroc)

                    t_m1_auroc = ood_m1_auroc
                    t_m1_auprc = ood_m1_auprc
                    t_m2_auroc = ood_m2_auroc
                    t_m2_auprc = ood_m2_auprc

                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
                    continue

            elif (args.model_type == "oc"):
                # Evaluate a OC model
                print("OC Model")

                if args.crossval:
                    embeddings, labels = get_embeddings(
                        net,
                        val_loader,
                        num_dim=model_to_num_dim[args.model],
                        dtype=torch.double,
                        device=device,
                        storage_device=device,
                    )
                else:
                    if args.dataset == 'imagenet':
                        if args.model == 'imagenet_vgg16':
                            embed_path = 'data/imagenet_train_vgg_embedding.pt'
                            # embed_path = 'data/imagenet_val_vgg_embedding.pt'
                        if args.model == 'imagenet_wide':
                            embed_path = 'data/imagenet_train_wide_embedding.pt'
                            # embed_path = 'data/imagenet_val_wide_embedding.pt'
                        if args.model == 'imagenet_vit':
                            embed_path = 'data/imagenet_train_vit_embedding.pt'
                            # embed_path = 'data/imagenet_val_vit_embedding.pt'
                        if os.path.exists(embed_path):
                            data = torch.load(embed_path, map_location=device)
                            embeddings = data['embeddings']
                            labels = data['labels']
                        else:
                            embeddings, labels = get_embeddings(
                                net,
                                train_loader,
                                num_dim=model_to_num_dim[args.model],
                                dtype=torch.double,
                                device=device,
                                storage_device=device,
                            )
                            torch.save({'embeddings': embeddings, 'labels': labels}, embed_path)
                    else:
                        embeddings, labels = get_embeddings(
                            net,
                            train_loader,
                            num_dim=model_to_num_dim[args.model],
                            dtype=torch.double,
                            device=device,
                            storage_device=device,
                        )

                try:
                    oc_model = oc_fit(embeddings=embeddings, device=device)
                    logits, OCs = oc_evaluate(
                        net, oc_model, test_loader,model_to_num_dim[args.model], device=device
                    )

                    ood_logits, ood_OCs = oc_evaluate(
                        net, oc_model, ood_test_loader, model_to_num_dim[args.model], device=device)

                    (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_logits(OCs, ood_OCs, certificate, device, confidence=True)
                    (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)

                    labels_array = np.array(labels_list)
                    pred_array = np.array(predictions)
                    correct_mask = labels_array == pred_array
                    logits_right = logits[correct_mask]
                    logits_wrong = logits[~correct_mask]
                    OCs_right = OCs[correct_mask]
                    OCs_wrong = OCs[~correct_mask]
                    (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(OCs_right, OCs_wrong, certificate, device, confidence=True)
                    (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_logits(logits_right, logits_wrong, entropy, device)

                    if args.dataset == 'imagenet':
                        adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                                       imagesize=model_to_input_dim[args.model],
                                                                                       pin_memory=args.gpu)
                    else:
                        adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,batch_size=args.batch_size)
                    (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions, adv_confidences,) = test_classification_net(net, adv_test_loader, device)
                    adv_logits, adv_OCs = oc_evaluate(net, oc_model, adv_test_loader, model_to_num_dim[args.model], device=device)
                    (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(OCs, adv_OCs, certificate, device, confidence=True)
                    (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy, device)

                    if args.sample_noise:
                        adv_eps = np.linspace(0, 0.4, 9)
                        for idx_ep, ep in enumerate(adv_eps):
                            adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                       batch_size=args.batch_size)
                            (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                             adv_confidences,) = test_classification_net(net, adv_loader, device)
                            adv_logits, adv_OCs = oc_evaluate(net, oc_model, adv_loader, model_to_num_dim[args.model], device=device)
                            uncertainties = -adv_OCs.cpu().numpy()
                            quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                            quantiles = np.delete(quantiles, 0)
                            unc_list = []
                            accuracy_list = []
                            for threshold in quantiles:
                                cer_indices = (uncertainties < threshold)
                                unc_indices = ~cer_indices
                                labels_list = np.array(adv_labels_list)
                                targets_cer = labels_list[cer_indices]
                                predictions = np.array(adv_predictions)
                                pred_cer = predictions[cer_indices]
                                targets_unc = labels_list[unc_indices]
                                pred_unc = predictions[unc_indices]
                                cer_right = np.sum(targets_cer == pred_cer)
                                cer = len(targets_cer)
                                unc_right = np.sum(targets_unc == pred_unc)
                                unc = len(targets_unc)
                                accuracy_cer = cer_right / cer
                                accuracy_unc = unc_right / unc
                                unc_list.append(threshold)
                                accuracy_list.append(accuracy_cer)
                                print('ACC:', accuracy_cer, accuracy_unc)
                            from scipy.stats import spearmanr
                            Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                            print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                            adv_unc[i][idx_ep]=uncertainties.mean()
                            adv_acc[i][idx_ep] = adv_accuracy

                            (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(OCs, adv_OCs, certificate, device, confidence=True)
                            (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy, device)
                            print('adv_m1_auroc,adv_m2_auroc', adv_m1_auroc, adv_m2_auroc)

                    t_m1_auroc = ood_m1_auroc
                    t_m1_auprc = ood_m1_auprc
                    t_m2_auroc = ood_m2_auroc
                    t_m2_auprc = ood_m2_auprc

                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
                    continue

            elif (args.model_type == "spc"):
                print("SPC Model")

                if args.crossval:
                    spc_model_path = os.path.join(
                        args.load_loc,
                        "Run" + str(i + 1),
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_valsize_"+str(args.val_size)+"_350_mar_model.pth",
                    )
                else:
                    spc_model_path = os.path.join(
                        args.load_loc,
                        "Run" + str(i + 1),
                        model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350_mar_model.pth",
                    )

                if os.path.exists(spc_model_path):
                    print(f"Loading existing spc_model from {spc_model_path}")
                    SPC_model = SPC_load(spc_model_path, model_to_num_dim[args.model], num_classes, device)
                else:
                    print(f"Model not found. Training a new one...")
                    if args.crossval:
                        embeddings, labels = get_embeddings(
                            net,
                            val_loader,
                            num_dim=model_to_num_dim[args.model],
                            dtype=torch.double,
                            device=device,
                            storage_device=device,
                        )
                    else:
                        if args.dataset == 'imagenet':
                            if args.model=='imagenet_vgg16':
                                embed_path = 'data/imagenet_train_vgg_embedding.pt'
                                # embed_path = 'data/imagenet_val_vgg_embedding.pt'
                            if args.model=='imagenet_wide':
                                embed_path = 'data/imagenet_train_wide_embedding.pt'
                                # embed_path = 'data/imagenet_val_wide_embedding.pt'
                            if args.model=='imagenet_vit':
                                embed_path = 'data/imagenet_train_vit_embedding.pt'
                                # embed_path = 'data/imagenet_val_vit_embedding.pt'
                            if os.path.exists(embed_path):
                                data = torch.load(embed_path, map_location=device)
                                embeddings = data['embeddings']
                                labels = data['labels']
                            else:
                                embeddings, labels = get_embeddings(
                                    net,
                                    train_loader,
                                    num_dim=model_to_num_dim[args.model],
                                    dtype=torch.double,
                                    device=device,
                                    storage_device=device,
                                )
                                torch.save({'embeddings': embeddings, 'labels': labels}, embed_path)
                        else:
                            embeddings, labels = get_embeddings(
                                net,
                                train_loader,
                                num_dim=model_to_num_dim[args.model],
                                dtype=torch.double,
                                device=device,
                                storage_device=device,
                            )

                    parts= model_to_last_layer[args.model].split('.')
                    net_last_layer = net

                    for attr in parts:
                        net_last_layer = getattr(net_last_layer, attr)

                    SPC_model=SPC_fit(net_last_layer, topt, embeddings, labels, model_to_num_dim[args.model], num_classes, device)
                    torch.save(SPC_model.state_dict(), spc_model_path)
                    print(f"Model saved at {spc_model_path}")

                logits,mars=SPC_evaluate(net, SPC_model, test_loader, model_to_num_dim[args.model], num_classes, device)

                soft = torch.nn.functional.softmax(logits, dim=1)
                delta = torch.min(torch.min(mars[2] - mars[3], mars[1] - 2 * mars[3]), 2 * mars[2] - mars[1])
                # delta=mars[2] - mars[3]

                uncertainty = abs(mars[2]+mars[3] - mars[1])

                threshold=0.05
                mask=(uncertainty<threshold).float()
                delta = delta*mask
                softmax_prob = soft + delta
                print(torch.sum(softmax_prob, dim=1))

                c_confidences, c_predictions = torch.max(softmax_prob, dim=1)
                c_predictions=c_predictions.tolist()
                c_confidences=c_confidences.tolist()
                c_ece = expected_calibration_error(c_confidences, c_predictions, labels_list, num_bins=15)
                print('ece', ece, 't_ece', t_ece, 'c_ece', c_ece)
                c_accuracy=accuracy_score(labels_list, c_predictions)
                print('accuracy',accuracy,'t_accuracy',t_accuracy,'c_accuracy',c_accuracy)

                ood_logits,ood_mars=SPC_evaluate(net, SPC_model, ood_test_loader, model_to_num_dim[args.model], num_classes, device)
                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc_logits(mars, ood_mars, self_consistency, device)
                (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                mars_right = [m[correct_mask] for m in mars]
                mars_wrong = [m[~correct_mask] for m in mars]
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(mars_right, mars_wrong, self_consistency, device)
                (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_logits(mars_right[0], mars_wrong[0], entropy, device)

                if args.dataset == 'imagenet':
                    adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                                   imagesize=model_to_input_dim[args.model],
                                                                                   pin_memory=args.gpu)
                else:
                    adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,batch_size=args.batch_size)
                (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,adv_confidences,) = test_classification_net(net, adv_test_loader, device)
                adv_logits, adv_mars = SPC_evaluate(net, SPC_model, adv_test_loader,model_to_num_dim[args.model], num_classes, device)
                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(mars, adv_mars,self_consistency, device)
                (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy, device)

                if args.sample_noise:
                    adv_eps = np.linspace(0, 0.4, 9)
                    print(adv_eps)
                    for idx_ep, ep in enumerate(adv_eps):
                        adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                   batch_size=args.batch_size)
                        (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                         adv_confidences,) = test_classification_net(net, adv_loader, device)
                        adv_logits, adv_mars=SPC_evaluate(net, SPC_model, adv_loader, model_to_num_dim[args.model], num_classes, device)
                        uncertainties = self_consistency(adv_mars).detach().cpu().numpy()
                        quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                        quantiles = np.delete(quantiles, 0)
                        unc_list = []
                        accuracy_list = []
                        for threshold in quantiles:
                            cer_indices = (uncertainties < threshold)
                            unc_indices = ~cer_indices
                            labels_list = np.array(adv_labels_list)
                            targets_cer = labels_list[cer_indices]
                            predictions = np.array(adv_predictions)
                            pred_cer = predictions[cer_indices]
                            targets_unc = labels_list[unc_indices]
                            pred_unc = predictions[unc_indices]
                            cer_right = np.sum(targets_cer == pred_cer)
                            cer = len(targets_cer)
                            unc_right = np.sum(targets_unc == pred_unc)
                            unc = len(targets_unc)
                            accuracy_cer = cer_right / cer
                            accuracy_unc = unc_right / unc
                            unc_list.append(threshold)
                            accuracy_list.append(accuracy_cer)
                            print('ACC:', accuracy_cer, accuracy_unc)
                        from scipy.stats import spearmanr
                        Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                        print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                        adv_unc[i][idx_ep] = uncertainties.mean()
                        adv_acc[i][idx_ep] = adv_accuracy

                        (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc_logits(mars, adv_mars, self_consistency,device)
                        (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc_logits(logits, adv_logits, entropy,device)
                        print('adv_m1_auroc,adv_m2_auroc',adv_m1_auroc,adv_m2_auroc)


                t_m1_auroc = ood_m1_auroc
                t_m1_auprc = ood_m1_auprc
                t_m2_auroc = ood_m2_auroc
                t_m2_auprc = ood_m2_auprc

            else:
                # Evaluate a normal Softmax model
                print("Softmax Model")
                (_, _, _), (_, _, _), ood_m1_auroc, ood_m1_auprc = get_roc_auc(net, test_loader, ood_test_loader, entropy, device)
                (_, _, _), (_, _, _), ood_m2_auroc, ood_m2_auprc = get_roc_auc(net, test_loader, ood_test_loader, logsumexp, device, confidence=True)

                (_, _, _), (_, _, _), t_m1_auroc, t_m1_auprc = get_roc_auc(temp_scaled_net, test_loader, ood_test_loader, entropy, device)
                (_, _, _), (_, _, _), t_m2_auroc, t_m2_auprc = get_roc_auc(temp_scaled_net, test_loader, ood_test_loader, logsumexp, device, confidence=True)

                labels_array = np.array(labels_list)
                pred_array = np.array(predictions)
                correct_mask = labels_array == pred_array
                logits, _ = get_logits_labels(net, test_loader, device)
                logits_right = logits[correct_mask]
                logits_wrong = logits[~correct_mask]
                (_, _, _), (_, _, _), err_m1_auroc, err_m1_auprc = get_roc_auc_logits(logits_right, logits_wrong, entropy, device)
                (_, _, _), (_, _, _), err_m2_auroc, err_m2_auprc = get_roc_auc_logits(logits_right, logits_wrong, logsumexp, device, confidence=True)

                if args.dataset == 'imagenet':
                    adv_test_loader = dataset_loader['imagenet_a'].get_test_loader(batch_size=args.batch_size,
                                                                                   imagesize=model_to_input_dim[args.model],
                                                                                   pin_memory=args.gpu)
                else:
                    adv_test_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=adv_ep,batch_size=args.batch_size)
                (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions, adv_confidences,) = test_classification_net(net, adv_test_loader, device)
                adv_logits, _ = get_logits_labels(net, adv_test_loader, device)
                (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc(net, test_loader, adv_test_loader, entropy, device)
                (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc(net, test_loader, adv_test_loader, logsumexp, device, confidence=True)

                if args.sample_noise:
                    adv_eps = np.linspace(0, 0.4, 9)
                    for idx_ep, ep in enumerate(adv_eps):
                        adv_loader = create_adversarial_dataloader(net, test_loader, device, epsilon=ep,
                                                                   batch_size=args.batch_size)
                        (adv_conf_matrix, adv_accuracy, adv_labels_list, adv_predictions,
                         adv_confidences,) = test_classification_net(net, adv_loader, device)
                        adv_logits, _ = get_logits_labels(net, adv_loader, device)
                        uncertainties = entropy(adv_logits).detach().cpu().numpy()
                        quantiles = np.quantile(uncertainties, np.linspace(0, 1, 10))
                        quantiles = np.delete(quantiles, 0)
                        unc_list = []
                        accuracy_list = []
                        for threshold in quantiles:
                            cer_indices = (uncertainties < threshold)
                            unc_indices = ~cer_indices
                            labels_list = np.array(adv_labels_list)
                            targets_cer = labels_list[cer_indices]
                            predictions = np.array(adv_predictions)
                            pred_cer = predictions[cer_indices]
                            targets_unc = labels_list[unc_indices]
                            pred_unc = predictions[unc_indices]
                            cer_right = np.sum(targets_cer == pred_cer)
                            cer = len(targets_cer)
                            unc_right = np.sum(targets_unc == pred_unc)
                            unc = len(targets_unc)
                            accuracy_cer = cer_right / cer
                            accuracy_unc = unc_right / unc
                            unc_list.append(threshold)
                            accuracy_list.append(accuracy_cer)
                            print('ACC:', accuracy_cer, accuracy_unc)
                        from scipy.stats import spearmanr

                        Spearman_acc, p_acc = spearmanr(unc_list, accuracy_list)
                        print("Spearman correlation:", Spearman_acc, "mean uncertainties:", uncertainties.mean())
                        adv_unc[i][idx_ep] = uncertainties.mean()
                        adv_acc[i][idx_ep] = adv_accuracy

                        (_, _, _), (_, _, _), adv_m1_auroc, adv_m1_auprc = get_roc_auc(net, test_loader, adv_loader, entropy, device)
                        (_, _, _), (_, _, _), adv_m2_auroc, adv_m2_auprc = get_roc_auc(net, test_loader, adv_loader, logsumexp, device, confidence=True)
                        print('adv_m1_auroc,adv_m2_auroc', adv_m1_auroc, adv_m2_auroc)


        accuracies.append(accuracy)
        if (args.model_type == "spc" or args.model_type == "joint"):
            c_accuracies.append(c_accuracy)
        else:
            c_accuracies.append(t_accuracy)


        # Pre-temperature results
        eces.append(ece)
        ood_m1_aurocs.append(ood_m1_auroc)
        ood_m1_auprcs.append(ood_m1_auprc)
        ood_m2_aurocs.append(ood_m2_auroc)
        ood_m2_auprcs.append(ood_m2_auprc)

        err_m1_aurocs.append(err_m1_auroc)
        err_m1_auprcs.append(err_m1_auprc)
        err_m2_aurocs.append(err_m2_auroc)
        err_m2_auprcs.append(err_m2_auprc)

        adv_m1_aurocs.append(adv_m1_auroc)
        adv_m1_auprcs.append(adv_m1_auprc)
        adv_m2_aurocs.append(adv_m2_auroc)
        adv_m2_auprcs.append(adv_m2_auprc)

        # Post-temperature results
        t_eces.append(t_ece)
        t_m1_aurocs.append(t_m1_auroc)
        t_m1_auprcs.append(t_m1_auprc)
        t_m2_aurocs.append(t_m2_auroc)
        t_m2_auprcs.append(t_m2_auprc)

        if (args.model_type == "spc" or args.model_type == "joint"):
            c_eces.append(c_ece)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if args.sample_noise:
        adv_unc_norm = (adv_unc - adv_unc.min(axis=1, keepdims=True)) / \
                       (adv_unc.max(axis=1, keepdims=True) - adv_unc.min(axis=1, keepdims=True) + 1e-8)
        mean_unc = np.mean(adv_unc_norm, axis=0)
        std_unc = np.std(adv_unc_norm, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(mean_unc, label='Uncertainty', color='orange')
        plt.fill_between(range(len(mean_unc)),
                         mean_unc - std_unc,
                         mean_unc + std_unc,
                         color='orange', alpha=0.3, label="±1 Std Dev")
        # plt.legend()
        # plt.title('Uncertainty Across Multiple Runs')
        plt.xlabel('Noise')
        plt.ylabel('Uncertainty')
        plt.savefig("adv_unc_"
            + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
            + "_"
            + args.model_type
            + "_"
            + args.dataset
            + ".png")
        plt.show()
        plt.close()


        mean_acc = np.mean(adv_acc, axis=0)
        std_acc = np.std(adv_acc, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(mean_acc, label='Accuracy', color='red')
        plt.fill_between(range(len(mean_acc)),
                         mean_acc - std_acc,
                         mean_acc + std_acc,
                         color='red', alpha=0.3, label="±1 Std Dev")
        # plt.legend()
        # plt.title('Uncertainty Across Multiple Runs')
        plt.xlabel('Noise')
        plt.ylabel('Accuracy')
        plt.savefig("adv_acc_"
            + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
            + "_"
            + args.model_type
            + "_"
            + args.dataset
            + ".png")
        plt.show()
        plt.close()

        save_dir = "curve_data"
        os.makedirs(save_dir, exist_ok=True)

        if args.sn:
            prefix = f"{args.dataset}_{args.model_type}_{args.model}_SN"
        else:
            prefix = f"{args.dataset}_{args.model_type}_{args.model}"


    accuracy_tensor = torch.tensor(accuracies)
    c_accuracy_tensor = torch.tensor(c_accuracies)
    ece_tensor = torch.tensor(eces)
    ood_m1_auroc_tensor = torch.tensor(ood_m1_aurocs)
    m1_auprc_tensor = torch.tensor(ood_m1_auprcs)
    ood_m2_auroc_tensor = torch.tensor(ood_m2_aurocs)
    ood_m2_auprc_tensor = torch.tensor(ood_m2_auprcs)

    err_m1_auroc_tensor = torch.tensor(err_m1_aurocs)
    err_m1_auprc_tensor = torch.tensor(err_m1_auprcs)
    err_m2_auroc_tensor = torch.tensor(err_m2_aurocs)
    err_m2_auprc_tensor = torch.tensor(err_m2_auprcs)

    adv_m1_auroc_tensor = torch.tensor(adv_m1_aurocs)
    adv_m1_auprc_tensor = torch.tensor(adv_m1_auprcs)
    adv_m2_auroc_tensor = torch.tensor(adv_m2_aurocs)
    adv_m2_auprc_tensor = torch.tensor(adv_m2_auprcs)

    t_ece_tensor = torch.tensor(t_eces)
    t_m1_auroc_tensor = torch.tensor(t_m1_aurocs)
    t_m1_auprc_tensor = torch.tensor(t_m1_auprcs)
    t_m2_auroc_tensor = torch.tensor(t_m2_aurocs)
    t_m2_auprc_tensor = torch.tensor(t_m2_auprcs)

    c_ece_tensor = torch.tensor(c_eces)

    mean_accuracy = torch.mean(accuracy_tensor)
    mean_c_accuracy = torch.mean(c_accuracy_tensor)
    mean_ece = torch.mean(ece_tensor)
    mean_ood_m1_auroc = torch.mean(ood_m1_auroc_tensor)
    mean_m1_auprc = torch.mean(m1_auprc_tensor)
    mean_m2_auroc = torch.mean(ood_m2_auroc_tensor)
    mean_m2_auprc = torch.mean(ood_m2_auprc_tensor)

    mean_err_m1_auroc = torch.mean(err_m1_auroc_tensor)
    mean_err_m1_auprc = torch.mean(err_m1_auprc_tensor)
    mean_err_m2_auroc = torch.mean(err_m2_auroc_tensor)
    mean_err_m2_auprc = torch.mean(err_m2_auprc_tensor)

    mean_adv_m1_auroc = torch.mean(adv_m1_auroc_tensor)
    mean_adv_m1_auprc = torch.mean(adv_m1_auprc_tensor)
    mean_adv_m2_auroc = torch.mean(adv_m2_auroc_tensor)
    mean_adv_m2_auprc = torch.mean(adv_m2_auprc_tensor)

    mean_t_ece = torch.mean(t_ece_tensor)
    mean_t_m1_auroc = torch.mean(t_m1_auroc_tensor)
    mean_t_m1_auprc = torch.mean(t_m1_auprc_tensor)
    mean_t_m2_auroc = torch.mean(t_m2_auroc_tensor)
    mean_t_m2_auprc = torch.mean(t_m2_auprc_tensor)

    mean_c_ece = torch.mean(c_ece_tensor)

    std_accuracy = torch.std(accuracy_tensor) / math.sqrt(accuracy_tensor.shape[0])
    std_c_accuracy = torch.std(c_accuracy_tensor) / math.sqrt(c_accuracy_tensor.shape[0])
    std_ece = torch.std(ece_tensor) / math.sqrt(ece_tensor.shape[0])
    std_ood_m1_auroc = torch.std(ood_m1_auroc_tensor) / math.sqrt(ood_m1_auroc_tensor.shape[0])
    std_m1_auprc = torch.std(m1_auprc_tensor) / math.sqrt(m1_auprc_tensor.shape[0])
    std_m2_auroc = torch.std(ood_m2_auroc_tensor) / math.sqrt(ood_m2_auroc_tensor.shape[0])
    std_m2_auprc = torch.std(ood_m2_auprc_tensor) / math.sqrt(ood_m2_auprc_tensor.shape[0])
    std_err_m1_auroc = torch.std(err_m1_auroc_tensor) / math.sqrt(err_m1_auroc_tensor.shape[0])
    std_err_m1_auprc = torch.std(err_m1_auprc_tensor) / math.sqrt(err_m1_auprc_tensor.shape[0])
    std_err_m2_auroc = torch.std(err_m2_auroc_tensor) / math.sqrt(err_m2_auroc_tensor.shape[0])
    std_err_m2_auprc = torch.std(err_m2_auprc_tensor) / math.sqrt(err_m2_auprc_tensor.shape[0])
    std_adv_m1_auroc = torch.std(adv_m1_auroc_tensor) / math.sqrt(adv_m1_auroc_tensor.shape[0])
    std_adv_m1_auprc = torch.std(adv_m1_auprc_tensor) / math.sqrt(adv_m1_auprc_tensor.shape[0])
    std_adv_m2_auroc = torch.std(adv_m2_auroc_tensor) / math.sqrt(adv_m2_auroc_tensor.shape[0])
    std_adv_m2_auprc = torch.std(adv_m2_auprc_tensor) / math.sqrt(adv_m2_auprc_tensor.shape[0])

    std_t_ece = torch.std(t_ece_tensor) / math.sqrt(t_ece_tensor.shape[0])
    std_t_m1_auroc = torch.std(t_m1_auroc_tensor) / math.sqrt(t_m1_auroc_tensor.shape[0])
    std_t_m1_auprc = torch.std(t_m1_auprc_tensor) / math.sqrt(t_m1_auprc_tensor.shape[0])
    std_t_m2_auroc = torch.std(t_m2_auroc_tensor) / math.sqrt(t_m2_auroc_tensor.shape[0])
    std_t_m2_auprc = torch.std(t_m2_auprc_tensor) / math.sqrt(t_m2_auprc_tensor.shape[0])

    std_c_ece = torch.std(c_ece_tensor) / math.sqrt(c_ece_tensor.shape[0])

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["ood_m1_auroc"] = mean_ood_m1_auroc.item()
    res_dict["mean"]["ood_m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["ood_m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["ood_m2_auprc"] = mean_m2_auprc.item()
    res_dict["mean"]["t_ece"] = mean_t_ece.item()
    res_dict["mean"]["t_m1_auroc"] = mean_t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = mean_t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = mean_t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = mean_t_m2_auprc.item()
    res_dict["mean"]["c_ece"] = mean_c_ece.item()

    res_dict["std"] = {}
    res_dict["std"]["accuracy"] = std_accuracy.item()
    res_dict["std"]["ece"] = std_ece.item()
    res_dict["std"]["ood_m1_auroc"] = std_ood_m1_auroc.item()
    res_dict["std"]["ood_m1_auprc"] = std_m1_auprc.item()
    res_dict["std"]["ood_m2_auroc"] = std_m2_auroc.item()
    res_dict["std"]["ood_m2_auprc"] = std_m2_auprc.item()
    res_dict["std"]["t_ece"] = std_t_ece.item()
    res_dict["std"]["t_m1_auroc"] = std_t_m1_auroc.item()
    res_dict["std"]["t_m1_auprc"] = std_t_m1_auprc.item()
    res_dict["std"]["t_m2_auroc"] = std_t_m2_auroc.item()
    res_dict["std"]["t_m2_auprc"] = std_t_m2_auprc.item()
    res_dict["std"]["c_ece"] = std_c_ece.item()

    res_dict["values"] = {}
    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["ood_m1_auroc"] = ood_m1_aurocs
    res_dict["values"]["ood_m1_auprc"] = ood_m1_auprcs
    res_dict["values"]["ood_m2_auroc"] = ood_m2_aurocs
    res_dict["values"]["ood_m2_auprc"] = ood_m2_auprcs
    res_dict["values"]["t_ece"] = t_eces
    res_dict["values"]["t_m1_auroc"] = t_m1_aurocs
    res_dict["values"]["t_m1_auprc"] = t_m1_auprcs
    res_dict["values"]["t_m2_auroc"] = t_m2_aurocs
    res_dict["values"]["t_m2_auprc"] = t_m2_auprcs
    res_dict["values"]["c_ece"] = c_eces

    res_dict["info"] = vars(args)

    print(f"{mean_accuracy.item() * 100:.2f} ± {std_accuracy.item() * 100:.2f}")
    print(f"{mean_c_accuracy.item() * 100:.2f} ± {std_c_accuracy.item() * 100:.2f}")
    print(f"{mean_ece.item()*100:.2f} ± {std_ece.item()*100:.2f}")
    print(f"{mean_t_ece.item()*100:.2f} ± {std_t_ece.item()*100:.2f}")
    print(f"{mean_c_ece.item() * 100:.2f} ± {std_c_ece.item() * 100:.2f}")
    print(f"{mean_adv_m1_auroc.item()*100:.2f} ± {std_adv_m1_auroc.item()*100:.2f}")
    print(f"{mean_err_m1_auroc.item()*100:.2f} ± {std_err_m1_auroc.item()*100:.2f}")
    print(f"{mean_ood_m1_auroc.item()*100:.2f} ± {std_ood_m1_auroc.item()*100:.2f}")


    with open(
        "res_"
        + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
        + "_"
        + args.model_type
        + "_"
        + args.dataset
        + "_"
        + args.ood_dataset
        + ".json",
        "w",
    ) as f:
        json.dump(res_dict, f)
