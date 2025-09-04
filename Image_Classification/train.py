"""
Script for training a single model for OOD detection.
"""

import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.imagenet as imagenet
import data.ood_detection.tinyimagenet as tinyimagenet

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.resnet_edl import resnet50_edl
from net.wide_resnet import wrn
from net.wide_resnet_edl import wrn_edl
from net.wide_resnet_uq import wrn_uq
from net.vgg import vgg16
from net.vgg_edl import vgg16_edl
from net.vgg_uq import vgg16_uq

# Import train and validation utilities
from utils.args import training_args
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "imagenet": 1000, "tinyimagenet": 200}

model_to_input_dim = {"resnet50": 32, "resnet50_edl":32, "wide_resnet": 32,"wide_resnet_edl": 32,"wide_resnet_uq": 32, "vgg16": 32, "vgg16_edl": 32,"vgg16_uq": 32}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "imagenet": imagenet,
    "tinyimagenet": tinyimagenet,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50_edl": resnet50_edl,
    "wide_resnet": wrn,
    "wide_resnet_edl": wrn_edl,
    "wide_resnet_uq": wrn_uq,
    "vgg16": vgg16,
    "vgg16_edl": vgg16_edl,
    "vgg16_uq": vgg16_uq,
}


if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    if args.model=='vgg16_edl' or args.model=='resnet50_edl' or args.model=='wide_resnet_edl':
        args.loss_function='edl_loss'

    if args.model=='vgg16_uq' or args.model=='wide_resnet_uq':
        args.loss_function='mar_loss'

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
        imagenet="imagenet" in args.dataset
    )

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay = 1e-4 if "imagenet" in args.dataset else args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if "imagenet" in args.dataset:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30], gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
        )

    if "imagenet" in args.dataset:
        args.train_batch_size=128
        args.epoch=90
        args.save_interval=10

    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        imagesize=model_to_input_dim[args.model],
        augment=args.data_aug,
        val_size=args.val_size,
        val_seed=args.seed,
        pin_memory=args.gpu,
    )

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")

    training_set_loss = {}

    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    print("Model save name", save_name)

    for epoch in range(0, args.epoch):
        print("Epoch %d/%d" % (epoch + 1, args.epoch))
        print("Starting epoch", epoch)
        train_loss = train_single_epoch(
            epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,
        )

        training_set_loss[epoch] = train_loss
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))

        scheduler.step()

        # if (epoch + 1) % args.save_interval == 0:
        #     saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
        #     torch.save(net.state_dict(), saved_name)

    if args.val_size == 0.1:
        saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"

    else:
        saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + "_0" + str(int(args.val_size * 10)) + ".model"

    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)

    writer.close()
    with open(saved_name[: saved_name.rfind("_")] + "_train_loss.json", "a") as f:
        json.dump(training_set_loss, f)