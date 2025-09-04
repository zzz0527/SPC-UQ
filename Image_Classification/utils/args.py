"""
Contains common args used in different scripts.
"""

import argparse


def training_args():

    default_dataset = "cifar10"
    dataset_root = "./"
    ood_dataset = "svhn"
    train_batch_size = 128
    test_batch_size = 128

    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    val_size=0.1
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 200
    save_loc = "./"
    saved_model_name = "resnet50_350.model"
    epoch = 350
    first_milestone = 150  # Milestone for change in lr
    second_milestone = 250  # Milestone for change in lr

    model = "resnet50"
    sn_coeff = 3.0

    parser = argparse.ArgumentParser(
        description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=dataset_root,
        dest="dataset_root",
        help="path of a dataset (useful for dirty mnist)",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument(
        "-b", type=int, default=train_batch_size, dest="train_batch_size", help="Batch size",
    )
    parser.add_argument(
        "-tb", type=int, default=test_batch_size, dest="test_batch_size", help="Test Batch size",
    )

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)

    parser.add_argument(
        "--val_size", type=float, default=val_size, dest="val_size", help="validation size for cross validation",
    )

    parser.add_argument("-e", type=int, default=epoch, dest="epoch", help="Number of training epochs")
    parser.add_argument(
        "--lr", type=float, default=learning_rate, dest="learning_rate", help="Learning rate",
    )
    parser.add_argument("--mom", type=float, default=momentum, dest="momentum", help="Momentum")
    parser.add_argument(
        "--nesterov", action="store_true", dest="nesterov", help="Whether to use nesterov momentum in SGD",
    )
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        "--decay", type=float, default=weight_decay, dest="weight_decay", help="Weight Decay",
    )
    parser.add_argument(
        "--opt", type=str, default=optimiser, dest="optimiser", help="Choice of optimisation algorithm",
    )

    parser.add_argument(
        "--loss", type=str, default=loss, dest="loss_function", help="Loss function to be used for training",
    )
    parser.add_argument(
        "--loss-mean",
        action="store_true",
        dest="loss_mean",
        help="whether to take mean of loss instead of sum to train",
    )
    parser.set_defaults(loss_mean=False)

    parser.add_argument(
        "--log-interval", type=int, default=log_interval, dest="log_interval", help="Log Interval on Terminal",
    )
    parser.add_argument(
        "--save-interval", type=int, default=save_interval, dest="save_interval", help="Save Interval on Terminal",
    )
    parser.add_argument(
        "--saved_model_name",
        type=str,
        default=saved_model_name,
        dest="saved_model_name",
        help="file name of the pre-trained model",
    )
    parser.add_argument(
        "--save-path", type=str, default=save_loc, dest="save_loc", help="Path to export the model",
    )

    parser.add_argument(
        "--first-milestone",
        type=int,
        default=first_milestone,
        dest="first_milestone",
        help="First milestone to change lr",
    )
    parser.add_argument(
        "--second-milestone",
        type=int,
        default=second_milestone,
        dest="second_milestone",
        help="Second milestone to change lr",
    )
    return parser

def eval_args():
    default_dataset = "cifar10"
    ood_dataset = "svhn"
    batch_size = 128
    load_loc = "../Models/Normal/"
    model = "vgg16"
    sn_coeff = 3.0
    runs = 5
    ensemble = 5
    val_size = 0.1
    model_type = "softmax"

    parser = argparse.ArgumentParser(
        description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",
    )
    parser.add_argument(
        "--ood_dataset",
        type=str,
        default=ood_dataset,
        dest="ood_dataset",
        help="OOD dataset for given training dataset",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=False)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-b", type=int, default=batch_size, dest="batch_size", help="Batch size")
    parser.add_argument(
        "--load-path", type=str, default=load_loc, dest="load_loc", help="Path to load the model from",
    )
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument(
        "--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",
    )

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)
    parser.add_argument(
        "--val_size", type=float, default=val_size, dest="val_size", help="validation size for cross validation",
    )
    parser.add_argument(
        "-crossval", action="store_true", dest="crossval", help="whether to start cross validation",
    )
    parser.set_defaults(crossval=False)
    parser.add_argument("--ensemble", type=int, default=ensemble, dest="ensemble", help="Number of models in ensemble")
    parser.add_argument(
        "--model-type",
        type=str,
        default=model_type,
        choices=["softmax", "ensemble", "gmm", "spc", "edl", "oc", "joint"],
        dest="model_type",
        help="Type of model to load for evaluation.",
    )
    parser.add_argument(
        "-sample_noise", action="store_true", dest="sample_noise", help="whether to generate noise samples in evaluation" ,
    )


    return parser

def laplace_eval_args():
    default_dataset = "cifar10"
    ood_dataset = "svhn"
    batch_size = 128
    load_loc = "../Models/Normal/"
    model = "vgg16"
    runs = 5
    val_size = 0.1

    parser = argparse.ArgumentParser(
        description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")

    parser.add_argument(
        "--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",
    )
    parser.add_argument(
        "--ood_dataset",
        type=str,
        default=ood_dataset,
        dest="ood_dataset",
        help="OOD dataset for given training dataset",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=False)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--load-path", type=str, default=load_loc, dest="load_loc", help="Path to load the model from",
    )
    parser.add_argument(
        "--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",
    )
    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=3.0, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)

    parser.add_argument(
        "--val_size", type=float, default=val_size, dest="val_size", help="validation size for cross validation",
    )
    parser.add_argument(
        "-crossval", action="store_true", dest="crossval", help="whether to start cross validation",
    )
    parser.set_defaults(crossval=False)

    parser.add_argument('--models_root', type=str, default='./models',
                        help='root of pre-trained models')
    parser.add_argument('--model_seed', type=int, default=None,
                        help='random seed with which model(s) were trained')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--hessians_root', type=str, default='./hessians',
                        help='root of pre-computed Hessians')
    parser.add_argument('--method', type=str,
                        choices=['map', 'ensemble',
                                 'laplace', 'mola',
                                 'swag', 'multi-swag',
                                 'bbb', 'csghmc'],
                        default='laplace',
                        help='name of method to use')
    parser.add_argument('--pred_type', type=str,
                        choices=['nn', 'glm'],
                        default='glm',
                        help='type of approximation of predictive distribution')
    parser.add_argument('--link_approx', type=str,
                        choices=['mc', 'probit', 'bridge'],
                        default='probit',
                        help='type of approximation of link function')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='nr. of MC samples for approximating the predictive distribution')

    parser.add_argument('--likelihood', type=str, choices=['classification', 'regression'],
                        default='classification', help='likelihood for Laplace')
    parser.add_argument('--subset_of_weights', type=str, choices=['last_layer', 'all'],
                        default='last_layer', help='subset of weights for Laplace')
    parser.add_argument('--backend', type=str, choices=['backpack', 'kazuki'], default='backpack')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--hessian_structure', type=str, choices=['diag', 'kron', 'full'],
                        default='kron', help='structure of the Hessian approximation')
    parser.add_argument('--last_layer_name', type=str, default='classifier',
                        help='name of the last layer of the model')
    parser.add_argument('--prior_precision', default=1.0,
                        help='prior precision to use for computing the covariance matrix')
    parser.add_argument('--optimize_prior_precision', default=None,
                        choices=['marglik', 'nll'],
                        help='optimize prior precision according to specified method')
    parser.add_argument('--prior_structure', type=str, default='scalar',
                        choices=['scalar', 'layerwise', 'all'])
    parser.add_argument('--sigma_noise', type=float, default=None,
                        help='noise standard deviation for regression (if -1, optimize it)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the likelihood.')

    parser.add_argument('--swag_n_snapshots', type=int, default=40,
                        help='number of snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_batches', type=int, default=None,
                        help='number of batches between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_epochs', type=int, default=1,
                        help='number of epochs between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_lr', type=float, default=1e-2,
                        help='learning rate for [Multi]SWAG')
    parser.add_argument('--swag_bn_update_subset', type=float, default=1.0,
                        help='fraction of train data for updating the BatchNorm statistics for [Multi]SWAG')

    parser.add_argument('--nr_components', type=int, default=1,
                        help='number of mixture components to use')
    parser.add_argument('--mixture_weights', type=str,
                        choices=['uniform', 'optimize'],
                        default='uniform',
                        help='how the mixture weights for MoLA are chosen')

    parser.add_argument("--model", type=str, default=model, dest="model", help='the neural network model architecture')
    parser.add_argument('--no_dropout', action='store_true', help='only for WRN-fixup.')
    parser.add_argument('--data_parallel', action='store_true',
                        help='if True, use torch.nn.DataParallel(model)')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='batch size for testing')
    parser.add_argument('--val_set_size', type=int, default=2000,
                        help='size of validation set (taken from test set)')
    parser.add_argument('--use_temperature_scaling', default=False,
                        help='if True, calibrate model using temperature scaling')

    parser.add_argument('--job_id', type=int, default=0,
                        help='job ID, leave at 0 when running locally')
    parser.add_argument('--config', default=None, nargs='+',
                        help='YAML config file path')
    parser.add_argument('--run_name', type=str, help='overwrite save file name')
    parser.add_argument('--noda', action='store_true')
    parser.add_argument(
        "-sample_noise", action="store_true", dest="sample_noise",
        help="whether to generate noise samples in evaluation",
    )

    return parser