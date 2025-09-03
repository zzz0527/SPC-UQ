import argparse
import numpy as np
import math
import torch
import data_loader
import trainers
import models
from data.data_params import data_params

# ===================== Argument Parser ===================== #
parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=20, type=int,
                    help="Number of trials for repeated training.")
parser.add_argument("--num-epochs", default=400, type=int)
parser.add_argument('--datasets', nargs='+', default=[
                    'boston', 'concrete', 'energy', 'kin8nm',
                    'naval', 'power', 'protein', 'wine', 'yacht'],
                    choices=[
                    'boston', 'concrete', 'energy', 'kin8nm',
                    'naval', 'power', 'protein', 'wine', 'yacht'])
parser.add_argument('--noise', default='', choices=['', 'tri', 'log'])
parser.add_argument('--load-model', action='store_false', help='Load saved model if available.')
parser.add_argument('--model-dir', default='save', help='Directory to save/load models.')
args = parser.parse_args()

# ===================== Config Setup ======================== #
training_schemes = [trainers.Dropout, trainers.Ensemble, trainers.Evidential, trainers.QRevidential, trainers.QROC, trainers.SPC]
datasets = args.datasets
num_trials = args.num_trials
num_epochs = args.num_epochs

print('Datasets:', datasets)
print('Training Schemes:', [t.__name__ for t in training_schemes])

# ===================== Metrics Init ======================== #
metric_names = ["RMSE", "MPIW", "PICP", "PICP_UP", "PICP_DOWN",
                "IS", "ECE", "SPIECE", "SMSE", "TRAINSPEED", "TESTSPEED"]

metrics = {name: np.zeros((len(datasets), len(training_schemes), num_trials))
           for name in metric_names}

# ===================== Main Experiment ===================== #
for d_idx, dataset in enumerate(datasets):
    for m_idx, trainer_cls in enumerate(training_schemes):
        np.random.seed(27)
        torch.manual_seed(27)
        for trial in range(num_trials):
            print(f"\n[Dataset: {dataset}] [Model: {trainer_cls.__name__}] [Trial: {trial+1}/{num_trials}]")

            train_set, test_set, y_mu, y_std = data_loader.load_dataset(
                dataset, noise=args.noise, split_seed=trial, return_as_tensor=True)
            input_dim = train_set.tensors[0].shape[1]
            batch_size = data_params[dataset]["batch_size"]

            model_fn = models.get_correct_model(trainer=trainer_cls)
            model = model_fn(input_shape=input_dim)
            trainer = trainer_cls(model, dataset=dataset, noise=args.noise, tag=trial,
                                  learning_rate=data_params[dataset]["learning_rate"],load_model=args.load_model, model_dir=args.model_dir)

            results = trainer.train(train_set, test_set, y_mu, y_std,
                                    num_epochs=num_epochs, batch_size=batch_size, verbose=False)

            rmse, mpiw, picp, picp_up, picp_down, is_score, ece, spearman_rmse, train_time, test_time = results[1:]

            spiece = abs(picp_up - 0.95) + abs(picp_down - 0.95)

            metrics["RMSE"][d_idx, m_idx, trial] = rmse
            metrics["MPIW"][d_idx, m_idx, trial] = mpiw
            metrics["PICP"][d_idx, m_idx, trial] = picp
            metrics["PICP_UP"][d_idx, m_idx, trial] = picp_up
            metrics["PICP_DOWN"][d_idx, m_idx, trial] = picp_down
            metrics["IS"][d_idx, m_idx, trial] = is_score
            metrics["ECE"][d_idx, m_idx, trial] = ece
            metrics["SPIECE"][d_idx, m_idx, trial] = spiece
            metrics["SMSE"][d_idx, m_idx, trial] = spearman_rmse
            metrics["TRAINSPEED"][d_idx, m_idx, trial] = train_time
            metrics["TESTSPEED"][d_idx, m_idx, trial] = test_time

# ===================== Print Summary ======================== #
def print_metric_summary(name, values):
    mean = np.mean(values, axis=-1)
    std = np.std(values, axis=-1) / math.sqrt(num_trials)
    print(f"\n=== {name} ===")
    for d in range(len(datasets)):
        for m in range(len(training_schemes)):
            print(f"{datasets[d]:>8s} | {training_schemes[m].__name__:<15s} : {mean[d][m]:.2f} ± {std[d][m]:.2f}")


for key in ["RMSE", "IS", "ECE", "SPIECE", "SMSE"]:
    print_metric_summary(key, metrics[key])
