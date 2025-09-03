import argparse
import numpy as np
import data_loader
import trainers
import models
from data.data_params import data_params

# ======================== Argument Parser ======================== #
parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=5, type=int,
                    help="Number of trials to repeat training for statistically significant results.")
parser.add_argument("--num-epochs", default=40, type=int,
                    help="Number of training epochs.")
args = parser.parse_args()

# ======================== Config Setup =========================== #
training_schemes = [trainers.SPC]
datasets = ["mnist"]
num_trials = args.num_trials
num_epochs = args.num_epochs

# ======================== Metric Initialization ================== #
print("Datasets:", datasets, "Models:", training_schemes)

ACC = np.zeros((len(datasets), len(training_schemes), num_trials))  # Overall accuracy
ACC_C = np.zeros_like(ACC)  # Accuracy on confident predictions
ACC_U = np.zeros_like(ACC)  # Accuracy on uncertain predictions
AUROC = np.zeros_like(ACC)  # AUROC for ID vs OOD uncertainty
TRAINSPEED = np.zeros_like(ACC)  # Training time (ms)
TESTSPEED = np.zeros_like(ACC)  # Evaluation time (ms)

# ======================== Main Loop ============================== #
for d_idx, dataset in enumerate(datasets):
    for m_idx, trainer_class in enumerate(training_schemes):
        for trial in range(num_trials):
            print(f"\n[Trial {trial + 1}/{num_trials}] Dataset: {dataset}, Model: {trainer_class.__name__}")

            # Load dataset and get model
            train_set, test_set, ood_set, input_shape, num_classes = data_loader.load_dataset(dataset)
            batch_size = data_params[dataset]["batch_size"]
            model_fn = models.get_correct_model(trainer=trainer_class, dataset=dataset)
            model = model_fn(input_shape=input_shape, output_shape=num_classes)

            trainer = trainer_class(model, learning_rate=data_params[dataset]["learning_rate"], optimizer_type='ADAM')
            results = trainer.train(train_set, test_set, ood_set,
                                    num_classes=num_classes,
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    verbose=True,
                                    freq=5)

            model, acc, acc_c, acc_u, auroc, train_time, test_time = results

            ACC[d_idx, m_idx, trial] = acc
            ACC_C[d_idx, m_idx, trial] = acc_c
            ACC_U[d_idx, m_idx, trial] = acc_u
            AUROC[d_idx, m_idx, trial] = auroc
            TRAINSPEED[d_idx, m_idx, trial] = train_time
            TESTSPEED[d_idx, m_idx, trial] = test_time


# ======================== Result Summary ========================= #
def print_metric_summary(name, values):
    mean = np.mean(values, axis=-1)
    std = np.std(values, axis=-1)
    print(f"\n=== {name} ===")
    for d in range(len(datasets)):
        for m in range(len(training_schemes)):
            print(f"{datasets[d]:>8s} | {training_schemes[m].__name__:<15s} : {mean[d][m]:.4f} ± {std[d][m]:.4f}")


print_metric_summary("Accuracy", ACC)
print_metric_summary("Confident Accuracy", ACC_C)
print_metric_summary("Uncertain Accuracy", ACC_U)
print_metric_summary("AUROC", AUROC)
print_metric_summary("Train Time (ms)", TRAINSPEED)
print_metric_summary("Test Time (ms)", TESTSPEED)
