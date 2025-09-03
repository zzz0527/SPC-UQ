# This file contains the hyperparmeters for reproducing the benchmark UCI
# dataset regression tasks. The hyperparmeters, which included the learning_rate
# and batch_size were optimized using grid search on an 80-20 train-test split
# of the dataset with the optimal resulting hyperparmeters saved in this file
# for quick reloading.

data_params = {
    'yacht': {'learning_rate': 1e-4, 'batch_size': 16},
    'naval': {'learning_rate': 5e-4, 'batch_size': 64},
    'concrete': {'learning_rate': 1e-4, 'batch_size': 64},
    'energy': {'learning_rate': 1e-4, 'batch_size': 32},
    'kin8nm': {'learning_rate': 1e-4, 'batch_size': 64},
    'power': {'learning_rate': 1e-4, 'batch_size': 64},
    'boston': {'learning_rate': 1e-4, 'batch_size': 64},
    'wine': {'learning_rate': 1e-4, 'batch_size': 64},
    'protein': {'learning_rate': 1e-4, 'batch_size': 64},
}
