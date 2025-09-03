import argparse
# import cv2
import h5py
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

import models
import trainers


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="spc", type=str,
                    choices=["evidential", "qrevidential", "dropout", "ensemble", "qroc", "spc"])
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--iters", default=60000, type=int)
parser.add_argument("--learning-rate", default=1e-4, type=float)
args = parser.parse_args()

### Load the data
def _load_depth():
    train = h5py.File("data/depth_train.h5", "r")
    test = h5py.File("data/depth_test.h5", "r")
    return (train["image"][:], train["depth"][:]), (test["image"][:], test["depth"][:])


(x_train_all, y_train_all), (x_test, y_test) = _load_depth()
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.01, random_state=42, shuffle=True)

### Create the trainer
if args.model == "evidential":
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate, lam=1e-1, epsilon=0., maxi_rate=0.)

if args.model == "qrevidential":
    trainer_obj = trainers.QRevidential
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate, lam=1e-1, epsilon=0., maxi_rate=0.)

elif args.model == "dropout":
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate)

elif args.model == "ensemble":
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate)

elif args.model == "spc":
    trainer_obj = trainers.SPC
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate)

elif args.model == "qroc":
    trainer_obj = trainers.QROC
    model_generator = models.get_correct_model(trainer=trainer_obj)
    model = model_generator()
    trainer = trainer_obj(model, learning_rate=args.learning_rate)


# training_schemes = [trainers.Dropout, trainers.Ensemble, trainers.Evidential, trainers.SPC]
# for scheme in training_schemes:
#     print(scheme)
#     trainer_obj = scheme
#     model_generator = models.get_correct_model(trainer=trainer_obj)
#     model = model_generator()
#     trainer = trainer_obj(model, learning_rate=args.learning_rate)
#
#     model = trainer.train(x_train, x_val, y_train, y_val, x_test, y_test, np.array([[1.]]), iters=args.iters, batch_size=args.batch_size, verbose=True)
#
#     print("Done training!")

## Train the model
model = trainer.train(x_train, x_val, y_train, y_val, x_test, y_test, iters=args.iters, batch_size=args.batch_size, verbose=True)
print("Done training!")
