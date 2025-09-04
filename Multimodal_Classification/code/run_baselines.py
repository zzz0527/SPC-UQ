import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose, Normalize

from baselines.classifiers import AudioClassifier, ImageClassifier, MultimodalClassifier, TextClassifier
from baselines.de_model import DEModel
from baselines.dirichlet import DirichletModel
from baselines.mc_model import MCDModel
from baselines.mar_regressor import AudioMAR,ImageMAR,TextMAR,MultimodalMAR
from baselines.consistency_model import SCModel
from baselines.pure_model import PureModel
from data_generation.text_processing import extract_deep_text_features
from dataset import LUMADataset

pl.seed_everything(42)
# torch.multiprocessing.set_start_method('spawn', force=True)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

try:
    import torchaudio
    import soundfile as sf
    import torch

    def safe_load(path):
        waveform, sr = sf.read(path, always_2d=True)
        waveform = torch.tensor(waveform.T).float()  # [channel, time]
        return waveform, sr

    torchaudio.load = safe_load
    print("Patched torchaudio.load to use soundfile.")
except Exception as e:
    print("Failed to patch torchaudio.load:", e)

class Text2FeatureTransform():
    def __init__(self, features_path):
        with open(features_path, 'rb') as f:
            self.features = np.load(f)

    def __call__(self, text, idx):
        return self.features[idx]


class PadCutToSizeAudioTransform():
    def __init__(self, size):
        self.size = size

    def __call__(self, audio):
        if audio.shape[-1] < self.size:
            audio = torch.nn.functional.pad(audio, (0, self.size - audio.shape[-1]))
        elif audio.shape[-1] > self.size:
            audio = audio[:, :self.size]
        return audio


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--noise_type', type=str, default='')
    args, unknown = parser.parse_known_args()
    return args, unknown


args, unknown = parse_args()
if args.noise_type not in ['', 'diversity', 'label', 'sample']:
    raise ValueError('Invalid noise type')

train_audio_data_path = test_audio_data_path = ood_audio_data_path = 'data/audio' if args.noise_type != 'sample' else 'data/noisy_audio'
suffix = ''
if args.noise_type == 'diversity':
    suffix = '_diversity'
elif args.noise_type == 'label':
    suffix = '_noise_label'
elif args.noise_type == 'sample':
    suffix = '_noise'

train_audio_path = f'{train_audio_data_path}/datalist_train{suffix if args.noise_type != "sample" else ""}.csv'
test_audio_path = f'{test_audio_data_path}/datalist_test{suffix if args.noise_type not in ("sample", "diversity") else ""}.csv'
ood_audio_path = f'{ood_audio_data_path}/datalist_ood{suffix if args.noise_type != "sample" else ""}.csv'
train_image_path = f'data/image_data_train{suffix}.pickle'
train_text_path = f'data/text_data_train{suffix}.tsv'
test_image_path = f'data/image_data_test{suffix if args.noise_type != "diversity" else ""}.pickle'
test_text_path = f'data/text_data_test{suffix if args.noise_type != "diversity" else ""}.tsv'
ood_image_path = f'data/image_data_ood{suffix}.pickle'
ood_text_path = f'data/text_data_ood{suffix}.tsv'

# extract_deep_text_features(train_text_path, output_path=f'text_features_train_{args.noise_type}.npy')
# extract_deep_text_features(test_text_path, output_path=f'text_features_test_{args.noise_type}.npy')
# extract_deep_text_features(ood_text_path, output_path=f'text_features_ood_{args.noise_type}.npy')

print(f'Loading data from {train_audio_path}, {test_audio_path}, {ood_audio_path}, {train_image_path}, {train_text_path}, {test_image_path}, {test_text_path}, {ood_image_path}, {ood_text_path}')
image_transform = Compose([
    ToTensor(),
    # Resize((224, 224)),
    Normalize(mean=(0.51, 0.49, 0.44),
              std=(0.27, 0.26, 0.28))
])
train_dataset = LUMADataset(train_image_path, train_audio_path, train_audio_data_path, train_text_path,
                            text_transform=Text2FeatureTransform(f'text_features_train_{args.noise_type}.npy'),
                            audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                            image_transform=image_transform)

test_dataset = LUMADataset(test_image_path, test_audio_path, test_audio_data_path, test_text_path,
                           text_transform=Text2FeatureTransform(f'text_features_test_{args.noise_type}.npy'),
                           audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                           image_transform=image_transform)

ood_dataset = LUMADataset(ood_image_path, ood_audio_path, ood_audio_data_path, ood_text_path,
                          text_transform=Text2FeatureTransform(f'text_features_ood_{args.noise_type}.npy'),
                          audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                          image_transform=image_transform, ood=True)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)),
                                                                           len(train_dataset) - int(
                                                                               0.8 * len(train_dataset))])

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Now we can use the loaders to train a model

acc_dict = {}
classes = 42
mc_samples = 100
dropout_p = 0.3
n_ensemble = 10

pure_models = [PureModel(c, classes, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
                                                                  MultimodalClassifier]]
mc_models = [MCDModel(c, classes, mc_samples, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
                                                                   MultimodalClassifier]]
de_models = [DEModel(c, classes, n_ensemble, 0) for c in [ImageClassifier, AudioClassifier, TextClassifier,
                                                                  MultimodalClassifier]]
dir_models = [DirichletModel(MultimodalClassifier, classes, dropout=0)]
sc_models = [SCModel(c, classes, 0) for c in [ImageMAR, AudioMAR, TextMAR, MultimodalMAR]]
models = mc_models + de_models + dir_models + sc_models


# model = SCModel(MultimodalMAR, classes, dropout_p)
# checkpoint = torch.load("lightning_logs/version_2/checkpoints/last.ckpt", map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])
# uncertainty_values = {}
# model_name = model.__class__.__name__ + '_' + model.model.__class__.__name__
# model.set_stage('mar')
# trainer = pl.Trainer(max_epochs=300,
#                              accelerator="auto",  # or "gpu" if torch.cuda.is_available() else "cpu"
#                              devices=1,
#                              # gpus=1 if torch.cuda.is_available() else 0,
#                              callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
#                                         pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)])
# print('Testing model')
# trainer.test(model, test_loader)
# acc_dict[model_name] = trainer.callback_metrics["test_acc"]
# acc_dict[model_name + '_ale'] = trainer.callback_metrics["test_ale"]
# acc_dict[model_name + '_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
# aleatoric_uncertainties = model.aleatoric_uncertainties
# epistemic_uncertainties = model.epistemic_uncertainties
# print(aleatoric_uncertainties, epistemic_uncertainties)
# print('Testing OOD')
# trainer.test(model, ood_loader)
# acc_dict[model_name + '_ood_ale'] = trainer.callback_metrics["test_ale"]
# acc_dict[model_name + '_ood'] = trainer.callback_metrics["test_acc"]
# acc_dict[model_name + '_ood_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
# aleatoric_uncertainties_ood = model.aleatoric_uncertainties
# epistemic_uncertainties_ood = model.epistemic_uncertainties
# print(aleatoric_uncertainties_ood, epistemic_uncertainties_ood)
# auc_score = roc_auc_score(
#     np.concatenate([np.zeros(len(epistemic_uncertainties)), np.ones(len(epistemic_uncertainties_ood))]),
#     np.concatenate([epistemic_uncertainties, epistemic_uncertainties_ood]))
# uncertainty_values[f'{model_name}_epistemic'] = epistemic_uncertainties
# uncertainty_values[f'{model_name}_aleatoric'] = aleatoric_uncertainties
# acc_dict[model_name + '_ood_auc'] = auc_score
# print(auc_score)
# for key, value in acc_dict.items():
#     print(f'{key}: {value}')



uncertainty_values = {}
for classifier in models:
    model = classifier
    try:
        model_name = classifier.__class__.__name__ + '_' + classifier.model.__class__.__name__
    except AttributeError:
        model_name = classifier.__class__.__name__ + '_' + classifier.models[0].__class__.__name__

    if classifier.__class__.__name__ == 'SCModel':
        # Stage1: train CLS head
        model.set_stage('cls')
        trainer1 = pl.Trainer(
            max_epochs=300,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)
            ]
        )
        trainer1.fit(model, train_loader, val_loader)

        # Stage2: freeze backbone，train MAR head
        model.set_stage('mar')
        trainer = pl.Trainer(
            max_epochs=300,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)
            ]
        )
        trainer.fit(model, train_loader, val_loader)

    else:
        trainer = pl.Trainer(max_epochs=300,
                             accelerator="auto",  # or "gpu" if torch.cuda.is_available() else "cpu"
                             devices=1,
                             # gpus=1 if torch.cuda.is_available() else 0,
                             enable_progress_bar=False,
                             callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                                        pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)])
        trainer.fit(model, train_loader, val_loader)



    print('Testing model')
    trainer.test(model, test_loader)
    acc_dict[model_name] = trainer.callback_metrics["test_acc"]
    acc_dict[model_name + '_ale'] = trainer.callback_metrics["test_ale"]
    acc_dict[model_name + '_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
    aleatoric_uncertainties = model.aleatoric_uncertainties
    epistemic_uncertainties = model.epistemic_uncertainties
    print(aleatoric_uncertainties, epistemic_uncertainties)
    print('Testing OOD')
    trainer.test(model, ood_loader)
    acc_dict[model_name + '_ood_ale'] = trainer.callback_metrics["test_ale"]
    acc_dict[model_name + '_ood'] = trainer.callback_metrics["test_acc"]
    acc_dict[model_name + '_ood_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
    aleatoric_uncertainties_ood = model.aleatoric_uncertainties
    epistemic_uncertainties_ood = model.epistemic_uncertainties
    print(aleatoric_uncertainties_ood, epistemic_uncertainties_ood)

    auc_score = roc_auc_score(
        np.concatenate([np.zeros(len(epistemic_uncertainties)), np.ones(len(epistemic_uncertainties_ood))]),
        np.concatenate([epistemic_uncertainties, epistemic_uncertainties_ood]))

    uncertainty_values[f'{model_name}_epistemic'] = epistemic_uncertainties
    uncertainty_values[f'{model_name}_aleatoric'] = aleatoric_uncertainties
    acc_dict[model_name + '_ood_auc'] = auc_score
    print('AAAAAAAAAAAAAAAAAAAAAA', auc_score)
for key, value in acc_dict.items():
    print(f'{key}: {value}')

uncertainty_df = pd.DataFrame(uncertainty_values)
uncertainty_df.to_csv(f'uncertainty_values{args.noise_type}.csv')