from .SPC import SPC_UNet
from .ensemble import Ensemble_UNet
from .dropout import Dropout_UNet
from .evidential import EvidentialUNet
from .QRevidential import QRevidentialUNet
from .QR import QR_UNet
import torch

def get_correct_model(trainer):
    """Helper function to grab the right model based solely on the trainer."""

    trainer_lookup = trainer.__name__.lower()

    if trainer_lookup == 'spc':
        model_pointer=SPC_UNet

    if trainer_lookup == 'ensemble':
        model_pointer = Ensemble_UNet

    if trainer_lookup == 'dropout':
        model_pointer = Dropout_UNet

    if trainer_lookup == 'evidential':
        model_pointer = EvidentialUNet

    if trainer_lookup == 'qrevidential':
        model_pointer = QRevidentialUNet

    if trainer_lookup == 'qroc':
        model_pointer = QR_UNet


    return model_pointer

def load_depth_model(method, path):
    if method == "Dropout" or method == "dropout":
        model = Dropout_UNet()

    elif method == "ensemble" or method == "Ensemble":
        model = Ensemble_UNet()

    elif method == "spc" or method == "SPC":
        model = SPC_UNet()

    elif method == "evidential" or method == "Evidential":
        model = EvidentialUNet()

    elif method == "qrevidential" or method == "QREvidential":
        model = QRevidentialUNet()

    elif method == "qroc" or method == "QROC":
        model = QR_UNet()

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    return model