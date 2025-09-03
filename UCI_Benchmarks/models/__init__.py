from .SPC import SPC_net
from .ensemble import Ensemble_model
from .dropout import Dropout_model
from .evidential import Evidential_net
from .QRensemble import QREnsemble_model
from .QRevidential import QRevidential_net
from .QR import QR_net

def get_correct_model(trainer):
    """Helper function to grab the right model based solely on the trainer."""

    trainer_lookup = trainer.__name__.lower()

    if trainer_lookup == 'spc':
        model_pointer=SPC_net

    if trainer_lookup == 'ensemble':
        model_pointer = Ensemble_model

    if trainer_lookup == 'dropout':
        model_pointer = Dropout_model

    if trainer_lookup == 'evidential':
        model_pointer = Evidential_net

    if trainer_lookup == 'qrensemble':
        model_pointer = QREnsemble_model

    if trainer_lookup == 'qrevidential':
        model_pointer = QRevidential_net

    if trainer_lookup == 'qroc':
        model_pointer = QR_net

    return model_pointer

