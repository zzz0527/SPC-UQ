from .spc import SPC_cls_net
from .ensemble import Ensemble_cls_model
from .dropout import Dropout_cls_model
from .evidential import Evidential_cls_net

def get_correct_model(trainer,dataset):
    """Helper function to grab the right model based solely on the trainer."""

    trainer_lookup = trainer.__name__.lower()

    if trainer_lookup == 'spc':
        model_pointer=SPC_cls_net

    if trainer_lookup == 'ensemble':
        model_pointer = Ensemble_cls_model

    if trainer_lookup == 'dropout':
        model_pointer = Dropout_cls_model

    if trainer_lookup == 'evidential':
        model_pointer = Evidential_cls_net

    return model_pointer

