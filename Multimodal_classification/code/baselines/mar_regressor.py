import torch
import torch.nn.functional as F
from torchvision import models

from baselines.utils import MCDropout


class ImageMAR(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.2, monte_carlo=False, aleatoric=False):
        super(ImageMAR, self).__init__()
        self.image_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(64 * 6 * 6, num_classes)

        def make_branch():
            num_layers=1
            layers = []
            in_features = 64 * 6 * 6
            neurons = 512
            for _ in range(num_layers):
                layers.append(torch.nn.Linear(in_features, neurons))
                layers.append(torch.nn.ReLU())
                # layers.append(nn.Dropout(dropout_p))
                in_features = neurons
                # neurons //= 2
            return torch.nn.Sequential(*layers), torch.nn.Linear(in_features, num_classes)

        self.hidden_mar, self.mar = make_branch()
        self.hidden_mar_up, self.mar_up = make_branch()
        self.hidden_mar_down, self.mar_down = make_branch()


    def forward(self, x):
        image, audio, text = x
        image = self.image_model(image.float())
        cls=self.classifier(image)
        mar = self.mar(self.hidden_mar(image))
        mar_up = self.mar_up(self.hidden_mar_up(image))
        mar_down = self.mar_down(self.hidden_mar_down(image))
        return cls, mar, mar_up, mar_down


class AudioMAR(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False, aleatoric=False):
        super(AudioMAR, self).__init__()
        self.audio_model = torch.nn.Sequential(  # from batch_size x 1 x 128 x 128 spectrogram
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(64 * 14 * 14, num_classes)

        def make_branch():
            num_layers=1
            layers = []
            in_features = 64 * 14 * 14
            neurons = 512
            for _ in range(num_layers):
                layers.append(torch.nn.Linear(in_features, neurons))
                layers.append(torch.nn.ReLU())
                # layers.append(nn.Dropout(dropout_p))
                in_features = neurons
                # neurons //= 2
            return torch.nn.Sequential(*layers), torch.nn.Linear(in_features, num_classes)

        self.hidden_mar, self.mar = make_branch()
        self.hidden_mar_up, self.mar_up = make_branch()
        self.hidden_mar_down, self.mar_down = make_branch()


    def forward(self, x):
        image, audio, text = x
        audio = self.audio_model(audio)
        cls=self.classifier(audio)
        mar = self.mar(self.hidden_mar(audio))
        mar_up = self.mar_up(self.hidden_mar_up(audio))
        mar_down = self.mar_down(self.hidden_mar_down(audio))
        return cls, mar, mar_up, mar_down


class TextMAR(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False, aleatoric=False):
        super(TextMAR, self).__init__()
        self.text_model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Linear(256, num_classes)

        def make_branch():
            num_layers=1
            layers = []
            in_features = 256
            neurons = 256
            for _ in range(num_layers):
                layers.append(torch.nn.Linear(in_features, neurons))
                layers.append(torch.nn.ReLU())
                # layers.append(nn.Dropout(dropout_p))
                in_features = neurons
                # neurons //= 2
            return torch.nn.Sequential(*layers), torch.nn.Linear(in_features, num_classes)

        self.hidden_mar, self.mar = make_branch()
        self.hidden_mar_up, self.mar_up = make_branch()
        self.hidden_mar_down, self.mar_down = make_branch()

    def forward(self, x):
        image, audio, text = x
        text = self.text_model(text)
        cls=self.classifier(text)
        mar = self.mar(self.hidden_mar(text))
        mar_up = self.mar_up(self.hidden_mar_up(text))
        mar_down = self.mar_down(self.hidden_mar_down(text))
        return cls, mar, mar_up, mar_down


class MultimodalMAR(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False, dirichlet=False, aleatoric=False):
        super(MultimodalMAR, self).__init__()
        self.image_model = ImageMAR(num_classes, dropout, monte_carlo, aleatoric)
        self.audio_model = AudioMAR(num_classes, dropout, monte_carlo, aleatoric)
        self.text_model = TextMAR(num_classes, dropout, monte_carlo, aleatoric)
        self.monte_carlo = monte_carlo
        self.dirichlet = dirichlet
        self.aleatoric = aleatoric
        if dirichlet and monte_carlo:
            raise ValueError("Dirichlet and Monte Carlo cannot be used together")

    def forward(self, x):
        image_outputs = self.image_model(x)
        audio_outputs = self.audio_model(x)
        text_outputs = self.text_model(x)

        # MARs = (image_outputs + audio_outputs + text_outputs) / 3
        outputs = tuple((i + a + t) / 3 for i, a, t in zip(image_outputs, audio_outputs, text_outputs))

        return outputs
