import os

import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class LUMADataset(Dataset):
    def __init__(self, image_path, audio_path, audio_data_path, text_path, image_transform=None, audio_transform=None,
                 text_transform=None,
                 target_transform=None, ood=False):
        self.image_path = image_path
        self.audio_path = audio_path
        self.audio_data_path = audio_data_path
        self.text_path = text_path
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.ood = ood
        self._load_data()

        self.label_mapping = {'man': 0, 'boy': 1, 'house': 2, 'woman': 3, 'girl': 4, 'table': 5, 'road': 6, 'horse': 7,
                              'dog': 8, 'ship': 9, 'bird': 10, 'mountain': 11, 'bed': 12, 'train': 13, 'bridge': 14,
                              'fish': 15, 'cloud': 16, 'chair': 17, 'cat': 18, 'baby': 19, 'castle': 20, 'forest': 21,
                              'television': 22, 'bear': 23, 'camel': 24, 'sea': 25, 'fox': 26, 'plain': 27, 'bus': 28,
                              'snake': 29, 'lamp': 30, 'clock': 31, 'lion': 32, 'tank': 33, 'palm': 34, 'rabbit': 35,
                              'pine': 36, 'cattle': 37, 'oak': 38, 'mouse': 39, 'frog': 40, 'ray': 41, 'bicycle': 42,
                              'truck': 43, 'elephant': 44, 'roses': 45, 'wolf': 46, 'telephone': 47, 'bee': 48,
                              'whale': 49}

    def _load_data(self):
        # Load data from file
        self.image_data = pd.read_pickle(self.image_path)
        self.audio_data = pd.read_csv(self.audio_path)
        self.text_data = pd.read_csv(self.text_path, sep='\t')
        self.targets = self.image_data['label'].values

    def __getitem__(self, index):
        image = self.image_data.loc[:, 'image'].iloc[index]
        # print(image)
        audio_path = self.audio_data.loc[:, 'path'].iloc[index]
        audio, sr = torchaudio.load(os.path.join(self.audio_data_path, audio_path))
        text = self.text_data.loc[:, 'text'].iloc[index]
        target = self.label_mapping[self.targets[index]] if not self.ood else 0

        from PIL import Image
        import numpy as np
        if isinstance(image, np.ndarray):
            # 如果是 uint8, 就可以直接转 PIL 后走 transform
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            image = Image.fromarray(image)
        else:
            raise TypeError(f"Image type is {type(image)}, expected np.ndarray")

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
        if self.text_transform is not None:
            text = self.text_transform(text, index)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(image)

        return image, audio, text, target

    def __len__(self):
        return len(self.image_data)


if __name__ == '__main__':
    image_path = 'data/image_data_train.pickle'
    audio_path = 'data/audio/datalist_train.csv'
    audio_data_path = 'data/audio'
    text_path = 'data/text_data_train.tsv'

    dataset = LUMADataset(image_path, audio_path, audio_data_path, text_path)
    print('\n'.join([dataset[i][-2] for i in range(1800, 1810)]))