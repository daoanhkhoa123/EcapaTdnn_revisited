from dataclasses import dataclass
import os
import soundfile as sf

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

@dataclass
class VSAVSDataloader_config:
    df_path: str
    data_prefix_path: str

class VSAVSDataset_Spoo(Dataset):
    def __init__(self, config: VSAVSDataloader_config, max_len: int = 64600, target_sr: int = 16000) -> None:
        super().__init__()
        df = pd.read_csv(config.df_path)
        self.df = df
        self.prefix = config.data_prefix_path
        self.max_len = max_len
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = os.path.join(self.prefix, row["path"])
        audio, sr = sf.read(file_path)

        # Convert to tensor
        audio = torch.tensor(audio, dtype=torch.float32)

        # Resample if needed
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.target_sr)

        # Pad or crop
        if audio.shape[0] > self.max_len:
            audio = audio[:self.max_len]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.max_len - audio.shape[0]))

        # Label: bonafide=1, spoof=0
        label = int(row["att_type"] == "bonafide")
        return audio, label

def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = torch.stack(audios)  # already same length
    labels = torch.tensor(labels, dtype=torch.long)
    return audios, labels

