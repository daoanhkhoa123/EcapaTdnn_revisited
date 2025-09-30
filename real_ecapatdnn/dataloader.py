from dataclasses import dataclass
import os
import soundfile as sf

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd

@dataclass
class VSAVSDataloader_config:
    df_path:str
    data_prefix_path:str
    bonafide_only:bool = True

class VSAVSDataset_SpkerEmbed(Dataset):
    BONDAFINE_ID = 1

    def __init__(self, config:VSAVSDataloader_config) -> None:
        super().__init__()
        df = pd.read_csv(config.df_path)
        # filters bonafide
        if config.bonafide_only:
            df[df["att_type"] == "bonafide"]

        # build labels
        unique_speakers = df["speaker_id_num"].unique()
        id2label = {spk: idx for idx, spk in enumerate(unique_speakers)}
        df["spk_label"] = df["speaker_id_num"].map(id2label)
        
        self.df = df
        self.prefix  = config.data_prefix_path
        self.n_speakers = len(unique_speakers)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = os.path.join(self.prefix, row["path"])
        label = row["spk_label"]
        audio, _ = sf.read(file_path)
        audio = torch.tensor(audio, dtype=torch.float32)
        return audio, label
    
def collate_fn(batch):
    audios, labels = zip(*batch)
    audios_padded = rnn_utils.pad_sequence(audios, batch_first=True) # type: ignore
    labels = torch.tensor(labels, dtype=torch.long)
    return audios_padded, labels
