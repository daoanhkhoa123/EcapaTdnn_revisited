from dataclasses import dataclass
import os
import soundfile as sf

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0,8), time_mask_width =(0,10)) -> None:
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
    
    def mask_along_axis(self, x:torch.Tensor, dim):
        origin_size = x.shape
        bsize, ftdim, time = x.shape
        if dim == 1:
            dee = ftdim
            width_range = self.freq_mask_width
        else:
            dee = time
            width_range = self.time_mask_width

        mask_len = torch.randint(*width_range, size=(bsize, 1), 
                            device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, int(max(1, dee - mask_len.max())), size=(bsize,1),
                                 device=x.device).unsqueeze(2)
        arange  = torch.arange(dee, device=x.device).view(1,1,-1)
        
        mask = (mask_pos <= arange) * (arange < (mask_pos+mask_len))
        mask =  mask.any(dim=1)
        mask = mask.unsqueeze(2) if dim==1 else mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*origin_size)
    
    def forward(self, x:torch.Tensor):
        with torch.no_grad():
            x  =  self.mask_along_axis(x, dim=2)
            x = self.mask_along_axis(x, dim=1)
            return x
    
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
