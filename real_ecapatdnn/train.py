from dataclasses import dataclass

from .model import ECAPA_TDNN
from .loss import SpeakerClipLoss
from .dataloader import VSAVSDataloader_config, VSAVSDataset_SpkerEmbed, collate_fn

from torch.utils.data import DataLoader

BATCH_SIZE = 8
NUM_WORKERS = 2

def train(traindataconfig:VSAVSDataloader_config, valdataconfig:VSAVSDataloader_config):
    model = ECAPA_TDNN(1024)
    loss = SpeakerClipLoss()

    train_dataset = VSAVSDataset_SpkerEmbed(traindataconfig)
    val_dataset = VSAVSDataset_SpkerEmbed(valdataconfig)
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
    )
    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # usually no shuffle for validation
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)
