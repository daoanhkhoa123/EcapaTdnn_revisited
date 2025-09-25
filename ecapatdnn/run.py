from dataclasses import dataclass, asdict
import logging, argparse, os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import torch

from .dataloader import VSAVSDataset_SpkerEmbed, VSAVSDataloader_config, collate_fn
from .loss import AAAMSoftmax_config, AAMSoftmax
from .model import Ecapa_Tdnn, Ecapa_dim


@dataclass
class Train_config:
    epochs: int 
    batch_size:int
    device:str
    lr:float = 1e-3
    weight_decay:float=2e-5
    lr_decay: float = 0.95
    test_step: int = 10


def setup():
    parser = argparse.ArgumentParser(description="Speaker Verification Training")
    parser.add_argument("--df_path", type=str, required=True,
                        help="Path to metadata CSV")
    parser.add_argument("--data_prefix_path", type=str, required=True,
                        help="Root folder where audio files are stored")
    parser.add_argument("--bondafide_only", type=bool, default=True)
    parser.add_argument("--augument", type=bool,  default=True)

    # training config
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--test_step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model config
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=768)

    # loss config
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=30)

    return parser


def setup_config(parser: argparse.ArgumentParser) -> tuple[Train_config, VSAVSDataloader_config, Ecapa_dim, AAAMSoftmax_config]:
    args = parser.parse_args()
    data_config = VSAVSDataloader_config(
        df_path=args.df_path,
        data_prefix_path=args.data_prefix_path,
        bonafide_only=args.bondafide_only,
        augument=args.augument
    )

    model_config = Ecapa_dim(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )

    # speaker count from df
    n_class = len(pd.read_csv(args.df_path)["speaker_id_num"].unique())

    loss_config = AAAMSoftmax_config(
        embed_dim=args.embed_dim,
        n_class=n_class,
        m=args.margin,
        s=args.scale
    )

    train_config = Train_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        test_step=args.test_step
    )

    return train_config, data_config, model_config, loss_config

def setup_logger():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join("logs", f"traintorch_testsmall_{current_time}.txt")
    logging.basicConfig(filename=filename, filemode="w", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Logger Initialized")
    return filename


def log_configs(model_cfg, loss_cfg, train_cfg):
    logging.info("===== Training Configuration =====")
    for k, v in asdict(train_cfg).items():
        logging.info(f"{k}: {v}")
    logging.info("===== Model Dimensions =====")
    for k, v in asdict(model_cfg).items():
        logging.info(f"{k}: {v}")
    logging.info("===== Loss Dimensions =====")
    for k, v in asdict(loss_cfg).items():
        logging.info(f"{k}: {v}")

def train(train_config: Train_config, data_config:VSAVSDataloader_config, model_config:Ecapa_dim, loss_config:AAAMSoftmax_config):
    model = Ecapa_Tdnn(model_config).to(train_config.device)
    loss_fn = AAMSoftmax(loss_config).to(train_config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()),
                                  lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config.test_step,
                                                 gamma=train_config.lr_decay)
    
    dataset = VSAVSDataset_SpkerEmbed(data_config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=train_config.batch_size,
                                         shuffle=True, num_workers=2, drop_last=True,
                                         collate_fn=collate_fn)

    loss = torch.zeros([])
    model.train()
    for epoch in range(1, train_config.epochs + 1):
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{train_config.epochs} Loss {loss.item():.4f}")

        for data, labels in pbar:
            optimizer.zero_grad()

            data, labels = data.to(train_config.device), labels.to(train_config.device)
            print(data.shape)
            m = Ecapa_Tdnn(Ecapa_dim(hidden_dim=1024, embed_dim=768))
            inp = torch.randn(data.shape)  # 1-second examples
            out = m(inp)
            print(out.shape)  # expect [2, 1024]
            print("aaaaaaaaaaaaaaaa")
            embedding = model(data)
            print("aaaaaaaaaaaaaaaa")
            loss = loss_fn(embedding, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        logging.info(f"[Epoch {epoch}] Loss={loss.item()} ") # type: iignore)

    return model, loss_fn


if __name__ == "__main__":
    parser = setup()
    train_config, data_config, model_config, loss_config = setup_config(parser)
    log_file = setup_logger()
    # print(model_config)
    # assert model_config == Ecapa_dim(1024, 768)
    # m = Ecapa_Tdnn(model_config)
    # inp = torch.randn(2, 16000)  # 1T-second examples
    # out = m(inp)
    # print(out.shape)  # expect [2, 1024]
    # raise ValueError()


    log_configs(model_config, loss_config, train_config)
    logging.info(f"Logging to {log_file}")
    train(train_config, data_config, model_config, loss_config)
    logging.info("Done training!")