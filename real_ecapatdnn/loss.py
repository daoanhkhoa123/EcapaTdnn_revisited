import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerClipLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))
    
    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeds = F.normalize(embeds, dim=1)
        temp = torch.exp(self.log_temp)
        sim = embeds @ embeds.t() / temp
        B = embeds.size(0)
        mask = torch.eye(B, dtype=torch.bool, device=embeds.device)
        labels = labels.view(-1, 1)
        positives = labels == labels.t()
        loss = []
        for i in range(B):
            pos_mask = positives[i] & ~mask[i]
            if pos_mask.sum() == 0:
                continue
            logits = sim[i]
            target = pos_mask.float() / pos_mask.sum()
            log_prob = F.log_softmax(logits, dim=0)
            loss_i = -(target * log_prob).sum()
            loss.append(loss_i)
        if len(loss) == 0:
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)
        return torch.stack(loss).mean()
