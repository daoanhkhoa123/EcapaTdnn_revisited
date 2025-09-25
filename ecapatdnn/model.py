from dataclasses import dataclass

import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as fn

class ConvFilter(nn.Module):
    def __init__(self, channels, bottle_neck) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottle_neck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottle_neck, channels, kernel_size=1, padding=0),

            nn.Sigmoid()
        )


    def forward(self, input):
        x = self.layers(input)
        return input * x

class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, 
                 kernel_size, dilation=1, scale=8, bottle_neck=128) -> None:
        super().__init__() 
        self.width = planes//scale
        self.n_blocks = scale-1

        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, self.width * scale, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.width*scale)
            )
        
        padding = kernel_size//2*dilation
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(self.width, self.width, kernel_size=kernel_size,
                                     dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(self.width))
            for _ in range(self.n_blocks)]
        )

        self.conv2=nn.Sequential(
            nn.Conv1d(self.width*scale, planes, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(planes),
        )


        self.convfiler = ConvFilter(planes, bottle_neck)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)

        splits = torch.split(out, self.width, 1)
        for i in range(self.n_blocks):
            sp = splits[i] if i == 0 else sp + splits[i] # type: ignore
            sp = self.blocks[i](sp)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        
        out = torch.cat((out, splits[self.n_blocks]), 1)

        out = self.conv2(out)
        out = self.convfiler(out)
        out = out + input
        return out
            
class PreEmphasis(nn.Module):
    def __init__(self, coef:float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        self.register_buffer("flipped_filter",
                             torch.tensor([-self.coef, 1.], dtype=torch.float32)
                             .unsqueeze(0).unsqueeze(0))

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        input= fn.pad(input,(1,0), "reflect")
        return fn.conv1d(input, self.flipped_filter).squeeze(1)

@dataclass
class Ecapa_dim:
    hidden_dim:int
    embed_dim:int
    # augument:bool

class Ecapa_Tdnn(nn.Module):
    def __init__(self, ecapa_dim:Ecapa_dim) -> None:
        super().__init__()
        self.fbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft= 512, win_length=400,
                                                 hop_length=160, f_min=20, f_max=7600, 
                                                 window_fn=torch.hamming_window, n_mels=80)
            )

        hid_dim = ecapa_dim.hidden_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, hid_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim)
        )
        self.bottles = nn.ModuleList(
            [BottleNeck(hid_dim, hid_dim, kernel_size=3, dilation=2, scale=8),
             BottleNeck(hid_dim, hid_dim,kernel_size=3, dilation=3, scale=8),
             BottleNeck(hid_dim, hid_dim, kernel_size=3, dilation=4, scale=8)]
        )
        self.conv_toattn = nn.Sequential(
            nn.Conv1d(3*hid_dim, 1536, kernel_size=1),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(3072, ecapa_dim.embed_dim),
            nn.GELU(),
            nn.Linear(ecapa_dim.embed_dim, ecapa_dim.embed_dim),
            nn.LayerNorm(ecapa_dim.embed_dim)
        )

    def forward(self, x:torch.Tensor):
        with torch.no_grad():
            x = self.fbank(x) + 1e-6
            x = x.log() # mel
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.conv1(x)

        for i in range(len(self.bottles)):
            x_temp = x if i == 0 else x_temp + x_res # type: ignore
            x_res = self.bottles[i](x_temp)
            out = x_res if i == 0 else torch.cat((out, x_res), dim=1) # type: ignore
        
        x = self.conv_toattn(out) # type: ignore

        t = x.size(-1)
        mean = torch.mean(x, dim=2, keepdim=True).repeat(1,1,t)
        sigma = torch.sqrt(torch.var(x,dim=2, keepdim=True
                                     ).clamp(min=1e-4)).repeat(1,1,t)
        global_x =  torch.cat((x, mean, sigma), dim=1)
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(torch.sum(x**2 * w, dim=2) - mu**2).clamp(min=1e-4)
        x= torch.cat((mu,sg), dim=1)
        x = self.conv2(x)
        return x
    
if __name__ == "__main__":
    m = Ecapa_Tdnn(Ecapa_dim(hidden_dim=512, embed_dim=1024))
    inp = torch.randn(1, 103200)  # 1-second examples
    out = m(inp)
    print(out.shape)  # expect [2, 1024]
