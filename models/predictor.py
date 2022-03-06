import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


class Audio2Spec(nn.Module):
    r"""Waveform to spectrogram."""

    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    ):
        super().__init__()
        window = torch.hann_window(win_length).float()
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        re, im = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
        ).unbind(-1)
        return re ** 2 + im ** 2
        # return torch.sqrt(torch.clamp(re ** 2 + im ** 2, min=1e-9))


class MLP(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        self.hp = hp
        self.device = device

        modules = []
        for p, h in zip(hp.hiddens, hp.hiddens[1:]):
            modules.append(nn.Linear(p, h))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hp.hiddens[-1], 1))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x = self.net(x)
        return x


class CNN(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        self.hp = hp
        self.device = device

        self.norm = np.inf if hp.norm == 'inf' else hp.norm

        dim = 32
        self.spec = Audio2Spec()
        modules = [wn_conv1d(513, dim, 3, padding=1)]
        for _ in range(hp.n_layers):
            modules.append(nn.LeakyReLU(0.2))
            modules.append(wn_conv1d(dim, min(512, dim * 2), 3, padding=1))
            modules.append(nn.AvgPool1d(2))
            dim = min(512, dim * 2)

        self.net = nn.Sequential(*modules)
        self.proj = nn.Linear(512, 1)

    def forward(self, x):
        x = x / torch.norm(x, p=self.norm, keepdim=True, dim=1)
        x = x.reshape(x.size(0), 1, -1)
        x = self.net(self.spec(x))
        x = torch.mean(x, dim=-1)
        x = self.proj(x)
        return x


class Predictor(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        self.hp = hp
        self.device = device
        self.net = CNN(hp, device)

    def forward(self, x):
        return self.net(x)

    def fit(self, x_train, y_train):
        hp = self.hp
        device = self.device
        self.inp_dim = x_train.shape[-1]

        self.net.apply(weight_reset)
        self.to(device)
        self.train()

        samples = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
        dataloader = DataLoader(samples, batch_size=hp.batch_size,
                                drop_last=False, shuffle=True)
        criteria = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        for epoch in (pbar := tqdm(range(hp.epochs))):
            mae, size = 0, 0
            for x, y in dataloader:
                loss = criteria(self(x.to(device)), torch.log(y.to(device)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mae += loss.item()*x.shape[0]
                size += x.shape[0]

            scheduler.step()
            pbar.set_description(f"Loss={mae/size:.4f}")

        return mae / size

    def optim_inputs(self):
        self.to(self.device)
        self.eval()

        x = torch.FloatTensor(self.hp.sample_size, self.inp_dim).uniform_(-1.0, 1.0).to(self.device)
        x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.01)

        for _ in (pbar := tqdm(range(self.hp.search_step))):
            optimizer.zero_grad()

            y_pred = self(x)
            loss = torch.mean(y_pred)
            loss.backward()

            optimizer.step()

            pbar.set_description(f'avg-min={torch.mean(torch.exp(y_pred)).item():.4f}')

        return x.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
