import torch
from speechbrain.pretrained import EncoderDecoderASR


class ASR:
    def __init__(self, hp, device) -> None:
        self.hp = hp
        self.model = EncoderDecoderASR.from_hparams(
            source=hp.source,
            savedir=hp.savedir,
            run_opts={"device": str(device)},
            freeze_params=True
        )
        self.b_len = torch.tensor([1.]).to(device)

    def transcribe(self, wave):
        wave = torch.tensor(wave).to(self.model.device).unsqueeze(0)
        pred = self.model.transcribe_batch(wave, self.b_len)[0][0]
        return pred
