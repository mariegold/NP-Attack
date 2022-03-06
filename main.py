import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from models.attacker import NPAttacker


@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    torch.manual_seed(args.seed)

    model = NPAttacker(args)
    wave = model.attack(args.wave_file)

    if wave is not None:
        # retest the output
        out = args.strategy.name + str(args.seed) + '.wav'
        sf.write(out, wave, args.sr)
        print(model.asr.model.transcribe_file(out))

        if args.out:
            model.eval_attack(wave)

if __name__ == '__main__':
    main()
