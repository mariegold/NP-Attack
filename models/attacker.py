import librosa as lr
import numpy as np
import torch
from jiwer import wer
from pathlib import Path

from .model import ASR
from .predictor import Predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NPAttacker:
    def __init__(self, hp):
        self.hp = hp
        self.asr = ASR(hp.asr, device)
        self.rng = np.random.RandomState(hp.seed)
        self.norm = np.inf if hp.norm == 'inf' else None
        self.sample_id = Path(hp.wave_file).stem
        self.pre = Predictor(hp.strategy.predictor, device)
        
    def attack(self, wave_file):
        """ Finds a minimal adversarial perturbation to an audio sample.

        Args:
            wav_file (str): Path to the original audio file.
        
        Returns:
            np.array: Adversarial example.        
        """
        wave = lr.load(wave_file, sr=self.hp.sr)[0].astype(np.float32)

        # set the ground-truth
        self.wave = wave
        self.label = self.asr.transcribe(wave)
        self.dim_attack = len(wave)
        self.num_queries = 0
        hp = self.hp
        success = False

        # warmup for the predictor
        x_train = self.rng.uniform(-1.0, 1.0, size=(hp.strategy.n_points, self.dim_attack))
        y_train = np.full((hp.strategy.n_points, 1), np.inf, dtype=np.float32)

        curr_queries = self.num_queries

        for i in range(hp.strategy.n_points):
            d = self.b_dist(x_train[i,:])
            if self.num_queries < hp.budget:
                curr_queries = self.num_queries
                y_train[i] = d
                print(f"queries={self.num_queries}"
                    f"\tperb={y_train.min():.4f}")
            else: 
                break
            if d <= hp.eps_perb:
                break

        # search the best theta
        while self.num_queries < hp.budget and y_train.min() > hp.eps_perb:
            self.pre.fit(x_train, y_train)
            x_new, y_pred = self.pre.optim_inputs()

            y_new = np.full((hp.strategy.predictor.sample_size, 1), np.inf, dtype=np.float32)
            for i in range(hp.strategy.predictor.sample_size):
                y_new[i] = self.b_dist(x_new[i,:])
                if y_new[i] <= hp.eps_perb:
                    success = True # avoid adding random points
                    break

            test = np.mean((np.log(y_new) - y_pred)**2)

            x_rand = self.rng.uniform(-1.0, 1.0, size=(2, x_train.shape[-1]))
            y_rand = np.full((2, 1), np.inf, dtype=np.float32)
            if not success:   # add a few random candidates
                for i in range(2):
                    y_rand[i] = self.b_dist(x_rand[i,:])
                    if y_rand[i] <= hp.eps_perb:
                        break

            if self.num_queries <= hp.budget:
                curr_queries = self.num_queries

                x_train = np.concatenate((x_train, x_new, x_rand))
                y_train = np.concatenate((y_train, y_new, y_rand))
                
                print(f"queries={curr_queries}"
                      f"\tperb={y_train.min():.4f}",
                      f"\tL-test={test:.4f}")
            else:
                break

        index = np.argmin(y_train.flatten())
        theta = self.get_theta(x_train[index])
        self.num_queries = curr_queries

        return  self.get_wave(theta, y_train[index][0])

    def get_theta(self, x):
        """ Normalizes the perturbation. 
        
        Args: 
            x (np.array): Perturbation direction.
        
        Returns:
            np.array: Normalized direction.
        """
        return x / np.linalg.norm(x, ord=self.norm)

    def get_wave(self, theta, mag):
        """ Applies a perturbation to the original sample.
        
        Args: 
            theta (np.array): Perturbation direction, normalized.
            mag (float): Perturbation magnitude.
        
        Returns:
            np.array: Perturbed sample clipped to range.
        """
        return np.clip(self.wave + theta*mag, -1.0, 1.0).astype(np.float32)

    def query(self, theta, mag):
        """ Queries the model with a perturbed sample.
        
        Args: 
            theta (np.array): Perturbation direction, normalized.
            mag (float): Perturbation magnitude.
        
        Returns:
            bool: Attack success.
        """
        self.num_queries += 1
        wave = self.get_wave(theta, mag)
        pred = self.asr.transcribe(wave.astype(np.float32))

        return wer(self.label, pred) > self.hp.min_wer

    def b_dist(self, x, tol=1e-4, incr=0.1):
        """Returns the minimum distance to the decision boundary.

        Args:
            x (np.ndarray): Search direction, normalized.
            tol (float): Precision tolerance for binary search.
                Defaults to 1e-4.
            incr (float): Step size for finding the binary search bound.
                Defaults to 0.1.

        Returns:
            float: Minimum distance.
        """
        upper=self.hp.upper_lim
        theta = self.get_theta(x)
        hi = incr
        while not self.query(theta, hi) and hi < upper:
            hi += incr

        if hi >= upper:
            return 1e9

        lo = hi - incr
        while hi - lo > tol and hi < upper:
            mid = (lo + hi) / 2
            if self.query(theta, mid):
                hi = mid
            else:
                lo = mid

        return hi

    def eval_attack(self, ae):
        """ Exports the results of the attack.
        
        Args: 
            ae (np.array): Adversarial example.
        """
        try:
            diff = ae - self.wave
        except NameError:
            "Run the attack first!"

        linf = np.linalg.norm(diff, ord=np.inf)
        orig_linf = np.linalg.norm(self.wave, ord=np.inf)
        snr_linf = 20*np.log10(orig_linf/linf)

        out_list = [self.sample_id, str(self.num_queries),
                    str(linf), str(snr_linf)]
        with open(self.hp.out_file, 'a') as f:
            f.write(",".join(out_list)) 
            f.write('\n')

