# NP-Attack: Neural Predictor for Black-Box Adversarial Attacks on Speech Recognition

#### Set-up

```python
# optional: create a new environment
conda create -n npattack
conda activate npattack

# install dependencies
conda install pytorch torchaudio -c pytorch
pip install -r requirements.txt
```

#### Usage

To run the NP-attack, modify `conf/config.yaml` to specify the audio file, query limit, perturbation budget etc. Then:

```python
python main.py
```

Alternatively, the configuration can be overwritten from the command line (as one would with hydra).


