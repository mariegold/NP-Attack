# NP-Attack: Neural Predictor for Black-Box Adversarial Attacks on Speech Recognition

## Requirements

Make sure you are running Python 3.8 or later.

```python
pip install -r requirements.txt
```

## Usage

To run the NP-attack, either modify `conf/config.yaml` directly, or via command line:

```python
python main.py [wave_file=<wave_file>] [budget=<budget>] [eps_perb=<eps_perb>]
```
For more information, refer to [Hydra documentation](https://hydra.cc/docs/intro/).

## Benchmark
To evaluate the attack on the benchmark, follow these steps:

*TODO: describe how to get the benchmark data and run the script*

## Reference
```
@article{biolkova2022npattack,
  title={NP-Attack: Neural Predictor for Black-Box Adversarial Attacks on Speech Recognition},
  author={Biolková Marie and Nguyen Bac},
  journal={arXiv preprint},
  year={2022}
}
```
