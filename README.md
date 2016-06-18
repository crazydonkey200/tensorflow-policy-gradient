# tensorflow-policy-gradient

Still under construction...

## Dependencies
- Python 2.7
- TensorFlow >= 0.8.0
- NumPy >= 1.10.0
- openai gym
- matplotlib

## Quick try
Run
```bash
python gym_experiment.py
```
to train a softmax policy (without bias) using vanilla policy gradient on [CartPole task](https://gym.openai.com/envs/CartPole-v0). You can see that the return is stochastically increasing until it reaches the maximum (200).
