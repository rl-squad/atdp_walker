# ATDP Walker PPO Trainer

This project helps you train reinforcement learning (RL) agents in the `Walker2d-v5` environment using different libraries that support the PPO algorithm.

## Getting Started

### Installation
1. Clone the repository:
```bash
https://github.com/rl-squad/atdp_walker.git
```
2. Install the required packages using Poetry:
```bash
cd atdp_walker
poetry install
```

### Train with Stable-Baselines3
To start training using Stable-Baselines3 PPO, run:
```bash
poetry run python main.py train --lib stable --job-id run001 --job-description "First SB3 PPO run"
```

### train with CleanRLTrainer
 poetry run python main.py train --lib cleanrl --job-id test_cleanrl_01 --job-description "REINFORCE baseline with simple MLP"
 

After training, you'll find your results in:
- `results/results_stable/training_log.csv` (episode rewards and losses)
- `results/results_stable/walker2d_ppo.zip` (saved model)

### Supported Libraries (Trainers)
You can switch libraries using the `--lib` option. Trainers currently available or planned:
- `stable`: Stable-Baselines3
- `cleanrl`: CleanRL (basic PyTorch PPO)
- `rllib`: Ray RLlib
- `garage`: RL Garage
- `custom`: PPO from scratch


### Project Structure
```
atdp_walker/
├── train.py               # Contains all trainer classes
├── main.py                # Main file to run training
├── results/               # Output logs and saved models
│   └── results_stable/    # Example: results from SB3 training
├── pyproject.toml         # Dependencies and config for Poetry
└── ...
```

