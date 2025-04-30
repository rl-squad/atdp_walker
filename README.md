# Deep Reinforcement Learning + [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/)

We train a 2D agent to walk using DRL methods (DDPG, TD3, SAC) with further optimisations (Prioritised Experience Replay, NoisyNetworks).

## Getting Started

- Clone this repository
```sh
git clone https://github.com/rl-squad/atdp_walker.git
```
- Install [poetry](https://python-poetry.org)

```sh
cd ~/Atdp_Walker
# install dependencies
poetry install
```

- How to run our experiments

```sh
# Comparing how SAC performs when using either:
# • a Uniform Replay Buffer
# • a Prioritised Experience Replay Buffer
poetry run python train_sac_prioritised.py

# Comparing which of the 3 improvements from DDPG to TD3
# contributed the most to performance gains in this environment
poetry run python train_td3_ablation.py

# Our final agent trained for 6m steps after observing the results
poetry run python train_final_agent.py

# Results are stored in ~/out/ after experiments complete

# Plots the results, scraping the ~/out/ directory for
# files output from the experiments
poetry run python utils/plot_*

# To manually run training for a single agent,
# replacing filename with desired output filename
OUT=filename poetry run python train_td3.py
```

## Cloud training workflow

```sh
# Login to cloud
ssh username@server
# Git pull latest code (clone if first time)
git pull https://github.com/rl-squad/atdp_walker.git
# Create docker image called atdp from Dockerfile
hare build -t username/atdp ./atdp_walker

# Create docker container from image, mounts volume and opens bash terminal
hare run -v $(pwd)/out:/app/out -it username/atdp

        # Alternatively, to run with GPUs:
        # Read https://hex.cs.bath.ac.uk/wiki/docker/Running-with-GPUs.md
        # Example command:
        hare run --gpus device=0 -v $(pwd)/out:/app/out -it username/atdp

        # Displays GPUs reserved for this running container
        nvidia-smi

# Starts agent training, replace filename to desired output file name
# and agent training script as needed
OUT=filename poetry run python train_agent_template.py

# Detach from running container with escape sequence Ctrl-P then Ctrl-Q
# The container will continue running until training has finished

########################################
# Returning to check training progress #
########################################

# Show all containers
hare ps -a
# Reattach to container (replace id with CONTAINER ID)
hare attach id
# Use the escape sequence CTRL-P then CTRL-Q to detach without killing the process

##############################
# Once training has finished #
##############################

# There should be a .npy file in the ./out directory wherever you ran the command:
# hare run -v $(pwd)/out:/app/out -it username/atdp

# Logout from cloud with Ctrl-D

# Copy data from ./out directory in the cloud to your local machine
# from wherever you run the command
rsync -uav username@server:~/out .

# Plot results for comparison on the same graph
# which grabs all .npy files from the directory which the command ran in
python utils/plot_npy_files.py
```
