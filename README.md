# Getting Started

-   Clone this repository

```
cd ~/Atdp_Walker
poetry install
```

```sh
# Login to cloud
ssh username@server
# Git pull latest code (clone if first time)
git pull https://github.com/rl-squad/atdp_walker.git
# Create docker image called atdp from Dockerfile
hare build -t username/atdp ./atdp_walker
# Create docker container from image, mounts volume and opens bash terminal
hare run -v $(pwd)/out:/app/out -it username/atdp
# Starts agent training, replace filename to desired output file name and agent training script as needed
OUT=filename poetry run python train_agent_template.py

# Detach from running container with escape sequence Ctrl-P and Ctrl-Q
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

# There should be a .npy file in the ./out directory wherever you ran the command: hare run -v $(pwd)/out:/app/out -it username/atdp

# Logout from cloud with Ctrl-D

# Copy data from ./out directory in the cloud to your local machine from wherever you run the command
rsync -uav username@server:~/out .

# Plot results for comparison on the same graph using python utils/plot_npy_files.py which grabs all .npy files from the directory which the command ran in
```