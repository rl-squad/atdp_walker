# Getting Started

-   Clone this repository

```
cd ~/Atdp_Walker
poetry install
```

```sh
# Create docker image called atdp from current directory
docker build -t atdp .
# Create docker container and run in interactive, terminal and detached mode
docker run -itd atdp

# The container will continue running until training has finished

########################################
# Returning to check training progress #
########################################

# Show all containers
docker ps
# Reattach to container (replace id with CONTAINER ID)
docker attach id
# Use the escape sequence CTRL-P then CTRL-Q to detach without killing the process (doesn't work for me in VSCode terminal)
```