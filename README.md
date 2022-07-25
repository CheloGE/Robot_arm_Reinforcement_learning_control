# Robot Arm Reinforcement learning control

The goal of this repo is to train an agent in the form of a 2-joint robot arm and maintain its position at the target location for as many time steps as possible.

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)


## Environment setup

All commands below must be run in the project's folder `cd <path_to_the_folder_with_this_project>`

### First lets donwload all required files

* Make sure to download the environment builded in unity with ml_agents extension

    For linux:
    1. A single robot-arm problem scenario: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    2. Multiple robot-arms problem scenario: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)

### Install dependencies with docker

1. Build Dockerfile

    `docker build -t unity_ml_agents:pytorch .`

2. Create container

    ``docker run --name mlagents_unity -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/project -it --env QT_X11_NO_MITSHM=1 --device /dev/dri --privileged --gpus all unity_ml_agents:pytorch``

3. Everytime we want to run container

    `docker start door_detection_ML`

    `docker exec -it door_detection_ML bash`



