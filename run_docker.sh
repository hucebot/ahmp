#!/bin/bash

docker run \
    --rm \
    -it \
    --net=host \
    -e DISPLAY \
    --privileged  \
    -v ${HOME}/.Xauthority:/home/robot/.Xauthority \
    -v "$(pwd)":/home/robot/code \
    --entrypoint /bin/bash \
    ahmp
