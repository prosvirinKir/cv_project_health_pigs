#!/bin/bash

export DOCKER_IMAGE_VERSION
export DOCKER_IMAGE_NAME
export DOCKER_CONTAINER_NAME
export DOCKER_USER_NAME
export DOCKER_PROJECT_PATH

yellow=`tput setaf 3`
green=`tput setaf 2`
violet=`tput setaf 5`
reset_color=`tput sgr0`

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

desktop_start() {
    docker run
	-itd \
        --gpus all \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name ${DOCKER_CONTAINER_NAME} \
        -p 1025:1025 \
        --net "host" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $workspace_dir/:/home/${DOCKER_USER_NAME}/:rw \
        ${ARCH}no-ros/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}
}

main () {

    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        docker rm ${DOCKER_CONTAINER_NAME};
    fi

    ARCH="$(uname -m)"

    if [ "$ARCH" = "x86_64" ]; then 
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        echo "There is no code for ${yellow}${ARCH}${reset_color} arch"
    fi

    docker exec \
	-it \
	--user ${DOCKER_USER_NAME} \
        ${DOCKER_CONTAINER_NAME} \
        bash -c "cd ${DOCKER_PROJECT_PATH}/${DOCKER_USER_NAME}; bash"
}

main;

