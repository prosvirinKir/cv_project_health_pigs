#!/bin/bash

export DOCKER_IMAGE_NAME
export DOCKER_IMAGE_VERSION
export DOCKER_USER_NAME
export DOCKER_USER_PASSWORD
export DOCKER_PROJECT_PATH

yellow=`tput setaf 3`
green=`tput setaf 2`
violet=`tput setaf 5`
reset_color=`tput sgr0`

ARCH="$(uname -m)"

main () {

    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then
        file="docker/Dockerfile.${ARCH}"
    else
        echo "There is no Dockerfile for ${yellow}${ARCH}${reset_color} arch"
    fi

    echo "Building image for ${yellow}${ARCH}${reset_color} arch. from Dockerfile: ${yellow}${file}${reset_color}"
    
    docker build . \
	-f ${file} \
	-t ${ARCH}no-ros/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} \
	--build-arg UID=${DOCKER_UID} \
	--build-arg PW=${DOCKER_USER_PASSWORD} \
	--build-arg USER=${DOCKER_USER_NAME} \
    	--build-arg PROJECT_PATH=${DOCKER_PROJECT_PATH}
}

main "$@"; exit;

