#!/bin/bash

export DOCKER_USER_NAME
export DOCKER_CONTAINER_NAME
export DOCKER_PROJECT_PATH

docker exec \
        -it \
        --user ${DOCKER_USER_NAME} \
        ${DOCKER_CONTAINER_NAME} \
        bash -c "cd ${DOCKER_PROJECT_PATH}/${DOCKER_USER_NAME}; bash"
