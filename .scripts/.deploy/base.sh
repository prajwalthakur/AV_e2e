#!/bin/bash
# base.sh (or wherever run_docker lives)
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_NAME="${BASE_NAME:-e2e}"
IMAGE_NAME="${IMAGE_NAME:-${BASE_NAME}}"
CONTAINER_NAME="${CONTAINER_NAME:-${BASE_NAME}}"

run_docker() {
    # parse args: everything before '--' -> docker options, after '--' -> command inside container
    local -a docker_opts=()
    local -a cmd=()
    local saw_sep=0

    for arg in "$@"; do
        if [ "$arg" = "--" ]; then
            saw_sep=1
            continue
        fi
        if [ $saw_sep -eq 0 ]; then
            docker_opts+=("$arg")
        else
            cmd+=("$arg")
        fi
    done

    # default command if none provided
    if [ ${#cmd[@]} -eq 0 ]; then
        cmd=(bash)
    fi

    # remove any previous container with same name to avoid conflicts
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    # AWS creds for s5cmd, if present on the host
    local -a aws_opts=()
    if [ -d "${HOME}/.aws" ]; then
        aws_opts+=(-v "${HOME}/.aws:/root/.aws:ro")
    fi
    [ -n "${AWS_PROFILE:-}" ] && aws_opts+=(-e "AWS_PROFILE=${AWS_PROFILE}")
    [ -n "${AWS_REGION:-}" ] && aws_opts+=(-e "AWS_REGION=${AWS_REGION}")

    docker run -it \
        --name "${CONTAINER_NAME}" \
        "${aws_opts[@]}" \
        "${docker_opts[@]}" \
        "${IMAGE_NAME}" \
        "${cmd[@]}"
}

stop_docker() 
{
    docker stop ${CONTAINER_NAME} && rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}