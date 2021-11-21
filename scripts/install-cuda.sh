#!/usr/bin/env bash

# first, determine if user running this script is a root
# if not, then set sudo for apt commands
if [[ "$(whoami)" == "root" ]]
then
    SUDO=""
else
    SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
        linux-headers-amd64 \
        software-properties-common

$SUDO apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
$SUDO add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
$SUDO add-apt-repository contrib
$SUDO add-apt-repository non-free

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends cuda-11-2
