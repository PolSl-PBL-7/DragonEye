#!/usr/bin/env bash

# first, determine if user running this script is a root
# if not, then set sudo for apt commands
if [[ "$(whoami)" == "root" ]]
then
    SUDO=""
else
    SUDO="sudo"
fi

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
$SUDO mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
$SUDO dpkg -i cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
rm -f cuda-repo-wsl-ubuntu-11-2-local_11.2.0-1_amd64.deb
$SUDO apt-key add /var/cuda-repo-wsl-ubuntu-11-2-local/7fa2af80.pub
$SUDO apt-get update
$SUDO apt-get -y install cuda
